"""Media-to-candidates pipeline, with inspectable automatic event proposals."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .physics import ideal_curve
from .search import spectrogram, run_search, geojson
from .archive import download_day


def prepare_media(media, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        probe = subprocess.run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(media)],
                               check=True, capture_output=True, text=True)
        info = json.loads(probe.stdout)
        out = directory/'audio.wav'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(media),'-vn','-ac','1','-ar','8000',
                        '-c:a','pcm_s16le',str(out)], check=True, capture_output=True)
    except FileNotFoundError as e:
        raise ValueError('Install FFmpeg (ffmpeg and ffprobe) to analyze video/compressed audio') from e
    except subprocess.CalledProcessError as e:
        raise ValueError('FFmpeg could not decode the input media') from e
    tags = [info.get('format', {}).get('tags', {})]+[s.get('tags', {}) for s in info.get('streams', [])]
    created = next((t['creation_time'] for t in tags if 'creation_time' in t), None)
    return out, created


def detect_events(audio, max_events=2):
    """Template-bank proposals. Stationary hum rejected by curved vs flat contrast.

    This is a heuristic, not a universal source separator. Persist measurements so
    users can inspect/override them before trusting a real-world conclusion.
    """
    f, t, image = spectrogram(audio)
    if t[-1]-t[0] < 45:
        raise ValueError('Automatic detection needs at least 45 seconds; use manual event windows for shorter clips')
    sample = RegularGridInterpolator((t,f), image.T, bounds_error=False, fill_value=-5.)
    f0s = np.arange(60, 251, 2.)
    proposals = []
    for center in np.arange(t[0]+22, t[-1]-22, 8):
        times = np.arange(center-20, center+20, .5)
        tt = np.broadcast_to(times[:,None], (len(times),len(f0s)))
        flat = sample(np.stack([tt, np.broadcast_to(f0s, tt.shape)], axis=-1)).mean(axis=0)
        best = None
        for speed in (45., 70., 100., 140.):
            for distance in (500., 1200., 2500., 5000.):
                ratios = ideal_curve(times, 1., speed, distance, center)
                predicted = ratios[:,None]*f0s
                scores = sample(np.stack([tt,predicted], axis=-1)).mean(axis=0)
                contrast = scores-np.maximum(0, flat)*.5
                i = np.argmax(contrast)
                if best is None or contrast[i] > best[0]:
                    best = (float(contrast[i]), float(f0s[i]), speed, distance)
        score, fundamental, speed, distance = best
        proposals.append(dict(detection_score=score,start_s=float(center-20),end_s=float(center+20),
                              audio_cpa_s=float(center), f0_hz=list(np.arange(fundamental-3,fundamental+3.1,1)),
                              speed_mps=[max(10,speed*.55), min(250,speed*1.5)],
                              slant_m=[max(50,distance*.35), min(15000,distance*2.5)]))
    selected = []
    for event in sorted(proposals,key=lambda e:e['detection_score'],reverse=True):
        if event['detection_score'] < 1.:
            continue
        if all(abs(event['audio_cpa_s']-e['audio_cpa_s']) > 55 for e in selected):
            selected.append(event)
            if len(selected) == max_events:
                break
    if not selected:
        raise ValueError('No confident Doppler events detected. Inspect the spectrogram and provide manual events')
    from scipy.optimize import differential_evolution
    for event in selected:
        times = np.arange(event['start_s'], event['end_s'], .25)
        center = event['audio_cpa_s']
        fundamental = float(np.mean(event['f0_hz']))
        def objective(p):
            hz, speed, distance, cpa = p
            predicted = ideal_curve(times,hz,speed,distance,cpa)
            return -float(np.mean(sample(np.c_[times,predicted])))
        fit = differential_evolution(objective,
            [(max(30,fundamental-25),fundamental+25),(20,200),(100,8000),(center-12,center+12)],
            seed=7,popsize=12,maxiter=80,tol=.005,polish=True)
        hz,speed,distance,cpa = map(float,fit.x)
        event.update(f0_hz=[float(v) for v in np.arange(hz-2,hz+2.1,.5)],
                     speed_mps=[max(10,speed*.6),min(250,speed*1.6)],
                     slant_m=[max(50,distance*.4),min(15000,distance*2.5)],
                     audio_cpa_s=cpa)
    return sorted(selected, key=lambda e:e['audio_cpa_s'])


def analyze(media, bounds, directory, day=None, recorded_at=None, config=None, max_gb=8):
    directory = Path(directory)
    audio, created = prepare_media(media, directory)
    recording = recorded_at or created
    utc = None
    if recording:
        parsed = datetime.fromisoformat(recording.replace('Z','+00:00'))
        if parsed.tzinfo is None:
            raise ValueError('Recording time must include UTC Z or timezone offset')
        utc = parsed.timestamp()
        day = parsed.astimezone(timezone.utc).date().isoformat()
    if not day:
        raise ValueError('Media has no recording date. Supply --date YYYY-MM-DD or --recorded-at ISO-8601')
    config = dict(config or {})
    config.setdefault('bounds', bounds)
    config.setdefault('spacing_m', 250)
    config.setdefault('auto_terrain', True)
    config.setdefault('terrain_cache', str(directory/'terrain'))
    if not config.get('events'):
        config['events'] = detect_events(audio)
        config.setdefault('time_shifts_s', list(range(-8,9,2)))
        config.setdefault('offset_tolerance_s', 20)
    if utc is not None:
        for event in config['events']:
            event['utc_cpa_s'] = utc+event['audio_cpa_s']
    elif len(config['events']) < 2:
        raise ValueError('Only one event detected: provide recording UTC time or manually specify two events')
    (directory/'search.json').write_text(json.dumps(config, indent=2))
    time_bounds = None
    if utc is not None:
        _, times, _ = spectrogram(audio)
        time_bounds = [utc-180, utc+float(times[-1])+180]
    flights = download_day(day, config['bounds'], directory/'archive', max_gb, time_bounds, cache_directory=config.get('archive_cache'))
    # Include adjoining UTC dates whenever propagation/window padding crosses midnight.
    if time_bounds:
        dates = {datetime.fromtimestamp(t,timezone.utc).date().isoformat() for t in time_bounds}-{day}
        for extra in sorted(dates):
            other = download_day(extra, config['bounds'], directory/extra, max_gb, time_bounds, cache_directory=config.get('archive_cache'))
            with open(flights,'a') as dest, open(other) as source:
                next(source)
                dest.write(source.read())
    result = run_search(audio, flights, config)
    result['measurement_source'] = 'Automatic template proposals unless events supplied in configuration'
    result['recording_time_source'] = 'user' if recorded_at else ('media metadata (verify independently)' if created else 'date only')
    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (directory/'result.geojson').write_text(json.dumps(geojson(result),allow_nan=False))
    return result

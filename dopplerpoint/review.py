"""Read-only, bounded-window review of explicitly selected local analysis files."""
import io
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from .physics import ideal_curve, ecef, read_tracks
from .search import spectrogram, terrain_sampler


class ReviewSession:
    def __init__(self, audio, report=None, results=None, config=None, flights=None):
        self.audio = Path(audio)
        self.rate, data = wavfile.read(self.audio, mmap=True)
        self.duration = len(data)/self.rate
        self.config = json.loads(Path(config).read_text()) if config else None
        detection = json.loads(Path(report).read_text()) if report else {}
        self.events = detection.get('proposals') or detection.get('accepted_events') or (self.config or {}).get('events', [])
        if not self.events:
            raise ValueError('Provide a detection report or a search config containing events')
        # Stable indices, including rejected proposals, survive review exports.
        self.events = sorted(self.events, key=lambda e:e['audio_cpa_s'])
        if 'accepted_events' in detection:
            selected = {(e['start_s'],e['end_s'],e['audio_cpa_s']) for e in detection['accepted_events']}
            for event in self.events:
                event['detector_selected'] = (event['start_s'],event['end_s'],event['audio_cpa_s']) in selected
        self.tracks = {t.flight_id:t for t in read_tracks(flights)} if flights else {}
        self.elevation = terrain_sampler(self.config) if self.config and flights else None
        self.candidates, self.kind, self.map_note = [], 'none', 'No coordinate results loaded.'
        if results:
            raw = json.loads(Path(results).read_text())
            if 'results' in raw:
                self.kind = 'scored_observer_candidates'
                self.candidates = raw['results']
                self.map_note = 'Scored observer hypotheses. Scores are relative; no location is confirmed.'
            elif 'candidates' in raw:
                self.kind = 'coarse_aircraft_samples'
                self.candidates = raw['candidates']
                self.map_note = 'Unscored aircraft sample positions from a coarse screen. These are NOT observer matches. Showing the saved sample only.'
            elif raw.get('type') == 'FeatureCollection':
                self.kind = 'imported_points'
                for f in raw['features']:
                    if (f.get('geometry') or {}).get('type') == 'Point':
                        lon,lat,*_ = f['geometry']['coordinates']
                        self.candidates.append(dict(f.get('properties') or {},latitude=lat,longitude=lon))
                self.map_note = 'Imported GeoJSON points; interpretation follows the source file.'
            else:
                raise ValueError('Results must contain results, candidates, or GeoJSON point features')
        self.total_candidates = len(self.candidates)
        self.candidates = self.candidates[:10000]
        for row in self.candidates:
            if not (-90 <= float(row['latitude']) <= 90 and -180 <= float(row['longitude']) <= 180):
                raise ValueError('Invalid coordinate in results')

    def manifest(self):
        return dict(events=self.events, duration_s=self.duration, config=self.config,
                    candidates=self.candidates, candidate_kind=self.kind, map_note=self.map_note,
                    total_candidates=self.total_candidates, displayed_candidates=len(self.candidates))

    def bounds(self, index, padding=30):
        event = self.events[index]
        if not 0 <= padding <= 120:
            raise ValueError('Padding must be 0–120 seconds')
        return max(0,event['start_s']-padding),min(self.duration,event['end_s']+padding)

    def clip(self, index, padding=30):
        start,end = self.bounds(index,padding)
        rate,data = wavfile.read(self.audio,mmap=True)
        output = io.BytesIO()
        wavfile.write(output,rate,data[int(start*rate):int(end*rate)])
        return output.getvalue()

    def window(self,index,candidate=None,padding=30,fmin=30,fmax=700,nfft=16384):
        if nfft not in (4096,8192,16384,32768) or not 0 <= fmin < fmax <= min(4000,self.rate/2):
            raise ValueError('Invalid frequency range or resolution')
        start,end = self.bounds(index,padding)
        f,t,z = spectrogram(self.audio,nfft=nfft,hop=max(256,nfft//16),start_s=start,end_s=end)
        band = (f>=fmin)&(f<=fmax)
        if band.sum()<2:
            raise ValueError('Frequency band is too narrow')
        stride = max(1,int(np.ceil(len(t)/1600)))
        times = np.linspace(start,end,1200)
        event = self.events[index]
        overlays = []
        parameters = event.get('fitted_parameters')
        if parameters:
            hz = ideal_curve(times,parameters['f0_hz'],parameters['speed_mps'],parameters['slant_m'],event['audio_cpa_s'])
            overlays.append(dict(label='Ideal fit',time=times.tolist(),frequency_hz=hz.tolist(),color='#54e3ff'))
        if event.get('trace'):
            trace = np.asarray(event['trace'])
            overlays.append(dict(label='Measured trace',time=trace[:,0].tolist(),frequency_hz=trace[:,1].tolist(),color='#ffb861'))
        note = 'Load scored results, the corresponding search config and flight CSV to overlay actual flight geometry.'
        if candidate is not None and self.kind == 'scored_observer_candidates' and self.config and self.tracks:
            row = self.candidates[candidate]
            configured = self.config['events']
            matches = [i for i,e in enumerate(configured) if abs(e['audio_cpa_s']-event['audio_cpa_s'])<.01]
            if matches:
                j = matches[0]
                track = self.tracks.get(row['flights'][j])
                f0 = row['f0_hz'][j]
                if track is not None and f0 is not None:
                    observer = ecef(row['latitude'],row['longitude'],self.elevation(row['latitude'],row['longitude']))
                    arrival,ratio = track.curve(observer,self.config.get('sound_speed_mps',343))
                    audio_time = arrival+row['clock_offset_s']
                    keep = (audio_time>=start)&(audio_time<=end)
                    overlays.append(dict(label='Flight geometry',time=audio_time[keep].tolist(),frequency_hz=(ratio[keep]*f0).tolist(),color='#ff72bb'))
                    note = 'Pink: actual flight geometry with the saved shared clock and source frequency. Cyan: ideal detection fit.'
            else:
                note = 'This detection was not included in the selected scored search config.'
        return dict(spectrogram=dict(f=f[band].tolist(),t=t[::stride].tolist(),z=z[band,::stride].tolist()),
                    overlays=overlays,event=event,start_s=start,end_s=end,model_note=note)

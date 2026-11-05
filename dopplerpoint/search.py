"""Bounded coarse-to-fine candidate filtering and locked multi-event scoring."""
import itertools
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, resample_poly
from scipy.interpolate import RegularGridInterpolator
from .physics import ecef, read_tracks


def spectrogram(path, nfft=4096, hop=512):
    rate, audio = wavfile.read(path)
    if audio.dtype.kind == 'u':
        audio = audio.astype(float)-128
    else:
        audio = audio.astype(float)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError("Audio must contain finite nonzero samples")
    # Bound FFT memory for long high-rate files; low aircraft tones survive.
    if rate > 8000:
        from math import gcd
        d = gcd(rate, 8000)
        audio = resample_poly(audio, 8000//d, rate//d)
        rate = 8000
    nfft = min(nfft, len(audio))
    hop = min(hop, nfft)
    f, t, z = stft(audio, rate, nperseg=nfft, noverlap=nfft-hop, boundary=None)
    power = 20*np.log10(np.maximum(abs(z), 1e-15))
    # Per-frequency robust whitening reduces stationary hum's influence.
    med = np.median(power, axis=1, keepdims=True)
    scale = np.maximum(np.median(abs(power-med), axis=1, keepdims=True)*1.4826, 3.)
    return f, t, np.clip((power-med)/scale, -5, 15)


def terrain_sampler(config):
    if config.get('auto_terrain'):
        from .archive import download_terrain
        return download_terrain(config['bounds'], config.get('terrain_cache', 'output/terrain'))
    if not config.get('dem'):
        elevation = float(config.get('elevation_m', 0))
        return lambda lat, lon: elevation
    import rasterio
    from rasterio.warp import transform
    dataset = rasterio.open(config['dem'])
    if dataset.crs is None:
        raise ValueError("DEM must declare a CRS")
    def sample(lat, lon):
        x, y = transform('EPSG:4326', dataset.crs, [lon], [lat])
        v = next(dataset.sample([(x[0], y[0])], masked=True))[0]
        if np.ma.is_masked(v) or not np.isfinite(v):
            raise ValueError("DEM has no elevation for a candidate")
        return float(v)
    return sample


def grid(bounds, spacing, max_points=100000):
    south, west, north, east = map(float, bounds)
    if not (-89 <= south < north <= 89 and -180 <= west < east <= 180 and spacing >= 10):
        raise ValueError("Use ordered bounds [south, west, north, east] and spacing >= 10 m")
    dy = spacing/111320
    dx = dy/np.cos(np.radians((south+north)/2))
    ny, nx = int(np.ceil((north-south)/dy))+1, int(np.ceil((east-west)/dx))+1
    if nx*ny > max_points:
        raise ValueError(f"Grid has {nx*ny:,} points; narrow bounds or increase spacing (limit {max_points:,})")
    return itertools.product(np.linspace(south, north, ny), np.linspace(west, east, nx))


def _serial_search(audio, flights, config):
    events = config['events']
    c = float(config.get('sound_speed_mps', 343))
    if not 300 <= c <= 380 or not 1 <= len(events) <= 3:
        raise ValueError("Use 1–3 events and sound speed 300–380 m/s")
    f, t, image = spectrogram(audio)
    tracks = read_tracks(flights, float(config.get('max_track_gap_s', 30)))
    if not tracks:
        raise ValueError("No usable flight segments")
    for e in events:
        if not (t[0] <= e['start_s'] < e['end_s'] <= t[-1]):
            raise ValueError("Event windows must lie within spectrogram time coverage")
        if not e['f0_hz'] or min(e['f0_hz']) <= 0:
            raise ValueError("Each event needs positive trial f0_hz values")
    from .compute import SpectralSampler
    sample = SpectralSampler(t, f, image.T, config.get('backend', 'cpu'))
    elevation = terrain_sampler(config)
    shifts = config.get('time_shifts_s', [-2, -1, 0, 1, 2])
    if not shifts:
        raise ValueError("time_shifts_s cannot be empty")
    results = []
    stats = dict(grid_points=0, filtered_pairs=0, scored_pairs=0)
    max_pairs = int(config.get('max_scored_pairs', 100000))

    # Conservative speed-envelope filtering happens once, before per-point geometry.
    envelopes = [(np.linalg.norm(tr.velocity, axis=1).min(), np.linalg.norm(tr.velocity, axis=1).max()) for tr in tracks]
    speed_candidates = [[i for i,tr in enumerate(tracks) if tr.engine.lower() != 'jet'
                         and envelopes[i][1] >= e['speed_mps'][0] and envelopes[i][0] <= e['speed_mps'][1]] for e in events]
    needed = set(itertools.chain.from_iterable(speed_candidates))
    if config.get('silence_radius_m'):
        needed = set(range(len(tracks)))
    # Conservative segment bounding spheres: no segment whose CPA can satisfy
    # the slant threshold is removed, even when the aircraft crosses a cell.
    from scipy.spatial import cKDTree
    spatial_ids, midpoints, half_lengths = [], [], []
    for i in needed:
        tr = tracks[i]
        midpoints.extend((tr.position[:-1]+tr.position[1:])/2)
        half_lengths.extend(np.linalg.norm(np.diff(tr.position,axis=0),axis=1)/2)
        spatial_ids.extend([i]*(len(tr.time)-1))
    tree = cKDTree(np.asarray(midpoints).reshape(-1,3))
    spatial_ids = np.asarray(spatial_ids,dtype=int)
    radius = max([e['slant_m'][1] for e in events]+[config.get('silence_radius_m',0) or 0])+max(half_lengths,default=0)
    def evaluate(lat, lon):
        observer = ecef(lat, lon, elevation(lat, lon))
        nearby = set(spatial_ids[tree.query_ball_point(observer,radius)])
        cpas = {i: tracks[i].cpa(observer,c) for i in nearby}
        eligible = []
        for event, candidates in zip(events, speed_candidates):
            eligible.append([i for i in candidates if i in nearby for arrival,distance,speed in [cpas[i]]
                             if tracks[i].engine.lower() != 'jet'
                             and event['speed_mps'][0] <= speed <= event['speed_mps'][1]
                             and event['slant_m'][0] <= distance <= event['slant_m'][1]
                             and ('utc_cpa_s' not in event or abs(arrival-event['utc_cpa_s']) <= config.get('offset_tolerance_s', 5))])
        order = sorted(range(len(events)), key=lambda i:len(eligible[i]))
        def consistent_pairs(depth=0, chosen=None):
            chosen = {} if chosen is None else chosen
            if depth == len(order):
                yield tuple(chosen[j] for j in range(len(events)))
                return
            j = order[depth]
            for i in eligible[j]:
                identity = tracks[i].flight_id.rsplit(':',1)[0]
                if any(tracks[k].flight_id.rsplit(':',1)[0] == identity for k in chosen.values()):
                    continue
                if any(abs((cpas[i][0]-cpas[k][0])-(events[j]['audio_cpa_s']-events[h]['audio_cpa_s'])) > config.get('offset_tolerance_s',5)
                       for h,k in chosen.items()):
                    continue
                yield from descend(depth,j,i,chosen)
        def descend(depth,j,i,chosen):
            updated = dict(chosen)
            updated[j] = i
            yield from consistent_pairs(depth+1,updated)
        for pair in consistent_pairs():
            base = cpas[pair[0]][0]
            if any(abs((cpas[i][0]-base)-(events[j]['audio_cpa_s']-events[0]['audio_cpa_s'])) > config.get('offset_tolerance_s', 5) for j, i in enumerate(pair)):
                continue
            stats['filtered_pairs'] += 1
            silence = config.get('silence_radius_m')
            if silence and len(pair) > 1:
                margin = config.get('silence_margin_s', 60)
                if any(i not in pair and distance <= silence and base+margin < arrival < cpas[pair[-1]][0]-margin
                       for i, (arrival, distance, _) in cpas.items()):
                    continue
            stats['scored_pairs'] += 1
            if stats['scored_pairs'] > max_pairs:
                raise ValueError("Scoring budget exceeded; narrow bounds or tighten filters")
            curves = [tracks[i].curve(observer, c) for i in pair]
            best = None
            for shift in shifts:
                clock = events[0]['audio_cpa_s']+shift-base
                scores, fundamentals = [], []
                for event, (arrivals, ratios) in zip(events, curves):
                    times = np.arange(event['start_s'], event['end_s'], .25)
                    # All events share one clock; never independently realign event two.
                    model = np.interp(times-clock, arrivals, ratios, left=np.nan, right=np.nan)
                    if not np.isfinite(model).all():
                        scores.append(-5.)
                        fundamentals.append(None)
                        continue
                    fundamentals_array = np.asarray(event['f0_hz'])
                    predicted = model[:,None]*fundamentals_array[None,:]
                    if event.get('trace'):
                        trace = np.asarray(event['trace'], float)
                        target = np.interp(times,trace[:,0],trace[:,1],left=np.nan,right=np.nan)
                        valid = np.isfinite(target)
                        values = -np.mean((predicted[valid]-target[valid,None])**2,axis=0) if valid.any() else np.full(len(fundamentals_array),-1e12)
                    else:
                        harmonics = np.asarray(event.get('harmonics',[1]))
                        hz = predicted[:,:,None]*harmonics[None,None,:]
                        ts = np.broadcast_to(times[:,None,None],hz.shape)
                        values = sample(np.stack([ts,hz],axis=-1)).mean(axis=(0,2))
                    winner = int(np.argmax(values))
                    score, fundamental = float(values[winner]), float(fundamentals_array[winner])
                    scores.append(score)
                    fundamentals.append(fundamental)
                total = float(sum(scores))
                if best is None or total > best['score']:
                    best = dict(latitude=float(lat), longitude=float(lon), score=total,
                                event_scores=scores, f0_hz=fundamentals, clock_offset_s=float(clock),
                                flights=[tracks[i].flight_id for i in pair],
                                delayed_cpa_utc_s=[cpas[i][0] for i in pair])
            results.append(best)
    for lat, lon in config.get('_points', grid(config['bounds'], config.get('spacing_m', 200), config.get('max_grid_points',100000))):
        stats['grid_points'] += 1
        evaluate(lat, lon)
    results.sort(key=lambda r:(-r['score'],r['latitude'],r['longitude'],r['flights']))
    # Refine around the best distinct coarse points; retain every scored result.
    refinement = config.get('refine_spacing_m')
    if refinement:
        coarse = results[:int(config.get('refine_top', 3))]
        seen = set()
        for row in coarse:
            lat, lon = row['latitude'], row['longitude']
            radius = config.get('spacing_m', 200)/111320
            for y, x in grid([max(config['bounds'][0], lat-radius), max(config['bounds'][1], lon-radius/np.cos(np.radians(lat))),
                              min(config['bounds'][2], lat+radius), min(config['bounds'][3], lon+radius/np.cos(np.radians(lat)))], refinement):
                key = (round(y, 8), round(x, 8))
                if key not in seen:
                    seen.add(key)
                    evaluate(y, x)
        results.sort(key=lambda r:(-r['score'],r['latitude'],r['longitude'],r['flights']))
    leaders = {}
    for row in results:
        leaders.setdefault(tuple(row['flights']), row)
    return dict(results=results[:config.get('top_k', 100)], flight_pair_leaders=list(leaders.values()), stats=stats,
                warnings=['Scores rank candidates; they are not location probabilities.',
                          'Silence is an optional audibility assumption, not proof of no aircraft.',
                          'Use compatible vertical datums for DEM and aircraft altitudes.'])


def geojson(result):
    return dict(type='FeatureCollection', features=[dict(type='Feature',
                geometry=dict(type='Point', coordinates=[r['longitude'], r['latitude']]),
                properties={k: v for k, v in r.items() if k not in ('longitude', 'latitude')}) for r in result['results']])


def _chunk_search(args):
    audio, flights, config, points = args
    config = dict(config, workers=1, _points=points, top_k=1000000000)
    config.pop('refine_spacing_m',None)
    return _serial_search(audio,flights,config)


def run_search(audio, flights, config):
    """Spawn independent CPU processes; deterministic reduction and global budgets."""
    import os
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    workers = int(config.get('workers', min(8,os.cpu_count() or 1)))
    if workers < 1:
        raise ValueError('workers must be positive')
    if config.get('backend','cpu') == 'cuda' and workers != 1:
        raise ValueError('Use workers=1 with CUDA to avoid competing GPU contexts')
    if workers == 1:
        return _serial_search(audio,flights,config)
    if config.get('auto_terrain'):
        terrain_sampler(config)  # Populate cache before processes can contend on downloads.
    points = list(grid(config['bounds'],config.get('spacing_m',200),config.get('max_grid_points',100000)))
    size = max(16,int(np.ceil(len(points)/(workers*4))))
    tasks = [(audio,flights,config,points[i:i+size]) for i in range(0,len(points),size)]
    rows, stats, warnings = [],dict(grid_points=0,filtered_pairs=0,scored_pairs=0),[]
    with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),mp_context=multiprocessing.get_context('spawn')) as pool:
        for part in pool.map(_chunk_search,tasks):
            rows.extend(part['results'])
            warnings = part['warnings']
            for key in stats:stats[key] += part['stats'][key]
            if stats['scored_pairs'] > config.get('max_scored_pairs',100000):
                raise ValueError('Global scoring budget exceeded')
    rows.sort(key=lambda r:(-r['score'],r['latitude'],r['longitude'],r['flights']))
    if config.get('refine_spacing_m') and rows:
        refine_points = set()
        for row in rows[:config.get('refine_top',3)]:
            lat,lon = row['latitude'],row['longitude']
            dy = config.get('spacing_m',200)/111320
            dx = dy/np.cos(np.radians(lat))
            b = config['bounds']
            refine_points.update(grid([max(b[0],lat-dy),max(b[1],lon-dx),min(b[2],lat+dy),min(b[3],lon+dx)],config['refine_spacing_m']))
        part = _chunk_search((audio,flights,config,sorted(refine_points)))
        rows.extend(part['results'])
        for key in ('filtered_pairs','scored_pairs'):stats[key] += part['stats'][key]
        if stats['scored_pairs'] > config.get('max_scored_pairs',100000):
            raise ValueError('Global scoring budget exceeded during refinement')
        rows.sort(key=lambda r:(-r['score'],r['latitude'],r['longitude'],r['flights']))
    leaders = {}
    for row in rows:leaders.setdefault(tuple(row['flights']),row)
    stats['workers'] = min(workers,len(tasks))
    return dict(results=rows[:config.get('top_k',100)],flight_pair_leaders=list(leaders.values()),stats=stats,warnings=warnings)

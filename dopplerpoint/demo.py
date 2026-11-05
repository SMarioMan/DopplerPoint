"""Generate freely reusable synthetic audio and flight data, never real locations."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from .physics import Track, ecef


def generate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    origin = 1700000000.
    lat, lon = 40., -100.
    observer = ecef(lat, lon, 0)
    rate = 4000
    audio_time = np.arange(0, 230, 1/rate)
    rng = np.random.default_rng(7)
    audio = rng.normal(0, .015, len(audio_time))
    rows, events = [], []
    for j, (center, speed, f0) in enumerate([(55, 75, 130), (165, 95, 165)]):
        ts = np.arange(center-50, center+51, 2.)
        points = []
        for t in ts:
            # Different headings and a gentle bend distinguish competing observers.
            x = speed*(t-center)
            y = (250 if j == 0 else -400)+.012*x*x/100
            if j:
                x, y = y, x
            plat, plon = lat+y/111320, lon+x/(111320*np.cos(np.radians(lat)))
            alt = 950+1.5*(t-center)
            points.append(ecef(plat, plon, alt))
            rows.append([f'demo-{j}', origin+t, plat, plon, alt, 'prop'])
        track = Track(f'demo-{j}', origin+ts, np.array(points))
        arrival, ratio = track.curve(observer, step=.05)
        freq = np.interp(audio_time, arrival-origin, ratio, left=1, right=1)*f0
        phase = np.cumsum(freq)*2*np.pi/rate
        envelope = np.exp(-((audio_time-center-3)/23)**4)
        audio += .12*envelope*np.sin(phase)+.04*envelope*np.sin(2*phase)
        cpa, distance, velocity = track.cpa(observer)
        events.append(dict(start_s=center-32, end_s=center+35, audio_cpa_s=cpa-origin,
                           f0_hz=[f0-1, f0, f0+1], harmonics=[1, 2],
                           speed_mps=[velocity-12, velocity+12], slant_m=[distance-400, distance+400]))
    wavfile.write(directory/'audio.wav', rate, np.int16(np.clip(audio, -1, 1)*32767))
    with open(directory/'flights.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['flight_id','timestamp','latitude','longitude','altitude_m','engine'])
        w.writerows(rows)
    config = dict(bounds=[lat-.006, lon-.008, lat+.006, lon+.008], spacing_m=250,
                  refine_spacing_m=50, refine_top=2, events=events, time_shifts_s=[-1, 0, 1], offset_tolerance_s=4)
    (directory/'search.json').write_text(json.dumps(config, indent=2))
    return dict(audio=str(directory/'audio.wav'), flights=str(directory/'flights.csv'), config=config,
                truth=dict(latitude=lat, longitude=lon))

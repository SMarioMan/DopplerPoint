"""ADSB.lol public daily archives (ODbL); streamed parsing without tar extraction."""
import csv
from datetime import date, datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request
import numpy as np


def fetch_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'DopplerPoint/0.1'}), timeout=60) as r:
        return json.load(r)


def resolve_day(day):
    day = date.fromisoformat(day)
    repo = f'adsblol/globe_history_{day.year}'
    url = f'https://raw.githubusercontent.com/{repo}/main/PREFERRED_RELEASES.txt'
    with urllib.request.urlopen(url, timeout=60) as r:
        lines = r.read().decode().splitlines()
    marker = 'v'+day.strftime('%Y.%m.%d')+'-'
    matches = [line for line in lines if marker in line]
    if not matches:
        raise ValueError(f'ADSB.lol has no preferred archive for {day}; choose another date or import CSV')
    tag = matches[0].split('/download/')[1].split('/')[0]
    release = fetch_json(f'https://api.github.com/repos/{repo}/releases/tags/{tag}')
    assets = sorted((a for a in release['assets'] if '.tar' in a['name']), key=lambda a:a['name'])
    if not assets:
        raise ValueError('Archive release has no tar assets')
    return dict(day=str(day), provider='ADSB.lol', license='ODbL-1.0', release=release['html_url'],
                bytes=sum(a['size'] for a in assets), assets=assets)


def verified_download(asset, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/asset['name']
    digest = asset.get('digest')
    def valid():
        if not path.exists() or path.stat().st_size != asset['size']:
            return False
        if digest:
            with path.open('rb') as f:
                checksum = hashlib.sha256()
                while chunk := f.read(1024*1024): checksum.update(chunk)
                return 'sha256:'+checksum.hexdigest() == digest
        return True
    if valid():
        return path
    part = path.with_suffix(path.suffix+'.part')
    try:
        with urllib.request.urlopen(asset['browser_download_url'], timeout=120) as r, part.open('wb') as f:
            while chunk := r.read(1024*1024):
                f.write(chunk)
                if f.tell() > asset['size']:
                    raise ValueError('Download exceeds declared asset size')
        part.replace(path)
        if not valid():
            path.unlink(missing_ok=True)
            raise ValueError('Archive size or published SHA-256 verification failed')
    finally:
        part.unlink(missing_ok=True)
    return path


class JoinedFiles(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None
    def readable(self):
        return True
    def readinto(self, buffer):
        while True:
            if self.current is None:
                try:
                    self.current = open(next(self.paths), 'rb')
                except StopIteration:
                    return 0
            n = self.current.readinto(buffer)
            if n:
                return n
            self.current.close()
            self.current = None
    def close(self):
        if self.current:
            self.current.close()
        super().close()


def trace_rows(trace, bounds, time_bounds=None):
    south, west, north, east = bounds
    # Include flight samples outside the observer grid for slant/arrival modeling.
    lat_margin = .18
    lon_margin = .18/max(.02, np.cos(np.radians((south+north)/2)))
    leg = 0
    for row in trace.get('trace', []):
        if len(row) < 8:
            continue
        if row[6] & 2:
            leg += 1
        timestamp = float(trace['timestamp'])+float(row[0])
        if time_bounds and not time_bounds[0] <= timestamp <= time_bounds[1]:
            continue
        if not (south-lat_margin <= row[1] <= north+lat_margin and west-lon_margin <= row[2] <= east+lon_margin):
            continue
        alt = row[10] if len(row) > 10 and isinstance(row[10], (int, float)) else row[3]
        if not isinstance(alt, (int, float)) or row[6] & 1:
            continue
        yield [str(trace['icao'])+f'-leg{leg}', timestamp, row[1], row[2], alt*.3048, 'unknown']


def convert_archive(paths, output, bounds, time_bounds=None):
    count = 0
    with open(output, 'w', newline='') as f, io.BufferedReader(JoinedFiles(paths)) as stream:
        writer = csv.writer(f)
        writer.writerow(['flight_id','timestamp','latitude','longitude','altitude_m','engine'])
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            for member in archive:
                if not member.isfile() or 'trace_full_' not in member.name or not member.name.endswith('.json'):
                    continue
                if member.size > 64*1024*1024:
                    raise ValueError('Unexpectedly large trace member')
                raw = archive.extractfile(member).read()
                if raw.startswith(b'\x1f\x8b'):
                    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                        raw = gz.read(64*1024*1024+1)
                    if len(raw) > 64*1024*1024:
                        raise ValueError('Decompressed trace exceeds limit')
                trace = json.loads(raw)
                for row in trace_rows(trace, bounds, time_bounds):
                    writer.writerow(row)
                    count += 1
    if not count:
        raise ValueError('Archive has no usable aircraft points in requested area/time')
    return count


def download_day(day, bounds, directory, max_gb=8, time_bounds=None, cache_directory=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = resolve_day(day)
    if manifest['bytes'] > max_gb*1e9:
        raise ValueError(f"Daily archive is {manifest['bytes']/1e9:.2f} GB, above {max_gb} GB budget")
    paths = [verified_download(a, Path(cache_directory) if cache_directory else directory/'cache') for a in manifest['assets']]
    output = directory/'flights.csv'
    manifest['points'] = convert_archive(paths, output, bounds, time_bounds)
    manifest['bounds'] = bounds
    manifest['time_bounds'] = time_bounds
    manifest['altitude_note'] = 'Geometric altitude preferred; barometric fallback may introduce vertical error'
    (directory/'provenance.json').write_text(json.dumps(manifest, indent=2))
    return output


def download_terrain(bounds, directory):
    """Public Mapzen/Tilezen elevation tiles in SRTM HGT form; cache by tile."""
    import math
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tiles = {}
    south, west, north, east = bounds
    count = (math.floor(north)-math.floor(south)+1)*(math.floor(east)-math.floor(west)+1)
    if count > 25:
        raise ValueError('Automatic terrain limited to 25 one-degree tiles; narrow bounds or supply a DEM')
    for lat in range(math.floor(south), math.floor(north)+1):
        for lon in range(math.floor(west), math.floor(east)+1):
            name = f"{'N' if lat>=0 else 'S'}{abs(lat):02}{'E' if lon>=0 else 'W'}{abs(lon):03}"
            path = directory/(name+'.hgt.gz')
            if not path.exists():
                url = f'https://elevation-tiles-prod.s3.amazonaws.com/skadi/{name[:3]}/{name}.hgt.gz'
                with urllib.request.urlopen(url, timeout=120) as r:
                    data = r.read(40*1024*1024+1)
                # Validate gzip CRC and expected tile shape before retaining.
                raw = gzip.decompress(data)
                n = int(np.sqrt(len(raw)//2))
                if n not in (1201, 3601) or len(raw) != n*n*2:
                    raise ValueError('Unexpected elevation tile shape')
                path.write_bytes(data)
            raw = gzip.decompress(path.read_bytes())
            n = int(np.sqrt(len(raw)//2))
            tiles[lat, lon] = np.frombuffer(raw, dtype='>i2').reshape(n,n)
    def sample(lat, lon):
        a, b = math.floor(lat), math.floor(lon)
        tile = tiles[a,b]
        y, x = int(round((a+1-lat)*(len(tile)-1))), int(round((lon-b)*(len(tile)-1)))
        value = float(tile[y,x])
        if value <= -32768:
            raise ValueError('Elevation tile contains a void at candidate')
        return value
    return sample

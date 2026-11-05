"""Loopback-only workspace. User-selected inputs stay on this machine."""
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from .demo import generate
from .physics import fit_trace
from .search import run_search, spectrogram, geojson


def serve(port=8765, review=None):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, status=200):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            from urllib.parse import urlparse,parse_qs
            url = urlparse(self.path)
            if url.path.startswith('/assets/'):
                name = url.path.removeprefix('/assets/')
                if name not in ('leaflet.js','leaflet.css','LEAFLET-LICENSE','review.js'):
                    return self.reply({'error':'Not found'},404)
                path = Path(__file__).with_name('web')/name
                self.send_response(200)
                self.send_header('Content-Type','text/css' if name.endswith('.css') else 'text/javascript' if name.endswith('.js') else 'text/plain')
                self.end_headers()
                self.wfile.write(path.read_bytes())
                return
            if review and url.path.startswith('/review/'):
                try:
                    if url.path == '/review/manifest': return self.reply(review.manifest())
                    q = parse_qs(url.query)
                    index = int(q.get('event',['0'])[0])
                    if not 0 <= index < len(review.events): raise ValueError('Invalid event index')
                    padding = float(q.get('padding',['30'])[0])
                    if url.path == '/review/audio':
                        body = review.clip(index,padding)
                        import re
                        length = len(body)
                        byte_range = self.headers.get('Range')
                        start,end = 0,length-1
                        if byte_range:
                            match = re.fullmatch(r'bytes=(\d*)-(\d*)',byte_range)
                            if not match or not any(match.groups()):
                                raise ValueError('Invalid audio byte range')
                            a,b = match.groups()
                            if a:
                                start = int(a)
                                end = min(int(b),end) if b else end
                            else:
                                start = max(0,length-int(b))
                            if start >= length or start > end:
                                self.send_response(416)
                                self.send_header('Content-Range',f'bytes */{length}')
                                self.end_headers()
                                return
                        self.send_response(206 if byte_range else 200)
                        self.send_header('Content-Type','audio/wav')
                        self.send_header('Accept-Ranges','bytes')
                        if byte_range: self.send_header('Content-Range',f'bytes {start}-{end}/{length}')
                        self.send_header('Content-Length',str(end-start+1))
                        self.end_headers();self.wfile.write(body[start:end+1])
                        return
                    if url.path == '/review/window':
                        candidate = int(q['candidate'][0]) if 'candidate' in q else None
                        if candidate is not None and not 0 <= candidate < len(review.candidates): raise ValueError('Invalid candidate index')
                        return self.reply(review.window(index,candidate,padding,float(q.get('fmin',['30'])[0]),float(q.get('fmax',['700'])[0]),int(q.get('nfft',['16384'])[0])))
                    return self.reply({'error':'Not found'},404)
                except (ValueError,IndexError,KeyError,OSError) as error:
                    return self.reply({'error':str(error)},400)
            if self.path == '/health':
                return self.reply({'status': 'ok'})
            if self.path != '/':
                return self.reply({'error': 'Not found'}, 404)
            body = Path(__file__).with_name('web').joinpath('review.html' if review else 'index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            # Reject browser cross-origin requests to this local processing service.
            origin = self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                return self.reply({'error': 'Cross-origin requests disabled'}, 403)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 48*1024*1024:
                    raise ValueError('Request must be between 1 byte and 48 MiB')
                payload = json.loads(self.rfile.read(length))
                if self.path == '/fit':
                    return self.reply(fit_trace(*zip(*payload['trace'])))
                if self.path not in ('/demo', '/search', '/spectrogram', '/analyze'):
                    return self.reply({'error': 'Not found'}, 404)
                with tempfile.TemporaryDirectory() as directory:
                    if self.path == '/analyze':
                        from .pipeline import analyze
                        import uuid
                        media = Path(directory)/'media'
                        media.write_bytes(base64.b64decode(payload['audio'], validate=True))
                        config = payload.get('config', {})
                        if config.get('dem'):
                            raise ValueError('DEM paths are supported through CLI only')
                        config['auto_terrain'] = True
                        config['terrain_cache'] = 'output/terrain'
                        config['archive_cache'] = 'output/archive-cache'
                        out = Path('output/browser')/uuid.uuid4().hex
                        result = analyze(media, config['bounds'], out,
                                         day=payload.get('date') or None,
                                         recorded_at=payload.get('recorded_at') or None,
                                         config=config, max_gb=float(payload.get('max_gb', 8)))
                        f, t, image = spectrogram(out/'audio.wav')
                        band = (f >= 30) & (f <= 700)
                        stride = max(1,len(t)//1200)
                        return self.reply(dict(result=result, geojson=geojson(result),
                            config=json.loads((out/'search.json').read_text()),
                            spectrogram=dict(f=f[band].tolist(),t=t[::stride].tolist(),z=image[band,::stride].tolist())))
                    if self.path == '/demo':
                        data = generate(directory)
                    else:
                        # Uploaded content has fixed server-side names, never client paths.
                        audio = Path(directory)/'audio.wav'
                        audio.write_bytes(base64.b64decode(payload['audio'], validate=True))
                        flights = Path(directory)/'flights.csv'
                        flights.write_text(payload.get('flights', ''))
                        data = dict(audio=audio, flights=flights, config=payload.get('config', {}))
                        if data['config'].get('dem'):
                            raise ValueError('DEM paths are supported through the CLI only')
                    f, t, image = spectrogram(data['audio'])
                    band = (f >= 30) & (f <= 700)
                    stride = max(1, len(t)//1200)
                    spec = dict(f=f[band].tolist(), t=t[::stride].tolist(), z=image[band, ::stride].tolist())
                    if self.path == '/spectrogram':
                        return self.reply(dict(spectrogram=spec))
                    result = run_search(data['audio'], data['flights'], data['config'])
                    self.reply(dict(result=result, geojson=geojson(result), config=data['config'], spectrogram=spec))
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.reply({'error': str(error)}, 400)
    print(f'DopplerPoint listening on loopback port {port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()

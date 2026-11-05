import argparse
import json
from pathlib import Path
from .search import run_search, geojson
from .demo import generate


def main():
    parser = argparse.ArgumentParser(description='Local Doppler curve analysis')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Generate and search synthetic flyovers')
    demo.add_argument('--output', default='output/demo')
    search = commands.add_parser('search')
    search.add_argument('--audio', required=True)
    search.add_argument('--flights', required=True)
    search.add_argument('--config', required=True)
    search.add_argument('--output', default='output/result.json')
    fit = commands.add_parser('fit', help='Fit an ideal curve to JSON [[seconds, Hz], ...]')
    fit.add_argument('trace')
    serve = commands.add_parser('serve', help='Start local browser workspace')
    serve.add_argument('--port', type=int, default=8765)
    auto = commands.add_parser('analyze', help='Decode media, detect events, download flights/terrain and search')
    auto.add_argument('media')
    auto.add_argument('--bounds', nargs=4, type=float, required=True, metavar=('S','W','N','E'))
    auto.add_argument('--date', help='UTC recording day, YYYY-MM-DD')
    auto.add_argument('--end-date',help='Last UTC date for an uncertain recording date range, inclusive')
    auto.add_argument('--recorded-at', help='Recording start timestamp, ISO-8601 with timezone')
    auto.add_argument('--config', help='Optional manual event/search overrides')
    auto.add_argument('--max-download-gb', type=float, default=8)
    auto.add_argument('--output', default='output/analysis')
    download = commands.add_parser('download', help='Download and normalize an ADSB.lol day archive')
    download.add_argument('date')
    download.add_argument('--bounds', nargs=4, type=float, required=True)
    download.add_argument('--output', default='output/archive')
    download.add_argument('--max-download-gb', type=float, default=8)
    for cmd in (search, auto, demo):
        cmd.add_argument('--workers', type=int, help='CPU process count (default: up to 8)')
        cmd.add_argument('--backend', choices=['cpu','cuda'], default='cpu')
    detect = commands.add_parser('detect', help='Autonomously scan media and save accepted/rejected event diagnostics')
    detect.add_argument('media')
    detect.add_argument('--workers',type=int,default=4)
    detect.add_argument('--output',default='output/detection')
    args = parser.parse_args()
    if args.command == 'detect':
        from .pipeline import prepare_media,detect_events
        out = Path(args.output)
        audio,_ = prepare_media(args.media,out)
        report = out/'detection-report.json'
        try:
            events = detect_events(audio,max_events=1000000,report_path=report,workers=args.workers)
        except ValueError as e:
            if report.exists() and str(e).startswith('No reliable'):
                print(json.dumps(dict(status='no_reliable_detection',report=str(report),reason=str(e)),indent=2))
                return
            raise
        print(json.dumps(dict(status='events_detected',accepted_count=len(events),report=str(report)),indent=2))
        return
    if args.command == 'download':
        from .archive import download_day
        print(download_day(args.date,args.bounds,args.output,args.max_download_gb))
        return
    if args.command == 'analyze':
        from .pipeline import analyze
        config = json.loads(Path(args.config).read_text()) if args.config else {}
        config['backend'] = args.backend
        if args.workers is not None: config['workers'] = args.workers
        elif args.backend == 'cuda': config['workers'] = 1
        result = analyze(args.media,args.bounds,args.output,args.date,args.recorded_at,config,args.max_download_gb,args.end_date)
        print(json.dumps(dict(output=args.output,stats=result['stats'],best=result['results'][:1]),indent=2))
        return
    if args.command == 'serve':
        from .server import serve
        serve(args.port)
        return
    if args.command == 'fit':
        from .physics import fit_trace
        trace = json.loads(Path(args.trace).read_text())
        print(json.dumps(fit_trace(*zip(*trace)), indent=2))
        return
    if args.command == 'demo':
        data = generate(args.output)
        data['config']['backend'] = args.backend
        if args.workers is not None: data['config']['workers'] = args.workers
        elif args.backend == 'cuda': data['config']['workers'] = 1
        result = run_search(data['audio'], data['flights'], data['config'])
        result['synthetic_truth'] = data['truth']
        out = Path(args.output)/'result.json'
    else:
        config = json.loads(Path(args.config).read_text())
        config['backend'] = args.backend
        if args.workers is not None: config['workers'] = args.workers
        elif args.backend == 'cuda': config['workers'] = 1
        result = run_search(args.audio,args.flights,config)
        out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    out.with_suffix('.geojson').write_text(json.dumps(geojson(result), allow_nan=False))
    print(json.dumps(dict(output=str(out), stats=result['stats'], best=result['results'][:1]), indent=2))

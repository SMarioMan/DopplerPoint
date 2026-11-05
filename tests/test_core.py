import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from dopplerpoint.physics import Track, ideal_curve, fit_trace, read_tracks
from dopplerpoint.search import run_search, grid, geojson
from dopplerpoint.demo import generate
from dopplerpoint.archive import convert_archive, trace_rows, verified_download
from dopplerpoint.pipeline import detect_events, prepare_media, analyze


class PhysicsTests(unittest.TestCase):
    def test_retarded_time(self):
        times = np.linspace(-60,60,241)
        track = Track('test',times,np.c_[times*80,np.full(len(times),1000),np.zeros(len(times))])
        cpa,d,v = track.cpa(np.zeros(3))
        self.assertAlmostEqual(cpa,1000/343)
        arrival,ratio = track.curve(np.zeros(3),step=.1)
        expected = ideal_curve(arrival,130,80,1000,cpa)
        np.testing.assert_allclose(ratio*130,expected,rtol=1e-10)
        self.assertTrue(np.all(np.diff(arrival)>0))
        self.assertGreater(ratio[0],1)
        self.assertLess(ratio[-1],1)

    def test_fit(self):
        t=np.linspace(0,100,100)
        hz=ideal_curve(t,130,80,1500,50)
        fit=fit_trace(t,hz)
        self.assertAlmostEqual(fit['speed_mps'],80,places=3)
        self.assertAlmostEqual(fit['slant_m'],1500,places=2)

    def test_grid_budget(self):
        with self.assertRaises(ValueError):grid([20,-130,50,-60],200)

    def test_bad_track(self):
        with self.assertRaises(ValueError):Track('x',np.array([0,0]),np.zeros((2,3)))


class IntegrationTests(unittest.TestCase):
    def test_demo_search_and_detection(self):
        with tempfile.TemporaryDirectory() as d:
            data=generate(d)
            result=run_search(data['audio'],data['flights'],data['config'])
            best=result['results'][0]
            self.assertLess(abs(best['latitude']-40),.0005)
            self.assertLess(abs(best['longitude']+100),.0005)
            self.assertEqual(best['f0_hz'],[130,165])
            self.assertEqual(len(geojson(result)['features']),len(result['results']))
            events=detect_events(data['audio'])
            self.assertEqual(len(events),2)
            for measured,expected in zip(events,data['config']['events']):
                self.assertLess(abs(measured['audio_cpa_s']-expected['audio_cpa_s']),15)
            # Wrong physical offset cannot be rescued by independent event shifts.
            data['config']['events'][1]['audio_cpa_s']+=40
            wrong=run_search(data['audio'],data['flights'],data['config'])
            self.assertEqual(wrong['results'],[])

    def test_parallel_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            data=generate(d)
            config=data['config'];config.pop('refine_spacing_m');config['spacing_m']=500
            config['workers']=1
            serial=run_search(data['audio'],data['flights'],config)
            config['workers']=2
            parallel=run_search(data['audio'],data['flights'],config)
            self.assertEqual(serial['results'],parallel['results'])
            self.assertEqual(serial['stats']['scored_pairs'],parallel['stats']['scored_pairs'])

    def test_archive_split_stream(self):
        trace={'icao':'abc123','timestamp':1000,'trace':[
            [0,40,-100,3000,90,0,0,0,None,'adsb_icao',3300],
            [5,40.001,-100,3000,90,0,0,0,None,'adsb_icao',3300],
            [10,40,-100,'ground',0,0,0,0]]}
        rows=list(trace_rows(trace,[39,-101,41,-99]))
        self.assertEqual(len(rows),2)
        self.assertAlmostEqual(rows[0][4],3300*.3048)
        with tempfile.TemporaryDirectory() as d:
            stream=io.BytesIO()
            with tarfile.open(fileobj=stream,mode='w') as tar:
                raw=gzip.compress(json.dumps(trace).encode())
                member=tarfile.TarInfo('traces/23/trace_full_abc123.json');member.size=len(raw)
                tar.addfile(member,io.BytesIO(raw))
            blob=stream.getvalue(); paths=[Path(d)/'part1',Path(d)/'part2']
            paths[0].write_bytes(blob[:777]);paths[1].write_bytes(blob[777:])
            output=Path(d)/'flights.csv'
            self.assertEqual(convert_archive(paths,output,[39,-101,41,-99]),2)
            self.assertEqual(len(read_tracks(output)),1)

    def test_download_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            asset=dict(name='part',size=3,digest='sha256:'+hashlib.sha256(b'abc').hexdigest(),browser_download_url='https://example.com/part')
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'bad')):
                with self.assertRaises(ValueError):verified_download(asset,d)
            self.assertFalse((Path(d)/'part').exists())
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'abc')):
                self.assertEqual(verified_download(asset,d).read_bytes(),b'abc')

    def test_media_pipeline_with_archive_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            data=generate(Path(d)/'source')
            config=data['config'];config['auto_terrain']=False
            config.pop('events')  # Exercise automatic measurements through the media pipeline.
            # Archive network boundary mocked; decode, detection/search and export run for real.
            with patch('dopplerpoint.pipeline.download_day',return_value=Path(data['flights'])):
                result=analyze(data['audio'],config['bounds'],Path(d)/'out',day='2023-11-14',config=config)
            self.assertLess(abs(result['results'][0]['latitude']-40),.0005)
            self.assertTrue((Path(d)/'out/result.geojson').exists())

if __name__=='__main__':unittest.main()

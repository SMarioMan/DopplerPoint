import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.io import wavfile
from dopplerpoint.demo import generate
from dopplerpoint.physics import ideal_curve
from dopplerpoint.review import ReviewSession


class ReviewTests(unittest.TestCase):
    def test_window_clip_and_actual_flight_overlay(self):
        with tempfile.TemporaryDirectory() as d:
            data=generate(d)
            config=data['config'];config['elevation_m']=0
            cp=Path(d)/'config.json';cp.write_text(json.dumps(config))
            results=Path(d)/'result.json'
            results.write_text(json.dumps(dict(results=[dict(latitude=40,longitude=-100,score=1,
                flights=['demo-0:0','demo-1:0'],f0_hz=[130,165],clock_offset_s=-1700000000)])))
            e=dict(config['events'][0],fitted_parameters=dict(f0_hz=130,speed_mps=75,slant_m=1000),validation=dict(accepted=True))
            report=Path(d)/'report.json';report.write_text(json.dumps(dict(proposals=[e])))
            review=ReviewSession(data['audio'],report,results,cp,data['flights'])
            result=review.window(0,candidate=0,padding=10,fmin=80,fmax=220)
            self.assertEqual([c['label'] for c in result['overlays']],['Ideal fit','Flight geometry'])
            flight=result['overlays'][1]
            self.assertTrue(all(result['start_s']<=t<=result['end_s'] for t in flight['time']))
            self.assertGreater(len(flight['time']),30)
            f=flight['frequency_hz'];self.assertGreater(f[0],f[-1])
            ideal=result['overlays'][0]
            np.testing.assert_allclose(ideal['frequency_hz'],ideal_curve(ideal['time'],130,75,1000,e['audio_cpa_s']))
            self.assertGreaterEqual(min(result['spectrogram']['f']),80)
            self.assertLessEqual(max(result['spectrogram']['f']),220)
            rate,clip=wavfile.read(io.BytesIO(review.clip(0,padding=10)))
            self.assertAlmostEqual(len(clip)/rate,result['end_s']-result['start_s'],places=3)
            self.assertEqual(review.manifest()['candidate_kind'],'scored_observer_candidates')

    def test_coarse_aircraft_points_are_never_labeled_as_observer_matches(self):
        with tempfile.TemporaryDirectory() as d:
            data=generate(d)
            config=Path(d)/'config.json';config.write_text(json.dumps(data['config']))
            results=Path(d)/'coarse.json';results.write_text(json.dumps(dict(candidates=[dict(latitude=40,longitude=-100,flight='abc')],surviving_samples=300000)))
            review=ReviewSession(data['audio'],results=results,config=config)
            self.assertEqual(review.manifest()['candidate_kind'],'coarse_aircraft_samples')
            self.assertIn('NOT observer',review.manifest()['map_note'])
            self.assertEqual(review.window(0,candidate=0)['overlays'],[])
            with self.assertRaises(ValueError):review.window(0,padding=5000)
            with self.assertRaises(ValueError):review.window(0,nfft=15)

    def test_temporal_deduplication_is_reflected_in_review_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            data=generate(d)
            first=dict(data['config']['events'][0],validation=dict(accepted=True))
            duplicate=dict(first,audio_cpa_s=first['audio_cpa_s']+1)
            report=Path(d)/'report.json'
            report.write_text(json.dumps(dict(proposals=[first,duplicate],accepted_events=[first])))
            review=ReviewSession(data['audio'],report=report)
            self.assertEqual([e['detector_selected'] for e in review.manifest()['events']],[True,False])
            self.assertTrue(all(e['validation']['accepted'] for e in review.manifest()['events']))

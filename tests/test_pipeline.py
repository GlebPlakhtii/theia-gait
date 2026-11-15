"""Synthetic correctness checks. These are not experiments or astronaut results."""
import copy
import sys
import tempfile
import unittest
import zipfile
import json
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import write_csv
from evaluate_monitoring import evaluate, match_events
from build_splits import validate as validate_splits
from validate_pose_outputs import validate as validate_poses, FIELDS
from verify_sources import signature, download
from audit_archive import audit


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.truth={'clips':[{'clip_id':'a','participant_id':'p','session_id':'s','duration_s':10,
          'annotation_status':'independent_review_complete','observable_intervals':[[0,10]],'events_s':[2,5,8]}]}
        self.pred={'clips':[{'clip_id':'a','events_s':[2.1,5.4,8.1],'active_intervals':[[0,10]]}]}

    def test_perfect_tolerance(self):
        r=evaluate(self.truth,self.pred)
        self.assertEqual(r['primary_macro_session_event_f1'],1)

    def test_abstention_does_not_hide_misses(self):
        self.pred['clips'][0].update(events_s=[],active_intervals=[])
        r=evaluate(self.truth,self.pred)
        self.assertEqual(r['clips'][0]['fn'],3)
        self.assertFalse(r['coverage_gate_passed'])
        self.assertEqual(r['primary_macro_session_event_f1'],0)

    def test_duplicate_events_and_missing_clips_rejected(self):
        self.pred['clips'][0]['events_s']=[2,2]
        with self.assertRaises(ValueError):evaluate(self.truth,self.pred)
        with self.assertRaises(ValueError):evaluate(self.truth,{'clips':[]})

    def test_draft_truth_rejected(self):
        self.truth['clips'][0]['annotation_status']='ai_draft'
        with self.assertRaises(ValueError):evaluate(self.truth,self.pred)

    def test_no_matching_across_occlusion(self):
        self.truth['clips'][0].update(observable_intervals=[[0,4.9],[5,10]],events_s=[4.8])
        self.pred['clips'][0]['events_s']=[5.1]
        r=evaluate(self.truth,self.pred)
        self.assertEqual(r['clips'][0]['tp'],0)

    def test_one_to_one(self):
        self.assertEqual(match_events([1,1.6],[1.3],.5),1)
        self.assertEqual(match_events([1,1.6],[.6,1.3],.5),2)


class IntegrityTests(unittest.TestCase):
    def test_zip_traversal_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);zpath=td/'bad.zip'
            with zipfile.ZipFile(zpath,'w') as z:
                z.writestr('../escape.txt','x')
            with self.assertRaises(ValueError):audit(zpath,td/'audit.json',td/'extracted')
            self.assertFalse((td/'escape.txt').exists())
            self.assertFalse((td/'extracted').exists())

    def test_resume_refuses_ignored_range(self):
        class Response:
            status=200
            headers={}
            def __enter__(self):return self
            def __exit__(self,*args):pass
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'video.mp4';part=Path(str(dest)+'.part')
            part.write_bytes(b'partial')
            Path(str(part)+'.json').write_text(json.dumps({'url':'https://example.test/file','validator':'v1'}))
            with patch('urllib.request.urlopen',return_value=Response()):
                with self.assertRaises(ValueError):download('https://example.test/file',dest,100)
            self.assertEqual(part.read_bytes(),b'partial')
            self.assertFalse(dest.exists())

    def test_split_leaks_and_exposure_rejected(self):
        r={'clip_id':'a','participant_id':'p','session_id':'s','source_recording_id':'r',
           'sha256':'hash','duplicate_group':'dup','duplicate_review_status':'reviewed',
           'prior_exposure':'none_documented','split':'train'}
        s=dict(r,clip_id='b',split='test')
        self.assertTrue(validate_splits([r,s]))
        s.update(prior_exposure='development')
        self.assertTrue(validate_splits([s]))

    def test_native_clock_and_missing_joints(self):
        with tempfile.TemporaryDirectory() as td:
            fm=Path(td)/'frames.csv'
            write_csv(fm,['source_frame_index','timestamp_s'],[{'source_frame_index':0,'timestamp_s':0},
                                                            {'source_frame_index':1,'timestamp_s':.04}])
            q=[{'clip_id':'a','frame_map':str(fm),'start_s':0,'end_s':1}]
            r=dict.fromkeys(FIELDS,'')
            r.update(clip_id='a',source_frame_index='0',timestamp_s='0',person_track_id='1',
                     joint_schema='coco17',joint_id='0',confidence='0',coordinate_system='pixel',
                     model_name='test',model_version='1',preprocessing='none')
            self.assertEqual(validate_poses([r],q),[])
            r['timestamp_s']='.04'
            self.assertTrue(validate_poses([r],q))
            r['timestamp_s']='0';r['confidence']='1'
            self.assertTrue(validate_poses([r],q))

    def test_reject_html_media(self):
        self.assertEqual(signature(b'<html>error</html>'),'unknown')
        self.assertEqual(signature(b'\x00\x00\x00 ftypmp42'),'mp4')


if __name__ == '__main__':
    unittest.main()

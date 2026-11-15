import unittest
from copy import deepcopy
from compare_gait_groups import aggregate_units,summarize,interval,clip_value
from gait_geometry import NAMES,measure_segment
import numpy as np

RULES=dict(minimum_segment_duration_s=1,minimum_fixed_L0_pixels=80,minimum_component_valid_fraction=.8,primary_minimum_source_clip_retained_duration_fraction=.5)
OUTCOME=dict(id='knee',view='side',family='M05',source='paired:knee:mean_rom95')

def row(clip,value,subject='P1',trial='T1'):
    return dict(clip_id=clip,value=value,subject=subject,trial_block=trial,source='synthetic')

def record():
    return dict(status='observations_extracted',fps=30,clip_duration_s=4,
                segments=[dict(segment_id=0,first_clip_frame=0,end_clip_frame_exclusive=120,
                    fixed_L0_pixels=100,quality_flags=[],view='side',metrics=[dict(metric_id='M05',measurement_status='uncertain')],
                    paired_geometry=dict(knee=dict(mean_rom95=12,valid_fraction=.9)))])

class ComparisonTests(unittest.TestCase):
    def test_repeated_views_are_one_trial_and_subject(self):
        rows=[row('a',10),row('b',20),row('c',90,trial='T2')]
        u=aggregate_units(rows)
        self.assertEqual(len(u),1);self.assertEqual(u[0]['trial_blocks'],2)
        self.assertEqual(u[0]['value'],52.5)
        self.assertEqual(summarize(rows)['participant_count'],1)
    def test_unknown_source_blocks_do_not_become_participants(self):
        s=summarize([row('a',1,None,'file1'),row('b',2,None,'file2')])
        self.assertIsNone(s['participant_count']);self.assertEqual(s['unit_count'],2)
        self.assertIsNone(interval(s,s,dict(minimum_known_subjects_each_group=5))['bounds'])
    def test_missing_is_not_zero(self):
        r=record();r['segments'][0]['paired_geometry']['knee']['mean_rom95']=None
        value,reasons,_=clip_value(r,OUTCOME,RULES)
        self.assertIsNone(value);self.assertIn('numeric_observation_unavailable',reasons)
    def test_view_and_shared_coverage_gates(self):
        r=record();r['segments'][0]['view']='front_rear'
        self.assertIsNone(clip_value(r,OUTCOME,RULES)[0])
        r=record();r['segments'][0]['paired_geometry']['knee']['valid_fraction']=.79
        self.assertIsNone(clip_value(r,OUTCOME,RULES)[0])
        self.assertEqual(clip_value(record(),OUTCOME,RULES)[0]['value'],12)
    def test_short_valid_segment_does_not_represent_whole_clip(self):
        r=record();r['segments'][0]['end_clip_frame_exclusive']=40
        result,reasons,_=clip_value(r,OUTCOME,RULES)
        self.assertIsNone(result);self.assertIn('insufficient_eligible_duration_fraction',reasons)
    def test_paired_geometry_uses_joint_support(self):
        x=np.zeros((60,23,2),float);t=np.arange(60)/30
        for side in ('left','right'):
            x[:,NAMES.index(side+'_hip')]=[0,100]
            x[:,NAMES.index(side+'_knee')]=[0,50]
            x[:,NAMES.index(side+'_ankle')]=[0,0]
        x[:10,NAMES.index('left_knee')]=np.nan
        x[-10:,NAMES.index('right_knee')]=np.nan
        d=measure_segment(x,t,dict(view='side',source='test',activity='walking',environment='overground',cohort='without_amputation'),100)
        self.assertEqual(d['paired_geometry']['knee']['common_frames'],40)
        self.assertAlmostEqual(d['paired_geometry']['knee']['valid_fraction'],2/3)
        self.assertEqual(d['paired_geometry']['knee']['mean_rom95'],0)
    def test_small_known_cohort_has_no_interval(self):
        s=summarize([row('a',1),row('b',2,'P2','T2')])
        self.assertIsNone(interval(s,s,dict(minimum_known_subjects_each_group=5))['bounds'])
    def test_visual_failure_cannot_contribute_high_confidence_value(self):
        r=record();r['segments'][0]['metrics'][0]['reason_codes']=['visual_pose_localization_failure']
        estimate,reasons,_=clip_value(r,OUTCOME,RULES)
        self.assertIsNone(estimate)
        self.assertIn('visual_pose_localization_failure',reasons)

    def test_region_exclusion_is_not_limited_to_one_reason_string(self):
        r=record();r['pose']={'weight_sha256':'sapiens'}
        r['pose_visual_review']=dict(status='regional_pose_failure',weight_sha256='sapiens',
            affected_metric_ids=['M05'],normalization_scale_unreliable=False,
            reason_code='landmarks_occluded')
        estimate,reasons,_=clip_value(r,OUTCOME,RULES)
        self.assertIsNone(estimate);self.assertIn('landmarks_occluded',reasons)
        r['pose_visual_review']['weight_sha256']='another_model'
        self.assertIsNotNone(clip_value(r,OUTCOME,RULES)[0])

if __name__=='__main__':unittest.main()

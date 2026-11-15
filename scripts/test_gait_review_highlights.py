import unittest
from build_gait_review_highlights import reference_fence
from compare_gait_groups import clip_value
from test_gait_comparison import record,RULES,OUTCOME
from extract_gait_metrics import apply_visual_review,apply_visibility_masks,review_evidence_frames
from audit_gait_outputs import audit_visual_exclusion_contract
from gait_geometry import NAMES,filtered,measure_segment
import numpy as np


class ReviewHighlightTests(unittest.TestCase):
    def units(self):
        return [dict(unit_id='participant:P'+str(i),unit_type='participant',value=float(i))for i in range(6)]

    def test_target_does_not_set_own_reference(self):
        units=self.units();units[-1]['value']=100000
        fence=reference_fence(units,'participant:P5')
        self.assertEqual(fence['participant_count'],5)
        self.assertNotIn('participant:P5',fence['participant_ids'])
        self.assertLess(fence['upper'],10)

    def test_too_few_independent_references_abstains(self):
        units=self.units()[:5]
        self.assertIsNone(reference_fence(units,'participant:P0'))
        units.append(dict(unit_id='source:unknown',unit_type='source_file_identity_unresolved',value=100))
        self.assertIsNone(reference_fence(units,'participant:P0'))

    def test_zero_variability_does_not_create_zero_tolerance(self):
        units=self.units()
        for u in units:u['value']=10
        self.assertIsNone(reference_fence(units,'someone_else'))

    def test_bad_leg_scale_blocks_normalized_upper_body_metric(self):
        r=record();r['pose']={'weight_sha256':'weights'}
        r['segments'][0]['metrics']=[dict(metric_id='M03',measurement_status='uncertain',
            observations={'angle':dict(median=8,valid_fraction=1),
                          'offset':dict(median=.1,valid_fraction=1)})]
        r['pose_visual_review']=dict(status='regional_pose_failure',weight_sha256='weights',affected_metric_ids=['M05'])
        outcome=dict(OUTCOME,family='M03',source='observation:offset:median',unit='L0')
        value,reasons,_=clip_value(r,outcome,RULES)
        self.assertIsNone(value)
        self.assertIn('leg_scale_unreliable_after_visual_pose_failure',reasons)
        self.assertIsNotNone(clip_value(r,dict(outcome,source='observation:angle:median',unit='deg'),RULES)[0])
        r['pose_visual_review']['weight_sha256']='different_model'
        self.assertIsNotNone(clip_value(r,outcome,RULES)[0])

    def test_leg_failure_removes_normalized_values_but_keeps_angular_components(self):
        summary={'rom95':3}
        m=dict(metric_id='M03',measurement_status='uncertain',reason_codes=[],
               observation_units={'angle':'deg','offset':'L0'},
               value={'angle':summary,'offset':summary},observations={'angle':summary,'offset':summary})
        review=dict(status='regional_pose_failure',weight_sha256='weights',affected_metric_ids=['M05'],reason_code='visual_pose_localization_failure')
        apply_visual_review({'metrics':[m]},review,{'weight_sha256':'weights'})
        self.assertEqual(set(m['value']),{'angle'})
        self.assertEqual(m['measurement_status'],'uncertain')
        self.assertIn('offset',m['observations'])
        self.assertIn('offset',m['excluded_observation_components'])

    def test_foot_only_failure_does_not_invalidate_hip_knee_ankle_scale(self):
        summary={'rom95':3}
        m=dict(metric_id='M03',measurement_status='uncertain',reason_codes=[],
               observation_units={'angle':'deg','offset':'L0'},value={'angle':summary,'offset':summary})
        review=dict(status='regional_pose_failure',weight_sha256='weights',affected_metric_ids=['M07','M08'],
                    reason_code='visual_pose_localization_failure',normalization_scale_unreliable=False)
        apply_visual_review({'metrics':[m]},review,{'weight_sha256':'weights'})
        self.assertEqual(set(m['value']),{'angle','offset'})
        self.assertNotIn('excluded_observation_components',m)
        r=record();r['pose']={'weight_sha256':'weights'};r['pose_visual_review']=review
        self.assertIsNotNone(clip_value(r,dict(OUTCOME,unit='L0'),RULES)[0])

    def test_occluded_ankle_excluded_from_paired_knee_support_without_gap_filling(self):
        valid=np.ones((60,23),bool);points=np.zeros((60,23,2));times=np.arange(60)/30
        for side in ('left','right'):
            for name,point in [('hip',[0,100]),('knee',[0,50]),('ankle',[0,0])]:points[:,NAMES.index(side+'_'+name)]=point
        review=dict(weight_sha256='weights',manual_visibility_masks=[dict(first_clip_frame=10,end_clip_frame_exclusive=20,joint_names=['left_ankle'])])
        self.assertEqual(apply_visibility_masks(valid,review,{'weight_sha256':'weights'}),10)
        points[~valid]=np.nan;smoothed=filtered(points,times)
        self.assertTrue(np.isnan(smoothed[10:20,NAMES.index('left_ankle')]).all())
        measured=measure_segment(smoothed,times,dict(view='side',source='test',activity='walking',environment='overground',cohort='without_amputation'),100)
        self.assertEqual(measured['paired_geometry']['knee']['common_frames'],50)
        self.assertEqual(measured['paired_geometry']['arm']['common_frames'],60)

    def test_visibility_review_does_not_leak_to_other_pose_method(self):
        valid=np.ones((30,23),bool)
        review=dict(weight_sha256='sapiens',manual_visibility_masks=[dict(first_clip_frame=0,end_clip_frame_exclusive=30,joint_names=['left_ankle'])])
        self.assertEqual(apply_visibility_masks(valid,review,{'method':'publisher_annotation'}),0)
        self.assertTrue(valid.all())

    def test_occlusion_boundaries_are_displayed_without_regional_failure(self):
        review=dict(status='sampled_pose_track_with_limitations',weight_sha256='weights',
                    manual_visibility_masks=[dict(first_clip_frame=170,end_clip_frame_exclusive=181)])
        self.assertEqual(review_evidence_frames(review,{'weight_sha256':'weights'},181),[170,180])
        self.assertEqual(review_evidence_frames(review,{'method':'publisher_annotation'},181),[])

    def test_display_evidence_combines_boundaries_and_rejects_out_of_range(self):
        review=dict(weight_sha256='weights',failure_clip_frames=[9,4,9],
                    manual_visibility_masks=[dict(first_clip_frame=4,end_clip_frame_exclusive=10)])
        self.assertEqual(review_evidence_frames(review,{'weight_sha256':'weights'},10),[4,9])
        review['failure_clip_frames']=[10]
        with self.assertRaises(AssertionError):review_evidence_frames(review,{'weight_sha256':'weights'},10)

    def test_export_audit_rejects_usable_values_for_visibility_exclusion(self):
        metric=dict(metric_id='M16',value=None,measurement_status='not_evaluable',
                    reason_codes=['landmarks_occluded','raw_observations_retained_for_debug_not_comparison'])
        r=dict(pose={'weight_sha256':'weights'},segments=[{'metrics':[metric]}],
               pose_visual_review=dict(status='regional_pose_failure',weight_sha256='weights',
                   affected_metric_ids=['M16'],normalization_scale_unreliable=False,reason_code='landmarks_occluded'))
        self.assertEqual(audit_visual_exclusion_contract(r),1)
        metric['value']={'arm':{'rom95':.1}}
        with self.assertRaises(AssertionError):audit_visual_exclusion_contract(r)

    def test_export_audit_catches_unmarked_bad_scale_even_if_comparison_omits_it(self):
        metric=dict(metric_id='M03',value={'angle':{'rom95':3}},measurement_status='uncertain',
                    reason_codes=[],observation_units={'angle':'deg','offset':'L0'},
                    excluded_observation_components={'offset':'leg_scale_unreliable_after_visual_pose_failure'})
        r=dict(pose={'weight_sha256':'weights'},segments=[{'metrics':[metric]}],
               pose_visual_review=dict(status='regional_pose_failure',weight_sha256='weights',
                   affected_metric_ids=['M05'],normalization_scale_unreliable=True,reason_code='visual_pose_localization_failure'))
        self.assertEqual(audit_visual_exclusion_contract(r),0)
        metric['value']['offset']={'rom95':.1}
        with self.assertRaises(AssertionError):audit_visual_exclusion_contract(r)
        del metric['value']['offset'];metric['excluded_observation_components']={}
        with self.assertRaises(AssertionError):audit_visual_exclusion_contract(r)

    def test_export_audit_keeps_review_specific_to_its_pose_method(self):
        r=dict(pose={'method':'publisher_annotation'},segments=[{'metrics':[
            dict(metric_id='M16',value={'arm':{'rom95':.1}},measurement_status='uncertain')]}],
            pose_visual_review=dict(status='regional_pose_failure',weight_sha256='sapiens',affected_metric_ids=['M16']))
        self.assertEqual(audit_visual_exclusion_contract(r),0)


if __name__=='__main__':unittest.main()

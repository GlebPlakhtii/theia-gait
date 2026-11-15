"""Checks for leakage, missingness and named landmark adaptation."""
import unittest
import numpy as np
from motionbert_gait_common import subject_splits, window_bounds, pose_adapter, reference_fence, classify
from prepare_motionbert_gait_dataset import measured_values
from train_motionbert_gait import average_precision


class TeacherTests(unittest.TestCase):
    def test_small_reference_and_zero_iqr_abstain(self):
        self.assertIsNone(reference_fence({str(i):i for i in range(4)}))
        self.assertIsNone(reference_fence({str(i):1 for i in range(8)}))

    def test_boundary_buffer_and_missing_are_not_negatives(self):
        fence=dict(lower=1,upper=5,buffer=.2)
        self.assertEqual(classify(6,fence,'high'),1)
        self.assertEqual(classify(4,fence,'high'),0)
        self.assertEqual(classify(5,fence,'high'),-1)
        self.assertEqual(classify(np.nan,fence,'high'),-1)
        self.assertEqual(classify(.5,fence,'low'),1)
        self.assertEqual(classify(2,fence,'low'),0)

    def test_repeated_views_do_not_change_subject_split(self):
        rows=[dict(source='progait',subject=f'progait_{i}') for i in range(4)]
        rows += [dict(source='ljmu360',subject=f'ljmu_{i}') for i in range(11)]
        self.assertEqual(subject_splits(rows),subject_splits(rows+rows))
        self.assertEqual(sum(v=='test' for k,v in subject_splits(rows).items() if k.startswith('progait')),1)

    def test_windows_stay_inside_segment(self):
        t=np.arange(100)/30
        bounds=window_bounds(t,15,70,30)
        self.assertTrue(bounds)
        self.assertTrue(all(a>=.5 and b<=70/30+1e-6 and abs(b-a-1)<1e-6 for a,b in bounds))
        self.assertEqual(window_bounds(t,0,20,30),[])

    def test_mapping_preserves_left_right_and_missing_mask(self):
        t=np.arange(60)/30; xy=np.zeros((60,23,2),np.float32)
        xy[:]=[100,200]; xy[:,12,0]=140; xy[:,11,0]=60
        confidence=np.ones((60,23)); valid=np.ones((60,23),bool)
        valid[:,13]=False
        out,ids,ok=pose_adapter(xy,confidence,valid,t,0,400,400)
        self.assertEqual(out.shape,(30,17,3));self.assertTrue(ok)
        np.testing.assert_allclose(out[:,1,0],(140-200)/200)
        np.testing.assert_allclose(out[:,4,0],(60-200)/200)
        self.assertTrue((out[:,5]==0).all())
        self.assertTrue(np.isfinite(out).all())

    def test_pose_failure_blocks_measurement_even_when_raw_value_exists(self):
        outcome=dict(view='side',family='M05',unit='deg',source='paired:knee:mean_rom95')
        seg=dict(view='side',metrics=[dict(metric_id='M05',measurement_status='uncertain')],paired_geometry=dict(knee=dict(valid_fraction=1,mean_rom95=50)))
        review=dict(weight_sha256='w',affected_metric_ids=['M05'],normalization_scale_unreliable=True)
        y,reasons=measured_values(seg,[outcome],review,dict(weight_sha256='w'),100,[],'none')
        self.assertTrue(np.isnan(y[0]));self.assertIn('visual_pose_localization_failure',reasons[0])

    def test_average_precision_handles_ties_and_single_class(self):
        self.assertAlmostEqual(average_precision(np.array([0,1]),np.array([.5,.5])),.5)
        self.assertAlmostEqual(average_precision(np.array([0,1]),np.array([.1,.9])),1.)
        self.assertIsNone(average_precision(np.ones(3),np.ones(3)))


if __name__=='__main__':unittest.main()

"""Synthetic boundary checks; no real train/validation/test data are opened."""
import unittest

import numpy as np

from gait_from_scratch_evaluation import binary_metrics, merge_review_intervals, numeric_metrics


class EvaluationTests(unittest.TestCase):
    def test_perfect_classification_and_fixed_threshold(self):
        r = binary_metrics([0, 1, 0, 1], [.1, .9, .2, .8])
        self.assertEqual(r['confusion'], dict(tp=2, fp=0, fn=0, tn=2))
        self.assertEqual([r[k] for k in ['precision', 'recall', 'f1', 'average_precision']], [1.] * 4)
        self.assertEqual(r['threshold'], .5)

    def test_tied_scores_are_grouped_before_average_precision(self):
        r = binary_metrics([1, 0, 1], [.8, .8, .2])
        self.assertAlmostEqual(r['average_precision'], .5 * .5 + .5 * (2 / 3))
        self.assertEqual(len(r['precision_recall_curve']), 3)
        self.assertEqual(r['confusion'], dict(tp=1, fp=1, fn=1, tn=0))

    def test_constant_baseline_ap_equals_prevalence(self):
        r = binary_metrics([0, 1, 0, 0], [.25] * 4)
        self.assertEqual(r['average_precision'], .25)
        self.assertIsNone(r['precision'])
        self.assertEqual(r['recall'], 0.)
        self.assertEqual(r['f1'], 0.)

    def test_missing_labels_do_not_become_negatives(self):
        r = binary_metrics([1, np.nan, 0], [.9, np.nan, .1], [True, False, True])
        self.assertEqual(r['available_windows'], 2)
        self.assertEqual(r['unavailable_windows'], 1)
        self.assertEqual(r['average_precision'], 1.)
        with self.assertRaises(ValueError):
            binary_metrics([1, np.nan], [.9, .1])

    def test_absent_class_or_empty_set_withholds_performance(self):
        for y, p in [([0, 0], [.1, .8]), ([1, 1], [.1, .8]), ([], [])]:
            r = binary_metrics(y, p)
            self.assertFalse(r['supported'])
            self.assertTrue(all(r[k] is None for k in ['precision', 'recall', 'f1', 'average_precision']))

    def test_invalid_shapes_masks_and_binary_values_rejected(self):
        for y, p, m in [([1], [.8, .9], None), ([1], [.8], [1]),
                        ([2, 0], [.8, .1], None), ([1, 0], [1.1, .1], None)]:
            with self.assertRaises(ValueError):
                binary_metrics(y, p, m)

    def test_original_unit_regression_errors_and_missing_targets(self):
        r = numeric_metrics([10., 20., np.nan], [12., 16., 1000.], [True, True, False])
        self.assertEqual(r['mae'], 3.)
        self.assertEqual(r['mean_signed_error'], -1.)
        self.assertEqual(r['unavailable_windows'], 1)
        self.assertIsNone(numeric_metrics([], [])['mae'])

    def test_overlap_and_multiple_class_events_count_time_once(self):
        intervals = [dict(sequence_id='source-a', start_time_s=s, end_time_s=e)
                     for s, e in [(0., 1.), (.5, 1.5), (.5, 1.5), (1.5, 2.5), (4., 5.)]]
        intervals += [dict(sequence_id='source-b', start_time_s=0., end_time_s=1.)]
        r = merge_review_intervals(intervals)
        self.assertEqual(r['review_event_count'], 3)
        self.assertEqual(r['review_duration_s'], 4.5)
        self.assertEqual(r['intervals'][0]['end_time_s'], 2.5)

    def test_invalid_review_intervals_rejected(self):
        for start, end in [(1., 1.), (2., 1.), (-1., 1.), (0., np.inf)]:
            with self.assertRaises(ValueError):
                merge_review_intervals([dict(sequence_id='a', start_time_s=start, end_time_s=end)])
        self.assertEqual(merge_review_intervals([])['review_duration_s'], 0.)


if __name__ == '__main__':
    unittest.main()

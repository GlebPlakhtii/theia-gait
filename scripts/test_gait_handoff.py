"""Regression checks for rejecting stale or incomplete training handoffs."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from audit_gait_handoff import check_records, stable_sha


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.review = dict(status='sampled_pose_track_with_limitations',
                           weight_sha256='model-a', inspected_clip_frames=[0, 9],
                           affected_metric_ids=[], evidence='Synthetic review')
        self.record = dict(clip_id='native', status='observations_extracted', clip_frames=10,
                           extraction_code_sha256='code-a', pose=dict(weight_sha256='model-a'),
                           pose_visual_review=copy.deepcopy(self.review))

    def check(self, records=None, reviews=None, expected=None, native=None):
        check_records(records if records is not None else [self.record],
                      expected if expected is not None else {'native'},
                      reviews if reviews is not None else {'native': self.review},
                      'code-a', native if native is not None else {'native'})

    def test_complete_reviewed_record_and_separate_publisher_branch(self):
        self.check()
        publisher = copy.deepcopy(self.record)
        publisher['clip_id'] = 'publisher'
        publisher['pose_visual_review'] = None
        publisher['pose'] = dict(method='publisher_annotation')
        self.check([publisher], reviews={}, expected={'publisher'}, native=set())

    def test_pending_duplicate_missing_or_extra_records_rejected(self):
        for records in [[], [self.record, self.record]]:
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                self.check(records)
        self.record['status'] = 'waiting_for_pose'
        with self.assertRaisesRegex(ValueError, 'Pending'):
            self.check()
        self.record['clip_id'] = 'extra'
        with self.assertRaisesRegex(ValueError, 'Missing, extra'):
            self.check()

    def test_changed_review_notes_or_mask_cannot_pass_cached_exports(self):
        self.review['evidence'] = 'New limitation'
        with self.assertRaisesRegex(ValueError, 'Stale QC'):
            self.check()
        self.record['pose_visual_review'] = copy.deepcopy(self.review)
        self.review['manual_visibility_masks'] = [dict(first_clip_frame=2, end_clip_frame_exclusive=4)]
        with self.assertRaisesRegex(ValueError, 'Stale QC'):
            self.check()

    def test_missing_review_and_review_of_different_model_rejected(self):
        self.record['pose_visual_review'] = None
        with self.assertRaisesRegex(ValueError, 'Missing sampled review'):
            self.check(reviews={})
        self.record['pose_visual_review'] = copy.deepcopy(self.review)
        self.record['pose']['weight_sha256'] = 'model-b'
        with self.assertRaisesRegex(ValueError, 'another pose model'):
            self.check()

    def test_unfinished_review_cannot_satisfy_coverage(self):
        self.review['status'] = 'pending'
        self.record['pose_visual_review'] = copy.deepcopy(self.review)
        with self.assertRaisesRegex(ValueError, 'Unfinished sampled review'):
            self.check()

    def test_changed_code_or_unobserved_frame_rejected(self):
        self.record['extraction_code_sha256'] = 'code-b'
        with self.assertRaisesRegex(ValueError, 'Stale extraction code'):
            self.check()
        self.record['extraction_code_sha256'] = 'code-a'
        for frames in [[], [-1], [10], [True], [1.5]]:
            self.review['inspected_clip_frames'] = frames
            self.record['pose_visual_review'] = copy.deepcopy(self.review)
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, 'review frames'):
                self.check()

    def test_byte_hash_detects_same_size_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'video.bin'
            path.write_bytes(b'first')
            initial = stable_sha(path)
            self.assertEqual(initial, hashlib.sha256(b'first').hexdigest())
            path.write_bytes(b'other')
            self.assertNotEqual(stable_sha(path), initial)


if __name__ == '__main__':
    unittest.main()

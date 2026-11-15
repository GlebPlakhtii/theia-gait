import copy
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_sapiens2_pose import FIELDS, load_progress, save_progress, write_pose_rows


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'checkpoint_308.npz'
        self.identity = {'version': 1, 'source': 'example-source-hash'}
        self.frames = [100, 101]
        self.times = [4., 4.04]
        self.eligible = [{'source_frame_index': str(f), 'timestamp_s': str(t)}
                         for f, t in zip(self.frames + [102], self.times + [4.08])]
        self.points = np.ones((2, 308, 2), dtype=np.float32) * 12.125
        self.scores = np.ones((2, 308), dtype=np.float32) * 1.025
        save_progress(self.path, self.identity, self.frames, self.times, self.points, self.scores)

    def tearDown(self):
        self.temp.cleanup()

    def test_raw_prefix_survives_incomplete_next_save(self):
        self.path.with_suffix('.npz.part').write_bytes(b'interrupted write')
        frames, times, points, scores = load_progress(self.path, self.identity, self.eligible)
        self.assertEqual(frames, self.frames)
        self.assertEqual(times, self.times)
        self.assertIsInstance(frames[0], int)
        np.testing.assert_array_equal(points, self.points)
        np.testing.assert_array_equal(scores, self.scores)

    def test_rejects_changed_source_or_configuration(self):
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            load_progress(self.path, {'version': 1, 'source': 'different'}, self.eligible)

    def test_rejects_reordered_or_changed_clock(self):
        with self.assertRaisesRegex(ValueError, 'native-frame prefix'):
            load_progress(self.path, self.identity, list(reversed(self.eligible)))
        changed = copy.deepcopy(self.eligible)
        changed[1]['timestamp_s'] = '4.05'
        with self.assertRaisesRegex(ValueError, 'timestamp mismatch'):
            load_progress(self.path, self.identity, changed)

    def test_rejects_truncated_landmarks(self):
        save_progress(self.path, self.identity, self.frames, self.times, self.points[:, :23], self.scores)
        with self.assertRaisesRegex(ValueError, 'tensor shape'):
            load_progress(self.path, self.identity, self.eligible)

    def test_resumed_csv_matches_uninterrupted_and_keeps_invalid_missing(self):
        self.points[0, 1] = [-1, 5]
        save_progress(self.path, self.identity, self.frames, self.times, self.points, self.scores)
        frames, times, points, scores = load_progress(self.path, self.identity, self.eligible)
        def render(points, scores):
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for f, p, s, row in zip(frames, points, scores, self.eligible):
                write_pose_rows(writer, 'example', f, row['timestamp_s'], p, s,
                                {0: 'nose', 1: 'left_eye'}, 100, 100)
            return stream.getvalue()
        actual = render(points, scores)
        self.assertEqual(actual, render(self.points, self.scores))
        rows = list(csv.DictReader(io.StringIO(actual)))
        self.assertEqual(rows[1]['x'], '')
        self.assertEqual(rows[1]['confidence'], '0')
        self.assertEqual(rows[0]['confidence'], '1.0')


if __name__ == '__main__':
    unittest.main()

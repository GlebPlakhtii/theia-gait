"""CPU contract checks; no production GPU or gait data is consumed."""
import unittest

import torch

from gait_from_scratch_benchmark import benchmark_assessment
from gait_from_scratch_model import GaitTCN


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261114)
        self.model = GaitTCN()
        self.pose = torch.zeros(1, 30, 17, 4)
        self.pose[..., 2:] = 1
        self.context = torch.zeros(1, 8)
        self.context[:, [1, 5]] = 1

    def test_actual_cpu_timing_preserves_model_rng_and_modes(self):
        self.model.train()
        self.model.blocks[0].eval()
        states = {k: v.clone() for k, v in self.model.state_dict().items()}
        modes = [m.training for m in self.model.modules()]
        rng = torch.get_rng_state().clone()
        r = benchmark_assessment(self.model, self.pose, self.context, warmup=1, repetitions=3)
        self.assertEqual(r['parameter_count'], 261387)
        self.assertEqual(r['batch_size'], 1)
        self.assertEqual(len(r['timed_durations_s']), 3)
        self.assertGreater(r['latency_ms']['median'], 0)
        self.assertAlmostEqual(r['windows_per_second'], 1000 / r['latency_ms']['mean'])
        self.assertIsNone(r['cuda_memory']['peak_allocated_bytes'])
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(states[name], value))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_invalid_window_fails_and_restores_training_mode(self):
        self.pose[..., 3] = 0
        self.model.train()
        with self.assertRaisesRegex(ValueError, 'all-invalid'):
            benchmark_assessment(self.model, self.pose, self.context, warmup=1, repetitions=2)
        self.assertTrue(self.model.training)

    def test_reject_invalid_protocol_or_precision(self):
        for options in [dict(warmup=0), dict(repetitions=1), dict(repetitions=True)]:
            with self.assertRaises(ValueError):
                benchmark_assessment(self.model, self.pose, self.context, **options)
        with self.assertRaisesRegex(ValueError, 'precision'):
            benchmark_assessment(self.model, self.pose.double(), self.context.double())

    def test_nonfinite_output_cannot_produce_a_successful_benchmark(self):
        with torch.no_grad():
            self.model.classifier.bias[0] = torch.nan
        with self.assertRaises(FloatingPointError):
            benchmark_assessment(self.model, self.pose, self.context, warmup=1, repetitions=2)
        self.assertTrue(self.model.training)


if __name__ == '__main__':
    unittest.main()

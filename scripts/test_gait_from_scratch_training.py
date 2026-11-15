import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gait_from_scratch_model import GaitTCN, TCNConfig
from gait_from_scratch_sampling import WindowIdentity
from gait_from_scratch_training import (
    TrainingConfig, WindowDataset, audit_inputs, selection_basis, state_digest, train_model,
)


def synthetic_data(split, n=8):
    generator = torch.Generator().manual_seed(17 if split == 'train' else 18)
    pose = torch.randn(n, 30, 17, 4, generator=generator)
    pose[..., 2] = .8
    pose[..., 3] = 1
    context = torch.zeros(n, 8)
    context[:, 1] = context[:, 5] = 1
    targets = dict(classification=torch.zeros(n, 13), regression=torch.zeros(n, 10),
                   evaluability=torch.ones(n, 10))
    targets['classification'][:, 0] = (torch.arange(n) // 2) % 2
    targets['regression'][:, 0] = pose[..., 0].mean((1, 2))
    masks = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in targets.items()}
    for mask in masks.values():
        mask[:, 0] = True
    prefix = 'T' if split == 'train' else 'V'
    identities = [WindowIdentity(prefix + str(i % 4), prefix + str(i % 4) + ':trial',
                                'ljmu360', 'sapiens2_1b', split, True) for i in range(n)]
    return WindowDataset(pose, context, targets, masks, torch.zeros(n, dtype=torch.long), identities)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.train, self.validation = synthetic_data('train'), synthetic_data('validation')
        self.enabled = torch.zeros(13, dtype=torch.bool)
        self.enabled[0] = True
        self.config = TrainingConfig(maximum_epochs=2, minimum_epochs=1, batch_size=4, time_cap_s=60)

    def run_training(self, path, **kwargs):
        return train_model(self.train, self.validation, output_directory=path,
                           enabled_classes=self.enabled, positive_class_weight=torch.ones(13),
                           input_provenance={'purpose': 'synthetic_CPU_preparation', 'actual_gait_data': False},
                           config=kwargs.pop('config', self.config), **kwargs)

    def test_actual_optimizer_updates_and_saved_random_initial_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run'
            result = self.run_training(path)
            self.assertEqual(result['status'], 'trained_with_validation')
            self.assertEqual(result['validation_checks'], 2)
            self.assertGreater(result['nonzero_gradient_steps'], 0)
            self.assertGreater(result['parameter_change_l2'], 0)
            self.assertNotEqual(result['initial_state_sha256'], result['last_state_sha256'])
            initial = torch.load(path / 'initial.pt', weights_only=True)
            last = torch.load(path / 'last.pt', weights_only=True)
            best = torch.load(path / 'best.pt', weights_only=True)
            torch.manual_seed(self.config.seed)
            self.assertEqual(state_digest(GaitTCN().state_dict()), initial['state_sha256'])
            self.assertFalse(initial['optimizer_state']['state'])
            self.assertTrue(last['optimizer_state']['state'])
            self.assertEqual(best['validation']['selection_score'], result['best_validation_score'])
            self.assertFalse(result['test_data_consumed'])
            self.assertFalse(result['held_out_evaluation_performed'])
            with self.assertRaises(FileExistsError):
                self.run_training(path)

    def test_fixed_seed_replays_sampling_and_optimization(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_training(Path(directory) / 'first')
            second = self.run_training(Path(directory) / 'second')
            self.assertEqual(first['last_state_sha256'], second['last_state_sha256'])
            self.assertEqual(first['best_validation_score'], second['best_validation_score'])

    def test_participant_or_trial_leakage_is_rejected_before_artifact_creation(self):
        for field in ('participant_id', 'physical_trial_id'):
            self.validation = synthetic_data('validation')
            old = self.validation.identities[0]
            values = dict(vars(old))
            values[field] = getattr(self.train.identities[0], field)
            self.validation.identities[0] = WindowIdentity(**values)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'never_created'
                with self.assertRaisesRegex(ValueError, 'leakage'):
                    self.run_training(path)
                self.assertFalse(path.exists())

    def test_class_enablement_requires_training_people_not_validation_support(self):
        self.train.targets['classification'][:, 0] = 0
        self.train.targets['classification'][2, 0] = 1
        with self.assertRaisesRegex(ValueError, 'training people'):
            audit_inputs(self.train, self.validation, self.enabled, TCNConfig())

    def test_selection_is_predeclared_and_falls_back_to_normalized_regression(self):
        self.assertEqual(selection_basis(self.validation, self.enabled)['class_indices'], [0])
        self.enabled[:] = False
        basis = selection_basis(self.validation, self.enabled)
        self.assertEqual(basis['name'], 'validation_macro_normalized_mae')
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_training(Path(directory) / 'regression')
            self.assertEqual(result['selection'], basis)
            self.assertEqual(result['status'], 'trained_with_validation')
        self.validation.masks['regression'][:] = False
        with self.assertRaisesRegex(ValueError, 'Neither supported'):
            selection_basis(self.validation, self.enabled)

    def test_expired_cap_does_not_fabricate_a_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'expired'
            c = TrainingConfig(maximum_epochs=2, minimum_epochs=1, time_cap_s=1e-9)
            result = self.run_training(path, config=c)
            self.assertEqual(result['status'], 'no_validated_training_result')
            self.assertEqual(result['stop_reason'], 'time_cap')
            self.assertEqual(result['updates'], 0)
            self.assertEqual(result['validation_checks'], 0)
            self.assertFalse((path / 'best.pt').exists())
            self.assertEqual(result['initial_state_sha256'], result['last_state_sha256'])

    def test_early_stopping_waits_for_minimum_epochs(self):
        c = TrainingConfig(maximum_epochs=6, minimum_epochs=4, patience=1, batch_size=8, time_cap_s=60)
        with tempfile.TemporaryDirectory() as directory, patch('gait_from_scratch_training.validate_model', return_value={'selection_score': .5}):
            result = self.run_training(Path(directory) / 'early', config=c)
            self.assertEqual(result['stop_reason'], 'early_stopping')
            self.assertEqual(result['epochs_entered'], 4)
            self.assertEqual(result['best_epoch'], 1)

    def test_auxiliary_publisher_cannot_supply_classification_or_validation(self):
        for split in ('train', 'validation'):
            data = synthetic_data(split)
            old = data.identities[0]
            data.identities[0] = WindowIdentity(old.participant_id, old.physical_trial_id, 'progait', 'publisher', split, True)
            data.method_index[0] = 1
            with self.assertRaises(ValueError):
                data.validate(split, TCNConfig())
            data.masks['classification'][0] = False
            if split == 'train':
                data.validate(split, TCNConfig())
            else:
                with self.assertRaisesRegex(ValueError, 'training people only'):
                    data.validate(split, TCNConfig())


if __name__ == '__main__':
    unittest.main()

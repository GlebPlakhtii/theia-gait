"""CPU checks of the model/loss contract, not an experiment on gait data."""
import unittest

import torch

from gait_from_scratch_model import GaitTCN, TCNConfig, masked_multitask_loss


def inputs(batch=4):
    pose = torch.randn(batch, 30, 17, 4)
    pose[..., 2] = .8
    pose[..., 3] = 1
    context = torch.zeros(batch, len(TCNConfig().context_names))
    context[:, 1] = 1  # side
    context[:, 5] = 1  # overground
    return pose, context


class ScratchModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261114)

    def test_size_shapes_and_independent_random_initializations(self):
        model = GaitTCN()
        self.assertLess(model.parameter_count, 500_000)
        out = model(*inputs())
        self.assertEqual(out['class_logits'].shape, (4, 13))
        self.assertEqual(out['numeric'].shape, (4, 2, 10))
        self.assertEqual(out['evaluability_logits'].shape, (4, 10))
        another = GaitTCN()
        self.assertFalse(torch.equal(model.frame_embedding.weight, another.frame_embedding.weight))
        torch.manual_seed(20261114)
        replay = GaitTCN()
        self.assertTrue(torch.equal(model.frame_embedding.weight, replay.frame_embedding.weight))

    def test_invalid_placeholders_cannot_change_predictions(self):
        model = GaitTCN().eval()
        pose, context = inputs()
        pose[:, 4:10, :, 3] = 0
        pose[:, :, 7, 3] = 0
        changed = pose.clone()
        changed[..., :3][pose[..., 3] == 0] = 1e6
        with torch.no_grad():
            first, second = model(pose, context), model(changed, context)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(first[key]).all())

    def test_reject_all_invalid_nonfinite_or_wrong_input_schema(self):
        model = GaitTCN(); pose, context = inputs()
        pose[1, ..., 3] = 0
        with self.assertRaisesRegex(ValueError, 'all-invalid'):
            model(pose, context)
        pose, context = inputs(); pose[0, 0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, 'finite placeholders'):
            model(pose, context)
        pose, context = inputs()
        with self.assertRaisesRegex(ValueError, 'declared view/environment'):
            model(pose, torch.cat((context, torch.zeros(4, 1)), dim=1))

    def test_unknown_targets_have_no_loss_and_no_gradient(self):
        out = dict(class_logits=torch.randn(2, 13, requires_grad=True),
                   numeric=torch.randn(2, 2, 10, requires_grad=True),
                   evaluability_logits=torch.randn(2, 10, requires_grad=True))
        shapes = dict(classification=(2, 13), regression=(2, 10), evaluability=(2, 10))
        targets = {k:torch.full(v, torch.nan) for k,v in shapes.items()}
        masks = {k:torch.zeros(v, dtype=torch.bool) for k,v in shapes.items()}
        masks['regression'][0, 2] = True; targets['regression'][0, 2] = 2
        masks['evaluability'][1, 3] = True; targets['evaluability'][1, 3] = 0
        loss = masked_multitask_loss(out, targets, masks, torch.tensor([0, 1]))
        self.assertEqual(loss['classification'].item(), 0)
        loss['total'].backward()
        self.assertEqual(out['class_logits'].grad.abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(out['numeric'].grad).item(), 1)
        self.assertEqual(torch.count_nonzero(out['evaluability_logits'].grad).item(), 1)
        self.assertGreater(out['numeric'].grad[0, 0, 2].abs().item(), 0)
        self.assertGreater(out['evaluability_logits'].grad[1, 3].abs().item(), 0)

    def test_empty_masks_keep_finite_zero_loss(self):
        model = GaitTCN(); out = model(*inputs())
        shapes = dict(classification=(4, 13), regression=(4, 10), evaluability=(4, 10))
        targets = {k:torch.full(v, torch.nan) for k,v in shapes.items()}
        masks = {k:torch.zeros(v, dtype=torch.bool) for k,v in shapes.items()}
        losses = masked_multitask_loss(out, targets, masks, torch.zeros(4, dtype=torch.long))
        self.assertEqual(losses['total'].item(), 0)
        losses['total'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_rescaling_sample_weights_does_not_rescale_mean_loss(self):
        model = GaitTCN(); out = model(*inputs())
        targets = dict(classification=torch.zeros(4, 13), regression=torch.ones(4, 10),
                       evaluability=torch.zeros(4, 10))
        masks = {k:torch.zeros_like(v, dtype=torch.bool) for k,v in targets.items()}
        for mask in masks.values():
            mask[0, 0] = True
        method = torch.zeros(4, dtype=torch.long)
        first = masked_multitask_loss(out, targets, masks, method, sample_weight=torch.full((4,), .01))
        second = masked_multitask_loss(out, targets, masks, method, sample_weight=torch.full((4,), 10.))
        for name in first:
            torch.testing.assert_close(first[name], second[name])

    def test_actual_optimizer_update_reaches_encoder_and_all_heads(self):
        model = GaitTCN(); pose, context = inputs()
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        out = model(pose, context)
        targets = dict(classification=torch.randint(0, 2, (4, 13)).float(),
                       regression=torch.randn(4, 10), evaluability=torch.randint(0, 2, (4, 10)).float())
        masks = {k:torch.ones_like(v, dtype=torch.bool) for k,v in targets.items()}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        loss = masked_multitask_loss(out, targets, masks, torch.tensor([0, 1, 0, 1]))
        self.assertTrue(torch.isfinite(loss['total']))
        loss['total'].backward()
        for prefix in ['frame_embedding', 'blocks', 'readout', 'classifier', 'numeric_heads.0', 'numeric_heads.1', 'evaluability']:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters() if n.startswith(prefix)), prefix)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        for prefix in ['frame_embedding', 'blocks', 'readout', 'classifier', 'numeric_heads.0', 'numeric_heads.1', 'evaluability']:
            self.assertTrue(any(not torch.equal(before[n], p) for n,p in model.named_parameters() if n.startswith(prefix)), prefix)


if __name__ == '__main__':
    unittest.main()

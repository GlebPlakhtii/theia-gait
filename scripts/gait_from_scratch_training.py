"""Bounded random-initialized TCN training on already prepared, audited tensors.

No dataset discovery, automatic label fitting, pretrained loading, or job launch
occurs on import. The caller must freeze and independently audit source hashes,
adapter, labels, splits, and train-only normalization before a real-data run.
This module accepts training and validation data only; it has no test-data path.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from gait_from_scratch_evaluation import binary_metrics, numeric_metrics
from gait_from_scratch_model import GaitTCN, TCNConfig, masked_multitask_loss
from gait_from_scratch_sampling import PRIMARY, WindowIdentity, sample_training_epoch


TASKS = ('classification', 'regression', 'evaluability')


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20261114
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    gradient_clip: float = 1.0
    maximum_epochs: int = 100
    minimum_epochs: int = 10
    patience: int = 10
    time_cap_s: float = 2700.0
    positive_weight_cap: float = 10.0


@dataclass
class WindowDataset:
    pose: torch.Tensor
    context: torch.Tensor
    targets: dict
    masks: dict
    method_index: torch.Tensor
    identities: list[WindowIdentity]

    def __len__(self):
        return len(self.identities)

    def validate(self, split, model_config):
        n, c = len(self), model_config
        if not n or split not in ('train', 'validation'):
            raise ValueError('Training requires nonempty training and validation sets')
        if tuple(self.pose.shape) != (n, c.frames, c.joints, len(c.channels)):
            raise ValueError('Incorrect prepared pose schema')
        if tuple(self.context.shape) != (n, len(c.context_names)):
            raise ValueError('Incorrect declared-context schema')
        if self.pose.dtype != torch.float32 or self.context.dtype != torch.float32:
            raise ValueError('Prepared model inputs must use float32')
        for tensor in (self.pose, self.context, self.method_index, *self.targets.values(), *self.masks.values()):
            if tensor.device.type != 'cpu':
                raise ValueError('Prepared datasets must remain on CPU; batches move to the device')
        if not torch.isfinite(self.pose).all() or not torch.isfinite(self.context).all():
            raise ValueError('Inputs require finite placeholders and explicit masks')
        valid = self.pose[..., 3]
        if not ((valid == 0) | (valid == 1)).all() or not valid.bool().any(dim=(1, 2)).all():
            raise ValueError('Invalid masks or all-invalid input windows')
        if not ((self.context == 0) | (self.context == 1)).all():
            raise ValueError('Context must be declared one-hot view/environment only')
        if not (self.context[:, :len(c.views)].sum(1) == 1).all() or not (self.context[:, len(c.views):].sum(1) == 1).all():
            raise ValueError('Each window needs exactly one declared view and environment')
        if self.method_index.dtype != torch.long or tuple(self.method_index.shape) != (n,):
            raise ValueError('One integer method index is required per window')
        if set(self.targets) != set(TASKS) or set(self.masks) != set(TASKS):
            raise ValueError('All three target tasks and availability masks are required')
        for task in TASKS:
            width = len(c.classes) if task == 'classification' else len(c.outcomes)
            target, mask = self.targets[task], self.masks[task]
            if tuple(target.shape) != (n, width) or mask.shape != target.shape or mask.dtype != torch.bool:
                raise ValueError('Target/mask schema mismatch: ' + task)
            if not torch.isfinite(target[mask]).all():
                raise ValueError('Available targets must be finite: ' + task)
            if task != 'regression' and not ((target[mask] == 0) | (target[mask] == 1)).all():
                raise ValueError('Available binary targets must be zero or one')
        for i, identity in enumerate(self.identities):
            if identity.split != split or identity.known_participant is not True:
                raise ValueError('Wrong split or unresolved participant in primary data')
            if not identity.participant_id or not identity.physical_trial_id:
                raise ValueError('Participant and physical-trial IDs are required')
            if identity.source not in ('progait', 'ljmu360', 'lbmc'):
                raise ValueError('Provisional and unresolved sources remain exploratory-only')
            if identity.method not in c.numeric_methods or int(self.method_index[i]) != c.numeric_methods.index(identity.method):
                raise ValueError('Method metadata and numeric head index disagree')
            if identity.method != PRIMARY:
                if identity.source != 'progait' or split != 'train':
                    raise ValueError('Publisher auxiliary targets are for ProGait training people only')
                if self.masks['classification'][i].any():
                    raise ValueError('Publisher inputs have no matched classification reference')

    def batch(self, indices, device, enabled):
        indices = torch.as_tensor(indices, dtype=torch.long)
        masks = {k: v[indices].to(device) for k, v in self.masks.items()}
        masks['classification'] = masks['classification'] & enabled.to(device)
        return (self.pose[indices].to(device), self.context[indices].to(device),
                {k: v[indices].to(device) for k, v in self.targets.items()}, masks,
                self.method_index[indices].to(device))


def audit_inputs(train, validation, enabled, config):
    train.validate('train', config)
    validation.validate('validation', config)
    if enabled.dtype != torch.bool or tuple(enabled.shape) != (len(config.classes),):
        raise ValueError('Enabled class mask must be explicit and match the declared classes')
    for field in ('participant_id', 'physical_trial_id'):
        training_ids = {getattr(i, field) for i in train.identities}
        validation_ids = {getattr(i, field) for i in validation.identities}
        if training_ids & validation_ids:
            raise ValueError('Training/validation identity leakage: ' + field)
    trial_owners = {}
    for identity in train.identities + validation.identities:
        if trial_owners.setdefault(identity.physical_trial_id, identity.participant_id) != identity.participant_id:
            raise ValueError('One physical trial cannot have different participant owners')
    # Independently recheck enabled-class support; never infer it from validation.
    for j in torch.nonzero(enabled, as_tuple=False).flatten().tolist():
        people = [set(), set()]
        for i, identity in enumerate(train.identities):
            if train.masks['classification'][i, j]:
                people[int(train.targets['classification'][i, j])].add(identity.participant_id)
        if min(map(len, people)) < 2:
            raise ValueError('Enabled class lacks two positive and two negative training people')
    active = train.masks['classification'] & enabled
    if not (active.any() or train.masks['regression'].any() or train.masks['evaluability'].any()):
        raise ValueError('No available training supervision')


def selection_basis(validation, enabled):
    """Predeclare selection from label support, before any optimization."""
    classes = []
    for j in torch.nonzero(enabled, as_tuple=False).flatten().tolist():
        y = validation.targets['classification'][:, j][validation.masks['classification'][:, j]]
        if (y == 0).any() and (y == 1).any():
            classes.append(j)
    if classes:
        return dict(name='validation_macro_average_precision',direction='maximize',class_indices=classes,
                    integration='non-interpolated average precision; ties grouped')
    outcomes = torch.nonzero(validation.masks['regression'].any(0), as_tuple=False).flatten().tolist()
    if not outcomes:
        raise ValueError('Neither supported validation classification nor regression is available')
    return dict(name='validation_macro_normalized_mae',direction='minimize',outcome_indices=outcomes,
                aggregation='Mean of per-outcome normalized window MAE')


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_model(model, dataset, enabled, basis, config, device, deadline, clock):
    """Return no selection result if the full validation pass exceeds the cap."""
    model.eval()
    probabilities, numeric = [], []
    with torch.inference_mode():
        for start in range(0, len(dataset), config.batch_size):
            if clock() >= deadline:
                return None
            pose, context, _, _, method = dataset.batch(range(start, min(start + config.batch_size, len(dataset))), device, enabled)
            out = model(pose, context)
            probabilities.append(out['class_logits'].sigmoid().cpu())
            numeric.append(out['numeric'][torch.arange(len(method), device=device), method].cpu())
    if clock() >= deadline:
        return None
    probabilities, numeric = torch.cat(probabilities).numpy(), torch.cat(numeric).numpy()
    if not np.isfinite(probabilities).all() or not np.isfinite(numeric).all():
        raise FloatingPointError('Nonfinite validation output')
    classes = [binary_metrics(dataset.targets['classification'][:, j].numpy(), probabilities[:, j],
               (dataset.masks['classification'][:, j] & enabled[j]).numpy()) for j in range(len(enabled))]
    outcomes = [numeric_metrics(dataset.targets['regression'][:, j].numpy(), numeric[:, j],
                dataset.masks['regression'][:, j].numpy()) for j in range(numeric.shape[1])]
    if basis['direction'] == 'maximize':
        score = float(np.mean([classes[j]['average_precision'] for j in basis['class_indices']]))
    else:
        score = float(np.mean([outcomes[j]['mae'] for j in basis['outcome_indices']]))
    return dict(selection_score=score, classification=classes, normalized_regression=outcomes)


def train_model(train, validation, *, output_directory, enabled_classes,
                positive_class_weight, input_provenance, config=None,
                model_config=None, device='cpu', clock=time.monotonic):
    """Train once from random weights; never read a checkpoint or held-out test.

    Targets are normalized using separately frozen training-only statistics.
    Validation contains primary Sapiens windows only. Initial, best and last
    checkpoints carry model/optimizer/RNG/config/provenance. A cap reached before
    one complete validation pass is explicitly unsuccessful, not a best model.
    """
    c, mc = config or TrainingConfig(), model_config or TCNConfig()
    if not 1 <= c.minimum_epochs <= c.maximum_epochs <= 100 or c.patience < 1 or c.batch_size < 1:
        raise ValueError('Invalid bounded epoch, patience or batch configuration')
    if not all(np.isfinite(x) for x in (c.learning_rate, c.weight_decay, c.gradient_clip, c.time_cap_s, c.positive_weight_cap)):
        raise ValueError('Training settings must be finite')
    if min(c.learning_rate, c.gradient_clip, c.time_cap_s, c.positive_weight_cap) <= 0 or c.weight_decay < 0 or c.time_cap_s > 2700:
        raise ValueError('Invalid optimizer settings or time cap above 45 minutes')
    enabled = torch.as_tensor(enabled_classes)
    audit_inputs(train, validation, enabled, mc)
    basis = selection_basis(validation, enabled)
    positive = torch.as_tensor(positive_class_weight, dtype=torch.float32)
    if tuple(positive.shape) != (len(mc.classes),) or not torch.isfinite(positive).all() or (positive <= 0).any() or (positive > c.positive_weight_cap).any():
        raise ValueError('Positive class weights must be finite, positive and within the declared cap')
    if not isinstance(input_provenance, dict) or not input_provenance:
        raise ValueError('Explicit frozen-input or synthetic-test provenance is required')
    json.dumps(input_provenance, allow_nan=False)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('This trainer has only CPU and CUDA verification paths')
    started = clock()
    deadline = started + c.time_cap_s
    torch.manual_seed(c.seed)
    model = GaitTCN(mc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    initial_digest = state_digest(initial)
    manifest = dict(training=asdict(c), model=mc.to_dict(), enabled_classes=enabled.tolist(),
                    positive_class_weight=positive.tolist(), selection=basis,
                    input_provenance=input_provenance, initialization='random; no checkpoint loaded',
                    initial_state_sha256=initial_digest, parameter_count=model.parameter_count,
                    device=str(device), torch_version=str(torch.__version__),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    model_inputs=['pose_coordinates_confidence_masks', 'declared_view_environment'],
                    test_data_consumed=False,
                    cap_scope='Training and validation; checked between batches. A batch already in flight and final checkpoint serialization may finish after the cap.')
    write_json(destination / 'resolved_training.json', manifest)

    def save_checkpoint(name, epoch, updates, validation_result=None):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise FloatingPointError('A model parameter became nonfinite')
        checkpoint = dict(model_state=state, optimizer_state=optimizer.state_dict(),
                          torch_rng_state=torch.get_rng_state(), epoch=epoch, updates=updates,
                          state_sha256=state_digest(state), configuration=manifest,
                          validation=validation_result,
                          cuda_rng_state=torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None)
        path = destination / (name + '.pt')
        temporary = path.with_suffix('.pt.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        return checkpoint['state_sha256']

    save_checkpoint('initial', 0, 0)
    positive = positive.to(device)
    history, best_score, best_epoch = [], None, None
    bad_checks = updates = nonzero_gradient_steps = 0
    maximum_gradient_norm = 0.0
    stop_reason, last_epoch = 'maximum_epochs', 0
    for epoch in range(1, c.maximum_epochs + 1):
        if clock() >= deadline:
            stop_reason = 'time_cap'
            break
        model.train()
        indices, sampling = sample_training_epoch(train.identities, epoch=epoch - 1, seed=c.seed)
        totals = {k: 0.0 for k in (*TASKS, 'total')}
        batches = skipped_batches = consumed = 0
        for start in range(0, len(indices), c.batch_size):
            if clock() >= deadline:
                stop_reason = 'time_cap'
                break
            pose, context, targets, masks, method = train.batch(indices[start:start + c.batch_size], device, enabled)
            consumed += len(pose)
            if not any(mask.any() for mask in masks.values()):
                skipped_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pose, context)
            losses = masked_multitask_loss(outputs, targets, masks, method, positive_class_weight=positive)
            if not all(torch.isfinite(loss) for loss in losses.values()):
                raise FloatingPointError('A training loss became nonfinite')
            losses['total'].backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), c.gradient_clip, error_if_nonfinite=True)
            norm = float(gradient)
            maximum_gradient_norm = max(maximum_gradient_norm, norm)
            nonzero_gradient_steps += int(norm > 0)
            optimizer.step()
            updates += 1
            batches += 1
            for name in totals:
                totals[name] += float(losses[name].detach())
        last_epoch = epoch
        row = dict(epoch=epoch, batches=batches, skipped_unsupervised_batches=skipped_batches,
                   consumed_windows=consumed, sampling=sampling,
                   training_loss={k: v / batches if batches else None for k, v in totals.items()},
                   elapsed_s=clock() - started, validation=None)
        if stop_reason != 'time_cap':
            row['validation'] = validate_model(model, validation, enabled, basis, c, device, deadline, clock)
            if row['validation'] is None:
                stop_reason = 'time_cap'
        if row['validation'] is not None:
            score = row['validation']['selection_score']
            improved = best_score is None or (score > best_score if basis['direction'] == 'maximize' else score < best_score)
            if improved:
                best_score, best_epoch, bad_checks = score, epoch, 0
                save_checkpoint('best', epoch, updates, row['validation'])
            else:
                bad_checks += 1
            if epoch >= c.minimum_epochs and bad_checks >= c.patience:
                stop_reason = 'early_stopping'
        history.append(row)
        write_json(destination / 'training_history.json', history)
        save_checkpoint('last', epoch, updates, row['validation'])
        if stop_reason in ('time_cap', 'early_stopping'):
            break
    final_digest = save_checkpoint('last', last_epoch, updates, history[-1]['validation'] if history else None)
    delta = float(sum((value.detach().cpu() - initial[name]).double().square().sum()
                      for name, value in model.state_dict().items()).sqrt())
    result = dict(status='trained_with_validation' if best_epoch is not None and nonzero_gradient_steps and delta > 0 else 'no_validated_training_result',
                  stop_reason=stop_reason, epochs_entered=last_epoch,
                  validation_checks=sum(row['validation'] is not None for row in history),
                  updates=updates, nonzero_gradient_steps=nonzero_gradient_steps,
                  maximum_gradient_norm_before_clipping=maximum_gradient_norm,
                  parameter_change_l2=delta, initial_state_sha256=initial_digest,
                  last_state_sha256=final_digest, best_epoch=best_epoch,
                  best_validation_score=best_score, selection=basis,
                  elapsed_s=clock() - started, test_data_consumed=False,
                  held_out_evaluation_performed=False,
                  checkpoints={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.glob('*.pt')})
    write_json(destination / 'training_result.json', result)
    return result

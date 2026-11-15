"""Bounded real encoder fine-tuning; never interrupts the existing pose job."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from motionbert_gait_common import ROOT, SOURCE, OUT, SEED, METHODS, CHECKPOINT_SHA, now, sha, write

sys.path.insert(0, str(ROOT / 'vendor/motionbert'))
from lib.model.DSTformer import DSTformer


class GaitModel(nn.Module):
    def __init__(self, outcomes=10, labels=13, load_initial=True):
        super().__init__()
        self.encoder = DSTformer(dim_feat=512, dim_rep=512, depth=5, num_heads=8,
                                 mlp_ratio=2, num_joints=17, maxlen=243)
        if load_initial:
            path = ROOT / 'models/motionbert/pose3d_h36m.bin'
            assert sha(path) == CHECKPOINT_SHA, 'Initial checkpoint checksum mismatch'
            original = torch.load(path, map_location='cpu', weights_only=True)['model_pos']
            self.encoder.load_state_dict({k.removeprefix('module.'): v for k, v in original.items()}, strict=True)
        self.readout = nn.Sequential(nn.LayerNorm(518), nn.Linear(518, 128), nn.GELU(), nn.Dropout(.1))
        self.classifier = nn.Linear(128, labels)
        self.regressors = nn.ModuleList([nn.Linear(128, outcomes) for _ in METHODS])
        self.evaluability = nn.Linear(128, outcomes)

    def forward(self, x, context, method):
        representation = self.encoder.get_representation(x)
        confidence = x[..., 2:3].clamp(0, 1)
        pooled = (representation * confidence).sum((1, 2)) / confidence.sum((1, 2)).clamp_min(1)
        z = self.readout(torch.cat([pooled, context], -1))
        regressions = torch.stack([head(z) for head in self.regressors], 1)
        selected = regressions[torch.arange(len(x), device=x.device), method]
        return self.classifier(z), selected, self.evaluability(z)


def load_data(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'dataset_manifest.json').read_text())
    assert sha(folder / 'dataset.npz') == manifest['dataset_sha256']
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(folder / name) == digest, 'Training artifact changed: ' + name
    with np.load(folder / 'dataset.npz', allow_pickle=False) as a:
        data = {k: a[k].copy() for k in a.files}
    data['windows'] = json.loads((folder / 'windows.json').read_text())
    data['protocol'] = json.loads((folder / 'label_protocol.json').read_text())
    data['normalization'] = json.loads((folder / 'normalization.json').read_text())
    data['manifest'] = manifest
    ctx = np.zeros((len(data['x']), 6), np.float32)
    for i, row in enumerate(data['windows']):
        ctx[i, {'front_rear': 0, 'side': 1}.get(row['view'], 2)] = 1
        ctx[i, {'overground': 3, 'treadmill': 4}.get(row['environment'], 5)] = 1
    data['context'] = ctx
    centers = np.asarray([data['normalization'][m]['center'] for m in METHODS], np.float32)
    scales = np.asarray([data['normalization'][m]['scale'] for m in METHODS], np.float32)
    counts = np.asarray([data['normalization'][m]['count'] for m in METHODS])
    data['regression_mask'] = data['measurement_mask'] & (counts[data['method']] > 0)
    data['standardized'] = np.nan_to_num((data['values'] - centers[data['method']]) / scales[data['method']], nan=0)
    enabled = np.array([x['classification_enabled'] for x in data['protocol']['labels']], bool)
    data['label_mask'] = (data['labels'] >= 0) & enabled[None] & (data['method'][:, None] == 0)
    train = data['split'] == 0
    pos = ((data['labels'] == 1) & data['label_mask'] & train[:, None]).sum(0)
    neg = ((data['labels'] == 0) & data['label_mask'] & train[:, None]).sum(0)
    data['positive_weights'] = np.clip(neg / np.maximum(pos, 1), .2, 10).astype(np.float32)
    return data


def average_precision(y, p):
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(-p, kind='stable')
    y, p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(y)[ends]
    recall = tp / np.sum(y)
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def metrics(data, ids, predictions):
    probabilities, values, assessable = predictions
    outcome_results = []
    normalized_errors = []
    for j, outcome in enumerate(data['protocol']['outcomes']):
        mask = data['regression_mask'][ids, j]
        absolute = abs(values[:, j] - data['values'][ids, j])
        scales = np.array([data['normalization'][METHODS[data['method'][i]]]['scale'][j] for i in ids])
        mae = float(np.mean(absolute[mask])) if mask.any() else None
        if mask.any():
            normalized_errors.append(float(np.mean((absolute / scales)[mask])))
        outcome_results.append(dict(id=outcome['id'], unit=outcome['unit'], n=int(mask.sum()), mae=mae))
    label_results = []; ap_values = []
    for j, label in enumerate(data['protocol']['labels']):
        mask = data['label_mask'][ids, j]
        actual = data['labels'][ids, j][mask]
        score = probabilities[:, j][mask]
        predicted = score >= .5
        tp = int(np.sum(predicted & (actual == 1))); fp = int(np.sum(predicted & (actual == 0)))
        fn = int(np.sum(~predicted & (actual == 1))); tn = int(np.sum(~predicted & (actual == 0)))
        ap = average_precision(actual, score) if len(actual) else None
        if ap is not None:
            ap_values.append(ap)
        label_results.append(dict(id=label['id'], n=len(actual), positive=int(np.sum(actual == 1)),
            negative=int(np.sum(actual == 0)), average_precision=ap,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            tp=tp, fp=fp, fn=fn, tn=tn, enabled=label['classification_enabled']))
    macro_ap = float(np.mean(ap_values)) if ap_values else None
    normalized_mae = float(np.mean(normalized_errors)) if normalized_errors else None
    return dict(windows=len(ids), participants=sorted({data['windows'][i]['subject'] for i in ids if data['windows'][i]['subject']}),
        macro_average_precision=macro_ap, mean_normalized_mae=normalized_mae,
        selection_score=macro_ap if macro_ap is not None else -normalized_mae if normalized_mae is not None else None,
        selection_basis='automatic_label_AP' if macro_ap is not None else 'measurement_regression_error',
        outcomes=outcome_results, labels=label_results,
        interpretation='Agreement with automatic engineering targets, not clinical accuracy. Overlapping windows are not independent people.')


def predict(model, data, ids, device, batch_size=4):
    model.eval(); cls, reg, quality = [], [], []
    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            with torch.autocast('cuda', dtype=torch.bfloat16) if use_amp else nullcontext():
                c, r, q = model(torch.from_numpy(data['x'][batch]).to(device),
                    torch.from_numpy(data['context'][batch]).to(device),
                    torch.from_numpy(data['method'][batch].astype(np.int64)).to(device))
            cls.append(c.float().sigmoid().cpu().numpy())
            quality.append(q.float().sigmoid().cpu().numpy())
            reg.append(r.float().cpu().numpy())
    count_labels = len(data['protocol']['labels']); count_values = len(data['protocol']['outcomes'])
    probabilities = np.concatenate(cls) if cls else np.empty((0, count_labels))
    standardized = np.concatenate(reg) if reg else np.empty((0, count_values))
    assessable = np.concatenate(quality) if quality else np.empty((0, count_values))
    center = np.asarray([data['normalization'][METHODS[data['method'][i]]]['center'] for i in ids], float).reshape(-1, count_values)
    scale = np.asarray([data['normalization'][METHODS[data['method'][i]]]['scale'] for i in ids], float).reshape(-1, count_values)
    return probabilities, standardized * scale + center, assessable


def balanced_epoch(data, rng):
    groups = []
    valid = (data['split'] == 0) & data['regression_mask'].any(1)
    primary = np.flatnonzero(valid & (data['method'] == 0))
    auxiliary = np.flatnonzero(valid & (data['method'] == 1))
    for ids, count in [(primary, len(primary)), (auxiliary, min(len(auxiliary), len(primary)))]:
        if not len(ids):
            continue
        trials = defaultdict(set); windows = Counter()
        for i in ids:
            m = data['windows'][i]; trials[m['subject']].add(m['trial_block'])
            windows[(m['subject'], m['trial_block'])] += 1
        weights = np.array([1 / (len(trials[data['windows'][i]['subject']]) * windows[(data['windows'][i]['subject'], data['windows'][i]['trial_block'])]) for i in ids])
        groups.append(rng.choice(ids, count, replace=True, p=weights / weights.sum()))
    if not groups:
        raise RuntimeError('No usable training windows')
    ids = np.concatenate(groups); rng.shuffle(ids)
    return ids


def training_loss(outputs, data, batch, device):
    logits, regression, evaluability = outputs
    target = torch.from_numpy(np.maximum(data['labels'][batch], 0).astype(np.float32)).to(device)
    cmask = torch.from_numpy(data['label_mask'][batch]).to(device)
    rmask = torch.from_numpy(data['regression_mask'][batch]).to(device)
    actual = torch.from_numpy(data['standardized'][batch]).to(device)
    pos = torch.from_numpy(data['positive_weights']).to(device)
    class_loss = (F.binary_cross_entropy_with_logits(logits.float(), target, pos_weight=pos, reduction='none') * cmask).sum() / cmask.sum().clamp_min(1)
    regression_loss = (F.huber_loss(regression.float(), actual, reduction='none') * rmask).sum() / rmask.sum().clamp_min(1)
    quality_target = torch.from_numpy(data['measurement_mask'][batch].astype(np.float32)).to(device)
    quality_loss = F.binary_cross_entropy_with_logits(evaluability.float(), quality_target)
    return class_loss + .5 * regression_loss + .2 * quality_loss


def save_checkpoint(path, model, optimizer, epoch, data, history, encoder_updated):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
        initial_checkpoint_sha256=CHECKPOINT_SHA, dataset_sha256=data['manifest']['dataset_sha256'],
        label_count=len(data['protocol']['labels']), outcome_count=len(data['protocol']['outcomes']),
        encoder_updated=encoder_updated, history=history, created_at=now()), temp)
    temp.replace(path)


def train(args):
    torch.set_num_threads(2)
    np.random.seed(SEED); torch.manual_seed(SEED)
    data = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.train.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.smoke_cpu:
        if (args.output/'checkpoints/best.pt').exists():
            raise RuntimeError('A selected checkpoint already exists; refusing to overwrite a completed run')
        assert data['manifest']['status'] == 'frozen', 'GPU training requires a frozen completed dataset'
        current = json.loads((SOURCE / 'production_status.json').read_text())
        assert current.get('phase') == 'pose_complete', 'Existing inference must finish naturally first'
        guard = (SOURCE / 'poses_sapiens2_1b/.inference.lock').open('a')
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert torch.cuda.is_available()
    device = torch.device('cpu' if args.smoke_cpu else 'cuda')
    model = GaitModel(len(data['protocol']['outcomes']), len(data['protocol']['labels'])).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.encoder.named_parameters()}
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=1e-5),
        dict(params=[p for p in model.parameters() if id(p) not in encoder_ids], lr=1e-4)], weight_decay=.01)
    rng = np.random.default_rng(SEED)
    if args.smoke_cpu:
        batch = balanced_epoch(data, rng)[:1]
        model.train(); started = time.monotonic()
        outputs = model(torch.from_numpy(data['x'][batch]), torch.from_numpy(data['context'][batch]), torch.from_numpy(data['method'][batch].astype(np.int64)))
        loss = training_loss(outputs, data, batch, device)
        assert torch.isfinite(loss)
        loss.backward(); gradient = sum(float(p.grad.float().norm()) for p in model.encoder.parameters() if p.grad is not None)
        assert gradient > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        changed = [name for name, p in model.encoder.named_parameters() if not torch.equal(initial[name], p.detach().cpu())]
        assert len(changed) > 10
        result = dict(status='cpu_training_step_passed', device='cpu', loss=float(loss.detach()),
                      encoder_gradient_norm_sum=gradient, changed_encoder_tensors=len(changed),
                      elapsed_s=time.monotonic()-started, gpu_used=False, production_training=False)
        smoke_path=args.output/'checkpoints/CPU_SMOKE_ONLY.pt'
        save_checkpoint(smoke_path,model,optimizer,0,data,[],True)
        write(args.output / 'cpu_smoke.json', result); print(json.dumps(result)); return
    latest_end = datetime.fromisoformat(args.latest_end).timestamp()
    start = time.monotonic(); deadline = min(time.time() + args.max_seconds, latest_end)
    if time.time() >= deadline:
        raise RuntimeError('Training cutoff has passed; current pose job was not interrupted')
    validation = np.flatnonzero((data['split'] == 1) & (data['method'] == 0) & data['regression_mask'].any(1))
    assert len(validation), 'No primary validation measurements available'
    history = []; best = -math.inf; stale = 0; steps = 0; trained_steps = 0
    use_amp = torch.cuda.is_bf16_supported()
    write(args.output / 'resolved_config.json', dict(seed=SEED, epochs=args.epochs, batch_size=args.batch_size,
        accumulation=args.accumulation, max_seconds=args.max_seconds, deadline_utc=args.latest_end,
        encoder_lr=1e-5, head_lr=1e-4, device=torch.cuda.get_device_name(),
        initial_checkpoint_sha256=CHECKPOINT_SHA, data=str(args.data), dataset_sha256=data['manifest']['dataset_sha256'],
        training_code_sha256=sha(Path(__file__)), encoder_code_sha256=sha(ROOT/'vendor/motionbert/lib/model/DSTformer.py'),
        method='one-epoch head warm-up then full-backbone fine-tuning', autocast='bfloat16' if use_amp else 'float32'))
    write(args.output/'training_status.json',dict(status='running',phase='head_warmup',updated_at=now(),
        epochs=0,encoder_optimizer_steps=0,deadline_utc=args.latest_end,pid=os.getpid()))
    for epoch in range(args.epochs):
        if time.time() >= deadline:
            break
        warmup = epoch == 0
        for p in model.encoder.parameters():
            p.requires_grad_(not warmup)
        model.train(); ids = balanced_epoch(data, rng); losses=[]; optimizer.zero_grad(set_to_none=True)
        epoch_start = time.monotonic(); pending = 0; last_progress=epoch_start
        for offset in range(0, len(ids), args.batch_size):
            if time.time() >= deadline or (warmup and time.monotonic()-epoch_start > 300):
                break
            batch = ids[offset:offset+args.batch_size]
            with torch.autocast('cuda', dtype=torch.bfloat16) if use_amp else nullcontext():
                outputs = model(torch.from_numpy(data['x'][batch]).to(device), torch.from_numpy(data['context'][batch]).to(device), torch.from_numpy(data['method'][batch].astype(np.int64)).to(device))
                loss = training_loss(outputs, data, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss; existing checkpoints retained')
            (loss / args.accumulation).backward(); pending += 1
            losses.append(float(loss.detach()))
            if pending == args.accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                steps += 1; trained_steps += int(not warmup); pending = 0
            if time.monotonic()-last_progress>=30:
                write(args.output/'training_status.json',dict(status='running',updated_at=now(),pid=os.getpid(),
                    phase='head_warmup' if warmup else 'encoder_finetuning',epoch=epoch+1,
                    windows_processed=min(offset+len(batch),len(ids)),epoch_windows=len(ids),
                    encoder_optimizer_steps=trained_steps,elapsed_s=time.monotonic()-start,
                    windows_per_second=(offset+len(batch))/(time.monotonic()-epoch_start),
                    loss=float(np.mean(losses[-30:]))))
                last_progress=time.monotonic()
        if pending:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(args.accumulation / pending)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            steps += 1; trained_steps += int(not warmup)
        # Save a recovery checkpoint before validation, which can itself take time.
        save_checkpoint(args.output/'checkpoints/last.pt', model, optimizer, epoch+1, data, history, trained_steps>0)
        if time.time() >= deadline:
            break
        write(args.output/'training_status.json',dict(status='running',phase='validation',epoch=epoch+1,
            updated_at=now(),pid=os.getpid(),encoder_optimizer_steps=trained_steps,elapsed_s=time.monotonic()-start))
        score = metrics(data, validation, predict(model, data, validation, device, args.batch_size))
        item = dict(epoch=epoch+1, warmup=warmup, loss=float(np.mean(losses)) if losses else None,
                    epoch_seconds=time.monotonic()-epoch_start, elapsed_s=time.monotonic()-start,
                    encoder_optimizer_steps=trained_steps, validation=score)
        history.append(item)
        if warmup:
            save_checkpoint(args.output/'checkpoints/warmup.pt', model, optimizer, epoch+1, data, history, False)
        elif score['selection_score'] is not None and score['selection_score'] > best:
            best = score['selection_score']; stale = 0
            save_checkpoint(args.output/'checkpoints/best.pt', model, optimizer, epoch+1, data, history, True)
        else:
            stale += int(not warmup)
        write(args.output/'training_history.json', history)
        write(args.output/'training_status.json', dict(status='running', **{k:v for k,v in item.items() if k!='validation'},
                                                      validation_score=score['selection_score'], updated_at=now()))
        print(json.dumps({k:v for k,v in item.items() if k!='validation'}), flush=True)
        if epoch >= 3 and stale >= 4:
            break
    changed = [name for name,p in model.encoder.named_parameters() if not torch.equal(initial[name], p.detach().cpu())]
    write(args.output/'backbone_update_audit.json', dict(updated_at=now(), changed_encoder_tensors=len(changed),
        encoder_optimizer_steps=trained_steps, changed_parameter_names=changed, actual_backbone_finetuning=bool(changed and trained_steps)))
    if not changed or not trained_steps:
        raise RuntimeError('No encoder updates completed; do not describe warm-up as fine-tuning')
    if not (args.output/'checkpoints/best.pt').exists():
        raise RuntimeError('Encoder updated but validation did not finish; last checkpoint is partial, not selected')
    write(args.output/'training_status.json', dict(status='complete', updated_at=now(), epochs=len(history),
        elapsed_s=time.monotonic()-start, encoder_optimizer_steps=trained_steps,
        best_validation_score=best, checkpoint=str(args.output/'checkpoints/best.pt')))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=OUT)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--smoke-cpu',action='store_true')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--max-seconds',type=float,default=7200)
    p.add_argument('--latest-end',default='2026-11-15T05:30:00+00:00')
    args=p.parse_args()
    try:
        train(args)
    except Exception as e:
        write(args.output/'training_status.json',dict(status='failed',updated_at=now(),error=type(e).__name__,message=str(e)))
        raise

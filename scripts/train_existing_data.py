"""Reproducible exploratory training on the accepted local Theia data.

No doctor conclusions are fabricated. Publisher gait categories, staged squat
labels and archived clinical error labels are different prediction tasks.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import softmax
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/training_all_data_20261114'
NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
         'right_knee', 'left_ankle', 'right_ankle', 'left_big_toe',
         'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel']
FPS, LENGTH, STRIDE = 15., 60, 30
SEEDS = [20261114, 20261115, 20261116]


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def status(stage, **kwargs):
    obj = dict(updated_at=now(), stage=stage, **kwargs)
    write(OUT / 'status.json', obj)
    print(json.dumps(obj), flush=True)


def load_csv_pose(path):
    frames = {}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            frame = int(row['source_frame_index'])
            if frame not in frames:
                frames[frame] = (float(row['timestamp_s']), np.full((23, 3), np.nan, np.float32))
            j = NAMES.index(row['joint_id'])
            frames[frame][1][j] = [float(row[k]) if row[k] else np.nan for k in ('x', 'y', 'confidence')]
    order = sorted(frames)
    assert len(order) > 0 and np.all(np.diff(order) == 1), path
    return np.array([frames[k][0] for k in order]), np.stack([frames[k][1] for k in order])


def normalize_pose(times, pose, bounds=None):
    """Per-record geometry; nearest-native-frame temporal sampling, no gap filling."""
    times = np.asarray(times, float)
    xy = pose[..., :2].copy()
    valid = np.isfinite(pose).all(-1) & (pose[..., 2] >= .3)
    if bounds:
        valid &= (xy[..., 0] >= 0) & (xy[..., 0] < bounds[0]) & (xy[..., 1] >= 0) & (xy[..., 1] < bounds[1])
    root_ok = valid[:, 11] & valid[:, 12]
    root = (xy[:, 11] + xy[:, 12]) / 2
    shoulder = (xy[:, 5] + xy[:, 6]) / 2
    scale_ok = root_ok & valid[:, 5] & valid[:, 6] & valid[:, 15] & valid[:, 16]
    scale_values = np.linalg.norm(shoulder - root, axis=1) + (
        np.linalg.norm(xy[:, 11] - xy[:, 15], axis=1) + np.linalg.norm(xy[:, 12] - xy[:, 16], axis=1)) / 2
    usable = scale_values[scale_ok & np.isfinite(scale_values) & (scale_values > 1)]
    if len(usable) == 0:
        raise ValueError('No valid body scale; do not manufacture normalized geometry')
    scale = float(np.median(usable))
    valid &= root_ok[:, None]
    norm = (xy - root[:, None]) / scale
    valid &= np.isfinite(norm).all(-1) & (np.abs(norm).max(-1) < 4)
    norm[~valid] = 0
    sampled = np.arange(times[0], times[-1] + 1e-7, 1 / FPS)
    right = np.searchsorted(times, sampled).clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(np.abs(times[left] - sampled) <= np.abs(times[right] - sampled), left, right)
    gap_ok = np.abs(times[nearest] - sampled) <= max(.51 * np.median(np.diff(times)), .001)
    arr = np.concatenate([norm[nearest], valid[nearest, :, None].astype(np.float32)], -1)
    arr[~gap_ok] = 0
    return arr.astype(np.float32), dict(scale_pixels=scale, native_frames=len(pose),
                                       resampled_frames=len(arr), valid_joint_fraction=float(arr[..., 2].mean()),
                                       native_start_s=float(times[0]), native_last_s=float(times[-1]))


def windows(a):
    if len(a) <= LENGTH:
        out = np.zeros((1, LENGTH, 23, 3), np.float32)
        out[0, :len(a)] = a
        return out, [0]
    starts = list(range(0, len(a) - LENGTH + 1, STRIDE))
    if starts[-1] != len(a) - LENGTH:
        starts.append(len(a) - LENGTH)
    return np.stack([a[s:s + LENGTH] for s in starts]), starts


def prepare():
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Existing immutable protocol: use train/report or a new output directory')
    OUT.mkdir(parents=True, exist_ok=True)
    records, excluded, all_windows, source_index = [], [], [], []
    offsets = []

    def add(cid, subject, task, label, path, times, pose, extra, bounds=None):
        try:
            a, quality = normalize_pose(times, pose, bounds)
        except ValueError as e:
            excluded.append(dict(id=cid, path=str(path.relative_to(ROOT)), reason=str(e)))
            return
        w, starts = windows(a)
        rec = dict(id=cid, subject=subject, task=task, label=label,
                   pose_path=str(path.relative_to(ROOT)), pose_sha256=sha(path),
                   windows=len(w), window_start_resampled_frame=starts, **quality, **extra)
        offsets.append(len(all_windows))
        source_index.extend([len(records)] * len(w))
        all_windows.extend(w)
        records.append(rec)

    reference = json.loads((ROOT / 'data/prepared/progait_20261114/reference_manifest.json').read_text())['rows']
    for r in reference:
        cid = r['clip_id']
        complete = json.loads((ROOT / 'data/prepared/progait_20261114' / cid / 'complete.json').read_text())
        path = ROOT / 'data/raw/ProGait/annotations' / r['scenario'] / (cid.split('_', 2)[2] + '_keypoints.npy.gz')
        if r['assessment_status'] == 'publisher_marks_not_for_analysis':
            excluded.append(dict(id=cid, path=str(path.relative_to(ROOT)), reason='Publisher explicitly marks recording not for analysis'))
            continue
        with gzip.open(path, 'rb') as f:
            pose = np.load(f, allow_pickle=False)
        if pose.shape != (complete['frames'], 23, 3):
            excluded.append(dict(id=cid, path=str(path.relative_to(ROOT)), reason='RGB / pose frame-count mismatch; unknown alignment'))
            continue
        category = r['gait_category_code']
        add(cid, 'PG_' + r['subject_id'], 'gait', category - 1 if category else None,
            path, np.arange(len(pose)) / 30., pose,
            dict(label_provenance='ProGait publisher CCC category; never a squat label',
                 scenario=r['scenario'], view=r['view'], filename_group_key=r['filename_group_key'],
                 source_sha256=complete['source_sha256'], source_reference_line=r['text_reference_line_1based']),
            (complete['width'], complete['height']))

    selection = json.loads((ROOT / 'data/poses/sapiens2_5b_full/analysis_selection.json').read_text())['clips']
    accepted_gym = []
    for r in selection:
        if not r['clip_id'].startswith('mendeley'):
            continue
        path = ROOT / 'data/poses/sapiens2_5b_full' / r['clip_id'] / 'poses.csv'
        if not r['included_in_provisional_export']:
            excluded.append(dict(id=r['clip_id'], path=str(path.relative_to(ROOT)), reason=r['analysis_status']))
            continue
        cid = r['clip_id']
        accepted_gym.append(path)
        t, p = load_csv_pose(path)
        add(cid, 'GYM_' + cid.split('_')[2], 'squat', int('_bad_' in cid), path, t, p,
            dict(label_provenance='Publisher staged good/bad label; health and clinical validity undocumented', domain='gym'))

    auxiliaries = [(ROOT / 'data/poses/ared_continuous_development/poses.csv', 'ARED_ADENOT', 'ARED_ADENOT', 'in_mission_ared')]
    for p in sorted((ROOT / 'data/poses/sapiens2_5b_gerst_full').glob('*/poses.csv')):
        auxiliaries.append((p, p.parent.name, 'ARED_GERST', 'in_mission_ared'))
    for p in sorted((ROOT / 'data/poses/sapiens2_5b_intake_full').glob('*/poses.csv')):
        auxiliaries.append((p, p.parent.name, p.parent.name.split('_')[0],
                            'ground_ared' if '_GROUND_' in p.parent.name else 'in_mission_ared'))
    for p, cid, subject, domain in auxiliaries:
        t, a = load_csv_pose(p)
        add(cid, subject, 'auxiliary', None, p, t, a,
            dict(label_provenance='None; unlabeled reconstruction only', domain=domain))

    # Document every local pose CSV rather than silently using duplicate pilot/full frames.
    selected_paths = {r['pose_path'] for r in records}
    explicit_holds = {r['path']: r['reason'] for r in excluded}
    coverage = []
    for p in sorted((ROOT / 'data/poses').rglob('poses.csv')):
        rel = str(p.relative_to(ROOT))
        coverage.append(dict(path=rel, included=rel in selected_paths,
                             reason=('included' if rel in selected_paths else explicit_holds.get(rel,
                                     'Pilot / comparison / overlapping parent / nonaccepted auxiliary; not an independent accepted source'))))
    x = np.stack(all_windows).astype(np.float32)
    assert np.isfinite(x).all()
    np.savez_compressed(OUT / 'training_windows.npz', x=x, source=np.array(source_index, np.int32))
    write(OUT / 'data_manifest.json', dict(records=records, exclusions=excluded, pose_file_coverage=coverage,
          notes=['No sparse pilot or duplicated parent poses added as independent data.',
                 'Raw videos without an accepted full pose track cannot be input to this pose model.',
                 'ProGait blank-reference recordings are eligible only for unlabeled training.',
                 'Publisher not-for-analysis recordings remain excluded.',
                 'KERAAL archived numerical features train separate models; not converted into fake pose sequences.']))
    shutil.copy2(ROOT / 'results/local_execution_20261114/clinical_features.json', OUT / 'clinical_features.json')
    jobs = []
    for task in ('gait', 'squat'):
        for subject in sorted({r['subject'] for r in records if r['task'] == task and r['label'] is not None}):
            test = [i for i, r in enumerate(records) if r['subject'] == subject and r['task'] == task and r['label'] is not None]
            for arm in (['with_rehab'] if task == 'gait' else ['without_rehab', 'with_rehab']):
                train = [i for i, r in enumerate(records) if r['subject'] != subject and (arm == 'with_rehab' or r['task'] != 'gait')]
                assert not ({records[i]['subject'] for i in train} & {subject})
                jobs.append(dict(id=f'{task}_{subject}_{arm}', task=task, held_out_subject=subject, arm=arm,
                                 train_source_indices=train, test_source_indices=test))
    jobs.append(dict(id='final_all_data', task='all', held_out_subject=None, arm='with_rehab',
                     train_source_indices=list(range(len(records))), test_source_indices=[]))
    protocol = dict(recorded_at=now(), phase='exploratory, fixed before fitting',
        objective='Train usable research models on all compatible existing data and measure held-out-person performance.',
        target_limitation='No independent ARED decisions exist yet. In-mission clinical benefit cannot be measured in this run.',
        seeds=SEEDS, label_maps={'gait': {str(i): f'CCC {i+1}' for i in range(9)}, 'squat': {'0': 'publisher good', '1': 'publisher bad'}},
        preprocessing=dict(joints=NAMES, confidence_gate=.3, centering='per-frame midpoint of hips',
            scale='per-record median torso length plus mean hip-to-ankle distance on valid frames',
            fps=FPS, window_frames=LENGTH, stride_frames=STRIDE,
            sampling='nearest native frame, reject missing root and large time gaps; zero coordinates plus binary validity mask',
            padding='zero values and invalid mask, never repeated valid frames',
            note='Explicit temporal resampling for learning; originals and native pose files preserved'),
        architecture='Shared 96-channel residual temporal CNN, dilations 1/2/4/8; separate 9-category gait and binary squat heads; masked pose decoder',
        training=dict(pretrain_steps=300, supervised_steps=600, batch_size=64, learning_rate=.001,
            optimizer='AdamW', weight_decay=.001, gradient_clip=1.,
            pretrain_loss='Smooth L1 reconstruction of 25% randomly hidden valid joint positions',
            supervised_loss='Equal gait and squat task weights; training-only inverse class frequency cross entropy',
            auxiliary_loss_weight=.1,
            source_sampling='Cycle shuffled complete window passes for 300 pretraining updates; every eligible window is visited. Supervised batches cycle source permutations, then draw windows uniformly within a source.',
            augmentation='Random ±10 degree image-plane rotation and 0.9-1.1 scale; mask unchanged',
            stopping='Fixed budget, no outer-test monitoring or selection'),
        evaluation=dict(split='Leave one person out, including all views/trials/windows and unlabeled samples from that person',
            predictions='Average window probabilities per source, then average three seed models; score original source videos, not cuts',
            primary_metrics=['balanced_accuracy', 'macro_f1'], additional_metrics=['accuracy', 'per_person_accuracy', 'confusion_matrix'],
            baselines=['training majority class', 'L2 multinomial logistic regression on normalized pose/velocity/angle summary features'],
            transfer='Identical squat outer folds and budgets, with versus without all ProGait rehabilitation data. Different task heads; never relabel gait as squat.',
            uncertainty='Report all person folds and seed range; four gait and three squat people are insufficient for a confirmatory generalization claim',
            prior_exposure='KERAAL patients, gym clips and ARED footage already used for development; exploratory data, not a fresh clinical test',
            mission_role='Unlabeled ARED recordings used in reconstruction. Future labels on these recordings would be transductive evaluation, not unseen mission generalization'),
        clinical=dict(model='Separate participant/class-weighted L2 logistic regression, lambda .1, threshold .5',
            evaluation='Leave one patient out, healthy-only baseline vs healthy plus other patients',
            final_fit='All archived eligible rows; preserve ELK geometry alternatives as separate models',
            caveat='Archived numerical features and error-specific labels; not new clinical RGB annotation or squat outcomes'),
        files={name: sha(OUT / name) for name in ['data_manifest.json', 'training_windows.npz', 'clinical_features.json']},
        runner_sha256=sha(__file__), jobs=jobs,
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda))
    shutil.copy2(__file__, OUT / 'executed_runner.py')
    write(OUT / 'protocol.json', protocol)
    status('prepared', sources=len(records), labeled_sources=sum(r['label'] is not None for r in records),
           windows=len(x), native_frames=sum(r['native_frames'] for r in records), exclusions=len(excluded),
           model_fits=len(jobs) * len(SEEDS))


class Residual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels), nn.SiLU(), nn.Dropout(.1),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels), nn.SiLU())

    def forward(self, x):
        return x + self.layers(x)


class PoseTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(69, 96, 1)
        self.blocks = nn.Sequential(*[Residual(96, d) for d in (1, 2, 4, 8)])
        self.decoder = nn.Conv1d(96, 46, 1)
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(192, 64), nn.SiLU(), nn.Dropout(.2), nn.Linear(64, c))
                                   for k, c in [('gait', 9), ('squat', 2)]})

    def forward(self, x, task=None):
        h = self.blocks(self.input(x.reshape(len(x), LENGTH, 69).transpose(1, 2)))
        if task == 'reconstruct':
            return self.decoder(h).transpose(1, 2).reshape(len(x), LENGTH, 23, 2)
        weight = (x[..., 2].sum(-1) > 0).to(x.dtype)[:, None]
        den = weight.sum(-1).clamp(min=1)
        mean = (h * weight).sum(-1) / den
        sd = (((h - mean[..., None]).square() * weight).sum(-1) / den + 1e-6).sqrt()
        features = torch.cat([mean, sd], -1)
        return features if task is None else self.heads[task](features)


def augment(x):
    x = x.clone()
    theta = (torch.rand(len(x), device=x.device) * 2 - 1) * math.pi / 18
    scale = .9 + .2 * torch.rand(len(x), device=x.device)
    c, s = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
    old = x[..., :2].clone()
    x[..., 0] = (c * old[..., 0] - s * old[..., 1]) * scale[:, None, None]
    x[..., 1] = (s * old[..., 0] + c * old[..., 1]) * scale[:, None, None]
    return x


def reconstruction_loss(model, x):
    hidden = (torch.rand(x.shape[:-1], device=x.device) < .25) & (x[..., 2] > 0)
    masked = x.clone()
    masked[hidden] = 0
    pred = model(masked, 'reconstruct')
    error = F.smooth_l1_loss(pred, x[..., :2], reduction='none').sum(-1)
    return (error * hidden).sum() / (2 * hidden.sum().clamp(min=1))


class SourceSampler:
    def __init__(self, sources, source_index, rng):
        self.sources = np.array(sources)
        self.by_source = {i: np.where(source_index == i)[0] for i in sources}
        self.rng, self.remaining, self.seen = rng, [], set()

    def batch(self, n):
        ids = []
        while len(ids) < n:
            if not self.remaining:
                self.remaining = self.rng.permutation(self.sources).tolist()
            ids.append(self.remaining.pop())
        self.seen.update(ids)
        return np.array([self.rng.choice(self.by_source[i]) for i in ids]), ids


def metrics(y, pred, classes, subjects=None):
    y, pred = np.asarray(y), np.asarray(pred)
    cm = np.zeros((classes, classes), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    support = cm.sum(1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(classes, float), where=support > 0)
    f1 = np.divide(2 * np.diag(cm), support + cm.sum(0), out=np.zeros(classes, float), where=(support + cm.sum(0)) > 0)
    present = support > 0
    result = dict(n=len(y), accuracy=float(np.mean(y == pred)), balanced_accuracy=float(recall[present].mean()),
                  macro_f1=float(f1[present].mean()), confusion_matrix=cm.tolist(), support=support.tolist(),
                  recall_by_class=recall.tolist(), macro_average_rule='Classes present in evaluation reference')
    if subjects is not None:
        result['per_person_accuracy'] = {s: float(np.mean(y[np.array(subjects) == s] == pred[np.array(subjects) == s])) for s in sorted(set(subjects))}
        result['mean_person_accuracy'] = float(np.mean(list(result['per_person_accuracy'].values())))
    return result


def predict(model, x, source_index, indices, task):
    model.eval()
    wi = np.where(np.isin(source_index, indices))[0]
    out = []
    with torch.no_grad():
        for start in range(0, len(wi), 128):
            out.append(model(x[wi[start:start + 128]], task).softmax(-1).cpu().numpy())
    p = np.concatenate(out)
    return {i: p[source_index[wi] == i].mean(0) for i in indices}


def train_one(job, seed, x, source_index, records, protocol):
    name = f'{job["id"]}_seed{seed}'
    directory = OUT / 'fits' / name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'complete.json').exists():
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = PoseTCN().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    train_sources = job['train_source_indices']
    assert not any(records[i]['subject'] == job['held_out_subject'] for i in train_sources)
    wi = np.where(np.isin(source_index, train_sources))[0]
    start_time = time.monotonic()
    history = []
    remaining, seen_windows, losses = [], set(), []
    model.train()
    for step in range(protocol['training']['pretrain_steps']):
        batch = []
        while len(batch) < 64:
            if not remaining:
                remaining = rng.permutation(wi).tolist()
            batch.append(remaining.pop())
        seen_windows.update(batch)
        loss = reconstruction_loss(model, augment(x[batch]))
        assert torch.isfinite(loss), 'nonfinite pretraining loss'
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            history.append(dict(phase='reconstruction', step=step + 1, mean_loss=float(np.mean(losses[-100:]))))
            status('training', fit=name, **history[-1], elapsed_s=round(time.monotonic() - start_time, 1))
    assert seen_windows == set(wi), 'not all eligible training windows visited'
    samplers, class_weights = {}, {}
    for task, classes in [('gait', 9), ('squat', 2)]:
        ids = [i for i in train_sources if records[i]['task'] == task and records[i]['label'] is not None]
        if ids:
            samplers[task] = SourceSampler(ids, source_index, rng)
            counts = np.bincount([records[i]['label'] for i in ids], minlength=classes)
            w = 1 / np.maximum(counts, 1)
            class_weights[task] = torch.tensor(w / w.mean(), dtype=torch.float32, device='cuda')
    aux_sampler = SourceSampler(train_sources, source_index, rng)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    cumulative = []
    for step in range(protocol['training']['supervised_steps']):
        model.train()
        losses = []
        for task, sampler in samplers.items():
            batch, ids = sampler.batch(32)
            labels = torch.tensor([records[i]['label'] for i in ids], device='cuda')
            losses.append(F.cross_entropy(model(augment(x[batch]), task), labels, weight=class_weights[task]))
        aux_ids, _ = aux_sampler.batch(32)
        loss = torch.stack(losses).mean() + .1 * reconstruction_loss(model, augment(x[aux_ids]))
        assert torch.isfinite(loss), 'nonfinite supervised loss'
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        cumulative.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            h = dict(phase='supervised', step=step + 1, mean_loss=float(np.mean(cumulative[-100:])))
            history.append(h)
            status('training', fit=name, **h, elapsed_s=round(time.monotonic() - start_time, 1))
    assert all(set(s.sources) == s.seen for s in samplers.values())
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    checkpoint = dict(state_dict=state, architecture='PoseTCN-v1', seed=seed,
                      training_source_ids=[records[i]['id'] for i in train_sources],
                      protocol_sha256=sha(OUT / 'protocol.json'), label_maps=protocol['label_maps'])
    torch.save(checkpoint, directory / 'model.pt')
    write(directory / 'history.json', history)
    predictions = []
    if job['test_source_indices']:
        prob = predict(model, x, source_index, job['test_source_indices'], job['task'])
        for i, p in prob.items():
            predictions.append(dict(id=records[i]['id'], subject=records[i]['subject'], y=records[i]['label'],
                                    predicted=int(p.argmax()), probabilities=p.tolist(), seed=seed,
                                    task=job['task'], arm=job['arm'], fit=name))
    else:
        # These scores are explicitly training-domain outputs, never held-out metrics.
        for task in ('gait', 'squat'):
            ids = [i for i, r in enumerate(records) if r['task'] == task or (task == 'squat' and r['task'] == 'auxiliary')]
            prob = predict(model, x, source_index, ids, task)
            for i, p in prob.items():
                predictions.append(dict(id=records[i]['id'], subject=records[i]['subject'], y=records[i]['label'],
                    predicted=int(p.argmax()), probabilities=p.tolist(), seed=seed, task=task, arm=job['arm'], fit=name,
                    scope='all-data fit; NOT held-out', decision='abstain_pending_independent_reference' if records[i]['task'] == 'auxiliary' else 'research_prediction'))
    write(directory / 'predictions.json', predictions)
    # Verify serialization gives identical logits, including the joint schema dimensions.
    model.eval()
    reload_model = PoseTCN().cuda().eval()
    reload_model.load_state_dict(torch.load(directory / 'model.pt', weights_only=True)['state_dict'])
    with torch.no_grad():
        assert torch.equal(model(x[wi[:2]], 'squat'), reload_model(x[wi[:2]], 'squat'))
    write(directory / 'complete.json', dict(completed_at=now(), seed=seed, job=job['id'],
        checkpoint_sha256=sha(directory / 'model.pt'), training_sources=len(train_sources),
        pretraining_windows=len(wi), all_supervised_sources_seen=True, checkpoint_reload_identical=True,
        parameter_count=sum(p.numel() for p in model.parameters()), elapsed_s=time.monotonic() - start_time))
    del model, reload_model, optimizer
    torch.cuda.empty_cache()


def feature_matrix(x, source_index, records):
    """Low-capacity comparison using only geometry, motion and missingness."""
    features = []
    for i in range(len(records)):
        a = x[source_index == i]
        valid = a[..., 2] > 0
        pos = a[..., :2].copy()
        pos[~valid] = np.nan
        flat = pos.reshape(-1, 23, 2)
        # Medians and spread ignore padded/missing points, not source IDs or filenames.
        quantiles = np.nanquantile(flat, [.1, .5, .9], axis=0).reshape(-1)
        velocity = np.diff(pos, axis=1) * FPS
        speed = np.nanmedian(np.abs(velocity), axis=(0, 1)).reshape(-1)
        features.append(np.nan_to_num(np.r_[quantiles, speed, valid.mean(axis=(0, 1))], nan=0.))
    return np.asarray(features)


def logistic_fit(features, labels, classes):
    counts = np.bincount(labels, minlength=classes)
    weights = 1 / np.maximum(counts[labels], 1)
    weights /= weights.sum()
    mean = np.average(features, axis=0, weights=weights)
    sd = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weights))
    sd[sd < 1e-5] = 1
    z = np.c_[np.ones(len(features)), (features - mean) / sd]
    target = np.eye(classes)[labels]
    def objective(flat):
        w = flat.reshape(z.shape[1], classes)
        p = softmax(z @ w, axis=1)
        loss = -np.sum(weights[:, None] * target * np.log(np.maximum(p, 1e-12))) + .05 * np.sum(w[1:] ** 2)
        grad = z.T @ ((p - target) * weights[:, None])
        grad[1:] += .1 * w[1:]
        return loss, grad.ravel()
    fit = minimize(objective, np.zeros(z.shape[1] * classes), jac=True, method='L-BFGS-B', options={'maxiter': 1500, 'gtol': 1e-7})
    if not fit.success:
        raise RuntimeError(f'Baseline optimizer failed: {fit.message}')
    return dict(mean=mean.tolist(), sd=sd.tolist(), coefficients=fit.x.reshape(z.shape[1], classes).tolist(),
                converged=True, iterations=fit.nit)


def baselines(x, source_index, records, jobs):
    features = feature_matrix(x, source_index, records)
    np.save(OUT / 'baseline_features.npy', features)
    output, saved = [], {}
    for job in jobs:
        if job['task'] == 'all' or job['arm'] == 'without_rehab':
            continue
        task = job['task']
        classes = 9 if task == 'gait' else 2
        ids = [i for i in job['train_source_indices'] if records[i]['task'] == task and records[i]['label'] is not None]
        labels = np.array([records[i]['label'] for i in ids])
        fit = logistic_fit(features[ids], labels, classes)
        saved[job['id']] = dict(**fit, training_ids=[records[i]['id'] for i in ids])
        test = job['test_source_indices']
        z = np.c_[np.ones(len(test)), (features[test] - fit['mean']) / fit['sd']]
        probs = softmax(z @ np.array(fit['coefficients']), axis=1)
        majority = int(np.bincount(labels, minlength=classes).argmax())
        for i, p in zip(test, probs):
            for method, pred in [('feature_logistic', int(p.argmax())), ('majority', majority)]:
                output.append(dict(id=records[i]['id'], subject=records[i]['subject'], task=task, method=method,
                    y=records[i]['label'], predicted=pred,
                    probabilities=p.tolist() if method == 'feature_logistic' else None))
    for task, classes in [('gait', 9), ('squat', 2)]:
        ids = [i for i, r in enumerate(records) if r['task'] == task and r['label'] is not None]
        saved['final_' + task] = dict(**logistic_fit(features[ids], np.array([records[i]['label'] for i in ids]), classes),
                                      training_ids=[records[i]['id'] for i in ids])
    write(OUT / 'baseline_models.json', saved)
    write(OUT / 'baseline_predictions.json', output)


def clinical():
    from run_local_experiments import fit, metrics as clinical_metrics
    from scipy.special import expit
    data = json.loads((OUT / 'clinical_features.json').read_text())
    models, predictions, results = {}, [], {}
    for name, rows in data.items():
        results[name] = {}
        for arm in ['healthy_only', 'healthy_plus_clinical']:
            arm_preds = []
            for subject in sorted({r['subject'] for r in rows if r['cohort'] == 'patient'}):
                train = [r for r in rows if r['subject'] != subject and (arm != 'healthy_only' or r['cohort'] != 'patient')]
                test = [r for r in rows if r['subject'] == subject]
                model = fit(train, .1)
                models[f'{name}_{arm}_{subject}'] = model
                z = (np.asarray([r['features'] for r in test]) - model['mean']) / model['sd']
                prob = expit(np.c_[np.ones(len(z)), z] @ np.asarray(model['coefficients']))
                for r, p in zip(test, prob):
                    arm_preds.append(dict(id=r['id'], subject=subject, y=r['y'], prediction=int(p >= .5),
                                          probability=float(p), task=name, arm=arm))
            predictions.extend(arm_preds)
            results[name][arm] = clinical_metrics(arm_preds)
        models['final_' + name] = fit(rows, .1)
    write(OUT / 'clinical_models.json', models)
    write(OUT / 'clinical_predictions.json', predictions)
    write(OUT / 'clinical_results.json', results)


def aggregate():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    records = json.loads((OUT / 'data_manifest.json').read_text())['records']
    all_preds = []
    for job in protocol['jobs']:
        for seed in SEEDS:
            p = OUT / 'fits' / f'{job["id"]}_seed{seed}'
            assert (p / 'complete.json').exists(), p
            all_preds.extend(json.loads((p / 'predictions.json').read_text()))
    ensemble = []
    groups = {}
    for p in all_preds:
        key = p['id'], p['task'], p['arm'], p.get('scope', 'held-out-person')
        groups.setdefault(key, []).append(p)
    for (cid, task, arm, scope), group in groups.items():
        assert len(group) == len(SEEDS)
        prob = np.mean([r['probabilities'] for r in group], axis=0)
        ensemble.append(dict(id=cid, task=task, arm=arm, scope=scope,
            subject=group[0]['subject'], y=group[0]['y'], predicted=int(prob.argmax()), probabilities=prob.tolist(),
            seed_predictions=[r['predicted'] for r in group],
            decision=group[0].get('decision', 'research_prediction')))
    write(OUT / 'predictions_ensemble.json', ensemble)
    results = {}
    for task, classes in [('gait', 9), ('squat', 2)]:
        results[task] = {}
        for arm in (['with_rehab'] if task == 'gait' else ['without_rehab', 'with_rehab']):
            selected = [r for r in ensemble if r['task'] == task and r['arm'] == arm and r['scope'] == 'held-out-person']
            results[task][arm] = metrics([r['y'] for r in selected], [r['predicted'] for r in selected], classes, [r['subject'] for r in selected])
            results[task][arm]['per_seed'] = {}
            for seed in SEEDS:
                selected_seed = [r for r in all_preds if r['task'] == task and r['arm'] == arm and r['seed'] == seed and 'scope' not in r]
                results[task][arm]['per_seed'][str(seed)] = metrics([r['y'] for r in selected_seed], [r['predicted'] for r in selected_seed], classes, [r['subject'] for r in selected_seed])
        for method in ('majority', 'feature_logistic'):
            selected = [r for r in json.loads((OUT / 'baseline_predictions.json').read_text()) if r['task'] == task and r['method'] == method]
            results[task][method] = metrics([r['y'] for r in selected], [r['predicted'] for r in selected], classes, [r['subject'] for r in selected])
    squat = {arm: {r['id']: r for r in ensemble if r['task'] == 'squat' and r['arm'] == arm and r['scope'] == 'held-out-person'} for arm in ('without_rehab', 'with_rehab')}
    paired = []
    for cid, a in squat['without_rehab'].items():
        b = squat['with_rehab'][cid]
        before, after = a['predicted'] == a['y'], b['predicted'] == b['y']
        paired.append(dict(id=cid, subject=a['subject'], y=a['y'], without_rehab=a['predicted'], with_rehab=b['predicted'],
                           effect='helped' if after and not before else 'harmed' if before and not after else 'unchanged_correct' if after else 'unchanged_wrong'))
    results['squat_transfer'] = dict(paired_cases=paired, counts=dict(Counter(r['effect'] for r in paired)),
        interpretation='Exploratory transfer to five staged terrestrial squat source videos from three participants; not an in-mission result.')
    results['coverage'] = dict(pose_sources=len(records), native_pose_frames=sum(r['native_frames'] for r in records),
        labeled_gait_sources=sum(r['task'] == 'gait' and r['label'] is not None for r in records),
        labeled_squat_sources=sum(r['task'] == 'squat' and r['label'] is not None for r in records),
        unlabeled_sources=sum(r['label'] is None for r in records), final_checkpoint_seeds=SEEDS)
    results['mission'] = dict(decision_accuracy=None, rehabilitation_benefit=None,
                              reason='No independent ARED decision references; predictions abstain from operational conclusions.')
    write(OUT / 'results.json', results)
    write(OUT / 'complete.json', dict(completed_at=now(), protocol_sha256=sha(OUT / 'protocol.json'),
        model_fits=len(protocol['jobs']) * len(SEEDS), results_sha256=sha(OUT / 'results.json'),
        checkpoints=[str((OUT / 'fits' / f'final_all_data_seed{s}' / 'model.pt').relative_to(ROOT)) for s in SEEDS]))
    status('complete', **results['coverage'])


def tests():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(80, 23, 3)).astype(np.float32) * 100 + 300
    a[..., 2] = 1
    t = np.arange(80) / 30
    x, _ = normalize_pose(t, a)
    b = a.copy()
    b[..., :2] = 2 * b[..., :2] + np.array([500, -20])
    z, _ = normalize_pose(t, b)
    assert np.allclose(x, z, atol=2e-6), 'translation / scale normalization'
    w, starts = windows(x)
    assert len(w) == 1 and not w[:, len(x):, :, 2].any(), 'short padding mask'
    a[12, 11, 2] = 0
    z, _ = normalize_pose(t, a)
    assert z[6, :, 2].sum() == 0, 'missing hip suppresses frame'
    m = metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert m['accuracy'] == .75 and m['balanced_accuracy'] == .75
    print('Preprocessing / metric checks passed', flush=True)


def train():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if sha(__file__) != protocol['runner_sha256']:
        raise RuntimeError('Runner changed after protocol: execute frozen runner or record amendment before continuing')
    for name, digest in protocol['files'].items():
        assert sha(OUT / name) == digest, f'Changed training input {name}'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tests()
    records = json.loads((OUT / 'data_manifest.json').read_text())['records']
    z = np.load(OUT / 'training_windows.npz', allow_pickle=False)
    cpu, source_index = z['x'], z['source']
    baselines(cpu, source_index, records, protocol['jobs'])
    clinical()
    x = torch.from_numpy(cpu).cuda()
    status('training_started', gpu=torch.cuda.get_device_name(), windows=len(x), fits=len(protocol['jobs']) * len(SEEDS))
    for job in protocol['jobs']:
        for seed in SEEDS:
            train_one(job, seed, x, source_index, records, protocol)
    aggregate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'aggregate', 'test'])
    args = parser.parse_args()
    if args.action == 'prepare':
        tests()
        prepare()
    elif args.action == 'train':
        try:
            train()
        except Exception as e:
            status('failed', error=repr(e))
            raise
    elif args.action == 'aggregate':
        aggregate()
    else:
        tests()

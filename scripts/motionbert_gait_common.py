"""Shared, versioned definitions for the deadline MotionBERT experiment."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/gait_metrics_20261114'
OUT = ROOT / 'results/motionbert_gait_finetune_v1'
SEED = 20261114
VERSION = 'automatic-gait-teacher-v1'
WINDOW_SECONDS = 1.0
STRIDE_SECONDS = .5
INPUT_FPS = 30
INPUT_FRAMES = 30
CHECKPOINT_SHA = '597c708bd585646cc6961df057ef73634edb83a4d5c1155d98a4cc0c04e81b98'
SAPIENS_SHA = '2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f'
METHODS = ['sapiens', 'publisher']
SPLITS = ['train', 'validation', 'test', 'exploratory']
REGIONS = ['knees', 'knees', 'knees', 'knees', 'trunk', 'trunk', 'hips', 'shoulders', 'arms', 'arms']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def definitions():
    frozen = json.loads((SOURCE / 'comparison_manifest_frozen.json').read_text())
    outcomes = frozen['outcomes']
    assert len(outcomes) == len(REGIONS) == 10
    labels = []
    for i, outcome in enumerate(outcomes):
        labels.append(dict(id=outcome['id'] + '_high', outcome=i, direction='high', region=REGIONS[i]))
        if outcome['id'] in ('knee_rom95_mean', 'knee_p95_mean', 'arm_rom95_mean'):
            labels.append(dict(id=outcome['id'] + '_low', outcome=i, direction='low', region=REGIONS[i]))
    return frozen, outcomes, labels


def subject_splits(rows):
    """One metadata-only split, identical across pose methods and all crops."""
    rng = np.random.default_rng(SEED)
    mapping = {}
    for source, counts in [('progait', (2, 1)), ('ljmu360', (7, 2))]:
        subjects = sorted({r['subject'] for r in rows if r['source'] == source and r['subject']})
        ordered = list(rng.permutation(subjects))
        a, b = counts
        for i, subject in enumerate(ordered):
            mapping[str(subject)] = 'train' if i < a else 'validation' if i < a + b else 'test'
    for row in rows:
        if row['source'] == 'lbmc':
            mapping[row['subject']] = 'train'
    return mapping


def window_bounds(t, lo, hi, fps):
    """One-second windows within the existing >=1s quality segments."""
    first = float(t[lo])
    end = float(t[hi - 1]) + 1 / fps
    if end - first < WINDOW_SECONDS - 1e-5:
        return []
    starts = list(np.arange(first, end - WINDOW_SECONDS + 1e-5, STRIDE_SECONDS))
    if not starts or abs(starts[-1] - (end - WINDOW_SECONDS)) > 1e-5:
        starts.append(end - WINDOW_SECONDS)
    return [(float(s), float(s + WINDOW_SECONDS)) for s in starts]


def reference_fence(units):
    if len(units) < 5:
        return None
    q1, q3 = np.percentile(list(units.values()), [25, 75])
    iqr = q3 - q1
    if iqr <= 1e-9:
        return None
    return dict(lower=float(q1 - 1.5 * iqr), upper=float(q3 + 1.5 * iqr),
                buffer=float(.1 * iqr), q25=float(q1), q75=float(q3),
                participants=sorted(units), participant_count=len(units))


def classify(value, fence, direction):
    if not np.isfinite(value) or fence is None:
        return -1
    boundary = fence['upper'] if direction == 'high' else fence['lower']
    delta = (value - boundary) * (1 if direction == 'high' else -1)
    if delta > fence['buffer']:
        return 1
    if delta <= -fence['buffer']:
        return 0
    return -1


def pose_adapter(xy, score, valid, timestamps, start, width, height):
    """Named 23->H36M adapter; nearest native sampling, no gap imputation."""
    target = start + np.arange(INPUT_FRAMES) / INPUT_FPS
    right = np.searchsorted(timestamps, target).clip(0, len(timestamps) - 1)
    left = np.maximum(0, right - 1)
    ids = np.where(abs(timestamps[left] - target) <= abs(timestamps[right] - target), left, right)
    dt = np.median(np.diff(timestamps))
    near = abs(timestamps[ids] - target) <= max(.51 * dt, .001)
    native_valid = valid[ids] & near[:, None]
    points = xy[ids].astype(np.float32)
    confidence = np.where(native_valid, np.clip(score[ids], 0, 1), 0).astype(np.float32)
    result = np.zeros((INPUT_FRAMES, 17, 3), np.float32)

    def assign(destination, indices):
        ok = native_valid[:, indices].all(1)
        result[:, destination, :2] = np.mean(points[:, indices], 1)
        result[:, destination, 2] = np.where(ok, confidence[:, indices].min(1), 0)

    # hips, legs, spine, thorax, nose, approximate head centre, arms
    mapping = {0: [11, 12], 1: [12], 2: [14], 3: [16], 4: [11], 5: [13], 6: [15],
               7: [11, 12, 5, 6], 8: [5, 6], 9: [0], 10: [1, 2],
               11: [5], 12: [7], 13: [9], 14: [6], 15: [8], 16: [10]}
    for joint, indices in mapping.items():
        assign(joint, indices)
    result[..., :2] = (result[..., :2] - [width / 2, height / 2]) / (min(width, height) / 2)
    finite = np.isfinite(result).all(-1)
    result[~finite] = 0
    result[result[..., 2] <= 0] = 0
    torso_coverage = (result[:, [0, 8], 2] > 0).all(1).mean()
    body_coverage = (result[:, [0, 1, 2, 3, 4, 5, 6, 8, 11, 14], 2] > 0).mean()
    return result, ids, bool(torso_coverage >= .8 and body_coverage >= .6)

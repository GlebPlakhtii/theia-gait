"""Audit delivered frame alignment and supported pose coverage; no accuracy claim."""
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'results/gait_selected_paper_20261115'
AUDIT = ROOT / 'audit/astronaut_treadmill_demo_20261115'

page = json.loads((WEB / 'page_data.json').read_text())
before = json.loads((AUDIT / 'page_data_before.json').read_text())
assert [c for c in page['clips'] if c['kind'] != 'astronaut'] == before['clips']
assert page['results'] == before['results']
processed = json.loads((WEB / 'astronaut_treadmill_provenance.json').read_text())
checks = []
for c in page['clips']:
    if c['kind'] != 'astronaut':
        continue
    folder = WEB / 'astronauts' / c['clip_id']
    p = json.loads((folder / 'viewer.json').read_text())
    source = json.loads((folder / 'source_provenance.json').read_text())
    record = next(r for r in processed if r['clip_id'] == c['clip_id'])
    names = p['joint_names']
    required = [names.index('nose')] + [names.index(s + '_' + j) for s in ['left', 'right'] for j in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']]
    q = np.array(p['quality_mask'], bool)
    xy = np.array(p['xy_pixels'])
    scores = np.array(p['confidence'])
    assert q[:, required].all(), 'Retained interval has a missing displayed joint'
    assert (scores[:, required] >= .5).all()
    assert np.isfinite(xy).all()
    assert c['windows'] == [] and c['clinical_label'] is None
    assert p['model'] == c['pose_model'] == 'Sapiens2-5B'
    n = len(p['timestamp_s'])
    assert n == c['video']['frames'] == len(p['clip_frame'])
    assert p['clip_frame'] == list(range(n))
    assert p['original_source_frame'] == list(range(c['video']['source_first_frame'], c['video']['source_first_frame'] + n))
    assert p['timestamp_s'][0] == 0 and np.all(np.diff(p['timestamp_s']) > 0)
    rawdir = Path(record['inference_run']).parent
    schema = json.loads((rawdir / 'joint_schema.json').read_text())['body_feet_23']
    selection = [int(k) for name in names for k, v in schema.items() if v == name]
    raw = np.load(rawdir / c['clip_id'] / 'raw_308.npz')
    start, end = record['trim_start'], record['trim_end_exclusive']
    assert np.allclose(xy, raw['keypoints'][start:end, selection], atol=.00051)
    assert np.allclose(p['timestamp_s'], raw['timestamp_s'][start:end] - raw['timestamp_s'][start], atol=1e-8)
    cap = cv2.VideoCapture(str(folder / 'video.mp4')); decoded = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        assert image.shape[:2] == (c['video']['height'], c['video']['width'])
        decoded += 1
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    assert decoded == n and abs(fps - c['video']['fps']) < 1e-5
    checks.append(dict(clip=c['clip_id'], delivered_frames=n, seconds=n / fps,
        displayed_joints=len(required), supported_every_frame=True,
        source_frames=[c['video']['source_first_frame'], c['video']['source_first_frame'] + n - 1],
        retained_input_frames=[start, end - 1], pose_matches_raw=True,
        anatomical_accuracy_validated=False, form_classification=False))
result = dict(clips=checks, existing_examples_unchanged=True, paper_scores_unchanged=True,
    measurement_color='neutral blue; no astronaut anomaly flags',
    caveat='Confidence coverage and source alignment do not establish anatomical accuracy or clinically correct form.')
(AUDIT / 'final_data_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

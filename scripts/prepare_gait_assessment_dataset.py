"""Freeze audited gait outputs for shared scratch and MotionBERT experiments."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
from gait_geometry import filtered, measure_segment
from extract_gait_metrics import apply_visual_review
from compare_gait_groups import clip_value
from gait_geometry import NAMES
from gait_from_scratch_model import CLASSES, OUTCOMES
from gait_from_scratch_sampling import WindowIdentity, conditional_balancing_weights
from motionbert_gait_common import (ROOT, SOURCE, OUT, SEED, VERSION, METHODS, SPLITS,
    SAPIENS_SHA, WINDOW_SECONDS, STRIDE_SECONDS, INPUT_FPS, definitions, subject_splits, window_bounds, pose_adapter,
    reference_fence, classify, now, sha, write)


SPLIT_PATH = OUT / 'pilot_1s/splits.json'
DESTINATION = ROOT / 'results/gait_from_scratch_v1/dataset'


def identity(m):
    return WindowIdentity(m['subject'], m['trial_block'], m['source'],
        'sapiens2_1b' if m['method'] == 'sapiens' else 'publisher', m['split'], bool(m['subject']))


def weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    return float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)])


def adapter(xy, score, valid, t, start, width, height, lo, hi, review, pose):
    # Upstream extraction stores this audited named schema; never take first 17.
    assert NAMES == ['nose','left_eye','right_eye','left_ear','right_ear',
        'left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
        'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle',
        'left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']
    mask = valid.copy()
    mask[:lo] = False; mask[hi:] = False
    if review.get('weight_sha256') == pose.get('weight_sha256'):
        affected = review.get('affected_landmarks', [])
        if review.get('reason_code') == 'head_landmarks_obscured_by_privacy_blur':
            affected = NAMES[:5]
        for name in affected:
            if name in NAMES: mask[:, NAMES.index(name)] = False
        if len(review.get('affected_metric_ids', [])) == 17:
            mask[:] = False
    return pose_adapter(xy, score, mask, t, start, width, height)


def measured_values(segment, outcomes, review, pose, scale, quality_flags, support):
    values = np.full(len(outcomes), np.nan, np.float32)
    reasons = []
    affected = set(review.get('affected_metric_ids', [])) if review.get('weight_sha256') == pose.get('weight_sha256') else set()
    bad_scale = bool(affected) and review.get('normalization_scale_unreliable',
                       bool(affected & {f'M{i:02d}' for i in range(4, 15)}))
    for i, outcome in enumerate(outcomes):
        why = []
        metric = next(m for m in segment['metrics'] if m['metric_id'] == outcome['family'])
        if segment['view'] != outcome['view']:
            why.append('view_incompatible')
        if quality_flags:
            why.append('segment_quality_flag')
        if scale is None or scale < 80:
            why.append('projected_scale_too_small')
        if outcome['family'] in affected:
            why.append('visual_pose_localization_failure')
        if bad_scale and outcome['unit'] == 'L0':
            why.append('leg_scale_unreliable')
        if metric['measurement_status'] == 'not_applicable':
            why.append('outside_straight_walking_scope')
        if outcome.get('free_arm_only') and support not in ('none', 'no_aid_recorded_in_curated_reference'):
            why.append('arm_support_context')
        kind, key, stat = outcome['source'].split(':')
        component = metric.get('observations', {}).get(key, {}) if kind == 'observation' else segment.get('paired_geometry', {}).get(key, {})
        if component.get('valid_fraction', 0) < .8:
            why.append('component_coverage')
        value = component.get(stat)
        if value is None or not np.isfinite(value):
            why.append('no_measurement')
        if not why:
            values[i] = value
        reasons.append(sorted(set(why)))
    return values, reasons


def prepare(destination, allow_incomplete=False):
    assert not allow_incomplete, "Only completed audited inputs may be frozen"
    destination.mkdir(parents=True, exist_ok=False)
    lock = (destination / '.prepare.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (destination / 'dataset_manifest.json').exists() and not allow_incomplete:
        raise RuntimeError('Frozen dataset already exists; use a new version directory.')
    production = json.loads((SOURCE / 'production_status.json').read_text())
    audit = json.loads((SOURCE / 'output_audit.json').read_text())
    post = json.loads((SOURCE / 'postprocessing_status.json').read_text())
    handoff = json.loads((SOURCE / 'data_handoff_audit.json').read_text())
    final = json.loads((SOURCE / 'final_verification.json').read_text())
    assert handoff['status'] == 'passed_data_gate' and final['final_ui_review_complete']
    if not allow_incomplete:
        assert production['phase'] == 'pose_complete', 'Current GPU run is not complete'
        assert post['status'] == 'derived_outputs_complete', 'Derived outputs still updating'
        assert audit['native_completed'] == 219 and not audit['failures']
    code_paths = [ROOT / ('scripts/' + n) for n in ['prepare_gait_assessment_dataset.py',
        'motionbert_gait_common.py','gait_geometry.py','extract_gait_metrics.py',
        'compare_gait_groups.py','gait_from_scratch_model.py','gait_from_scratch_sampling.py']]
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_paths}
    frozen, outcomes, labels = definitions()
    mapping = json.loads(SPLIT_PATH.read_text())['subjects']
    assert mapping == subject_splits(frozen['rows'])
    assert [o['id'] for o in outcomes] == list(OUTCOMES)
    labels = sorted(labels, key=lambda label: CLASSES.index(label['id']))
    rows = [dict(r) for r in frozen['rows']]
    rows += [dict(r, same_model_subset=False, paired_publisher=True) for r in frozen['rows'] if r['same_model_subset']]
    samples, metadata, numeric, clip_inventory, file_hashes = [], [], [], [], {}
    for p in [SPLIT_PATH, SOURCE/'comparison_manifest_frozen.json', SOURCE/'pose_visual_review.json',
              SOURCE/'processing_protocol.json', SOURCE/'data_handoff_audit.json', SOURCE/'final_verification.json']:
        file_hashes[str(p.relative_to(ROOT))] = sha(p)
    for row in rows:
        cid = row['clip_id']
        base = SOURCE / ('harmonized' if row['same_model_subset'] else '') / 'clips' / cid
        r = json.loads((base / 'metrics.json').read_text())
        entry = dict(clip_id=cid, source=row['source'], subject=row['subject'],
                     split=mapping.get(row['subject'], 'exploratory'), status=r['status'], windows=0,
                     metric_path=str((base/'metrics.json').relative_to(ROOT)))
        if r['status'] != 'observations_extracted':
            entry['reason'] = 'waiting_for_pose'; clip_inventory.append(entry); continue
        method = 'sapiens' if r['pose']['method'] == 'sapiens_inference' else 'publisher'
        if method == 'sapiens':
            assert r['pose']['weight_sha256'] == SAPIENS_SHA
        paths = [base / 'metrics.json', base / 'observations.npz']
        file_hashes.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
        with np.load(base / 'observations.npz', allow_pickle=False) as a:
            t, xy, score, valid = [a[k].copy() for k in ('timestamp_s', 'xy_pixels', 'confidence', 'quality_mask')]
            source_frame = a['source_frame'].copy()
        assert len(t) == r['processed_frames'] and np.all(np.diff(t) > 0)
        context = dict(r)
        context['contains_turns_or_transitions'] = (r.get('context_review') or {}).get('contains_turns_or_transitions', False)
        review = r.get('pose_visual_review') or {}
        context['context_review'] = r.get('context_review') or {}
        clip_gates = [clip_value(r, o, frozen['quality_rules'])[1] for o in outcomes]
        if r['clip_duration_s'] < frozen['quality_rules']['minimum_clip_duration_s']:
            clip_gates = [why + ['source_clip_too_short'] for why in clip_gates]
        for seg in r['segments']:
            for start, end in window_bounds(t, seg['first_clip_frame'], seg['end_clip_frame_exclusive'], r['fps']):
                lo, hi = np.searchsorted(t, [start - 1e-7, end - 1e-7])
                if hi - lo < 20:
                    continue
                x, sampled_frames, input_ok = adapter(xy, score, valid, t, start,
                    r['width'], r['height'], lo, hi, review, r['pose'])
                local = xy[lo:hi].copy().astype(float)
                local[..., 1] *= -1
                local[~valid[lo:hi]] = np.nan
                local = filtered(local, t[lo:hi])
                scale = seg.get('fixed_L0_pixels')
                measured = measure_segment(local, t[lo:hi], context, scale)
                apply_visual_review(measured, review, r['pose'])
                y, reasons = measured_values(measured, outcomes, review, r['pose'], scale,
                                              seg['quality_flags'], row['support_context'])
                for j, why in enumerate(clip_gates):
                    if why:
                        y[j] = np.nan; reasons[j] = sorted(set(reasons[j] + why))
                if not input_ok:
                    y[:] = np.nan
                    reasons = [v + ['model_input_coverage'] for v in reasons]
                key = f'{cid}:{method}:{lo}:{hi}'
                item = dict(key=key, clip_id=cid, source=row['source'], subject=row['subject'],
                    trial_block=row['trial_block'], source_sha256=row['source_sha256'],
                    split=entry['split'], method=method, activity=row['activity_group'],
                    environment=row['environment'], view=measured['view'], support_context=row['support_context'],
                    start_time_s=start, end_time_s=end, first_clip_frame=int(lo), end_clip_frame_exclusive=int(hi),
                    original_source_first_frame=int(source_frame[lo]), original_source_end_frame_exclusive=int(source_frame[hi - 1] + 1),
                    sampled_clip_frames=sampled_frames.tolist(), segment_id=seg['segment_id'],
                    input_eligible=input_ok, measurement_reasons=reasons, value=y.tolist(),
                    input_any_valid=bool((x[..., 2] > 0).any()), source_fps=r['fps'],
                    source_time_start_s=float(source_frame[lo]/r['fps']),
                    source_time_end_s=float((source_frame[hi-1]+1)/r['fps']),
                    paired_publisher=bool(row.get('paired_publisher')),
                    metric_family_records=[{k:v for k,v in mm.items() if k not in ('value','observations')} for mm in measured['metrics']],
                    evidence_version=VERSION, metric_code_sha256=r['extraction_code_sha256'],
                    metric_path=str((base / 'metrics.json').relative_to(ROOT)), pose_sha256=r['pose']['sha256'],
                    clinical_form_label=None)
                samples.append(x); numeric.append(y); metadata.append(item); entry['windows'] += 1
        entry.update(method=method, reason=None if entry['windows'] else 'no_eligible_one_second_segment')
        clip_inventory.append(entry)
    assert len(clip_inventory) == 609 and len({c['clip_id'] for c in clip_inventory}) == 596
    if not samples:
        raise RuntimeError('No windows available')
    x = np.stack(samples); y = np.stack(numeric)
    # Fit all reference distributions only from training LJMU/LBMC Sapiens windows.
    reference = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, m in enumerate(metadata):
        if m['split'] != 'train' or m['source'] not in ('ljmu360', 'lbmc') or m['method'] != 'sapiens' or not m['input_eligible']:
            continue
        for j, value in enumerate(y[i]):
            if np.isfinite(value):
                reference[(m['activity'], m['environment'], m['view'], j)][m['subject']][m['trial_block']].append(float(value))
    balanced = {}
    for key, people in reference.items():
        balanced[key] = {person: float(np.median([np.median(values) for values in trials.values()])) for person, trials in people.items()}
    reference_fits = {}; target = np.full((len(x), len(labels)), -1, np.int8)
    for i, m in enumerate(metadata):
        m['labels'] = []
        for j, label in enumerate(labels):
            oi = label['outcome']; key = (m['activity'], m['environment'], m['view'], oi)
            units = dict(balanced.get(key, {}))
            if m['split'] == 'train' and m['source'] in ('ljmu360', 'lbmc'):
                units.pop(m['subject'], None)
            fence = reference_fence(units) if m['method'] == 'sapiens' else None
            if fence is not None:
                fence['sha256'] = hashlib.sha256(json.dumps(fence, sort_keys=True).encode()).hexdigest()
            fit_key = '|'.join(map(str, key)) + '|exclude=' + (m['subject'] if m['split'] == 'train' and m['source'] in ('ljmu360', 'lbmc') else '')
            if fence is not None:
                reference_fits[fit_key] = fence
            value = y[i, oi]
            v = classify(value, fence, label['direction'])
            target[i, j] = v
            state = 'positive' if v == 1 else 'negative' if v == 0 else 'uncertain' if fence and np.isfinite(value) else 'unavailable'
            why = list(m['measurement_reasons'][oi])
            if m['method'] == 'publisher':
                why.append('no_method_matched_reference')
            elif fence is None:
                why.append('insufficient_reference_people_or_zero_IQR')
            elif state == 'uncertain':
                why.append('reference_boundary_buffer')
            m['labels'].append(dict(id=label['id'], state=state, value=float(value),
                unit=outcomes[oi]['unit'], reference_fit=fit_key if fence else None,
                reference_sha256=fence['sha256'] if fence else None,
                region=label['region'], side='bilateral_or_midline', reasons=why,
                metric_change_direction=('decrease' if label['direction'] == 'high' else 'increase') if v == 1 else None,
                treatment_or_prosthetic_adjustment=None))
    support = []
    for j, label in enumerate(labels):
        signs = [{m['subject'] for i, m in enumerate(metadata) if m['split'] == 'train' and m['method'] == 'sapiens' and m['input_eligible'] and target[i, j] == sign} for sign in [0, 1]]
        enabled = all(len(s) >= 2 for s in signs)
        support.append(dict(**label, classification_enabled=enabled, negative_training_subjects=sorted(signs[0]), positive_training_subjects=sorted(signs[1])))
    split_id = np.array([SPLITS.index(m['split']) for m in metadata], np.int8)
    method_id = np.array([METHODS.index(m['method']) for m in metadata], np.int8)
    valid_y = np.isfinite(y)
    stats = {}
    for method in range(2):
        mask = (split_id == 0) & (method_id == method)
        center, scale, counts = [], [], []
        for j in range(y.shape[1]):
            ids = np.flatnonzero(mask & valid_y[:, j])
            vals = y[ids,j]
            weights = conditional_balancing_weights([identity(metadata[i]) for i in ids]) if len(ids) else np.array([])
            center.append(weighted_median(vals, weights) if len(vals) else 0.)
            mean = np.average(vals, weights=weights) if len(vals) else 0.
            scale.append(max(float(np.sqrt(np.average((vals-mean)**2, weights=weights))), .01) if len(vals) else 1.)
            counts.append(len(vals))
        stats[METHODS[method]] = dict(center=center, scale=scale, count=counts)
    training = split_id == 0
    testing = split_id == 2
    assert not ({m['subject'] for i, m in enumerate(metadata) if training[i]} & {m['subject'] for i, m in enumerate(metadata) if testing[i]})
    for field in ['subject', 'trial_block', 'source_sha256']:
        partitions = defaultdict(set)
        for m in metadata:
            if m[field] is not None:
                partitions[m[field]].add(m['split'])
        assert all(len(v) == 1 for v in partitions.values()), 'Split leakage: ' + field
    for p in code_paths:
        assert sha(p) == code_hashes[str(p.relative_to(ROOT))], 'Extraction code changed while building dataset'
    if not allow_incomplete:
        for path, digest in file_hashes.items():
            assert sha(ROOT / path) == digest, 'Input changed during snapshot: ' + path
    # Context/reference absence is not a failed-pose target.
    quality_scope = np.array([[not set(why) & {'view_incompatible','outside_straight_walking_scope','arm_support_context'}
        for why in m['measurement_reasons']] for m in metadata], bool)
    positive_weights, prevalence = [], []
    for j in range(len(labels)):
        ids = [i for i,m in enumerate(metadata) if m['split']=='train' and m['method']=='sapiens' and m['input_eligible'] and target[i,j]>=0]
        if ids:
            w = conditional_balancing_weights([identity(metadata[i]) for i in ids])
            p = float(np.average(target[ids,j],weights=w))
        else: p = 0.
        prevalence.append(p); positive_weights.append(float(np.clip((1-p)/max(p,1e-8),.2,10)))
    write(destination / 'training_baselines.json', dict(class_prevalence=prevalence,positive_class_weight=positive_weights,
        fitted_on='training people only; equal participant then physical trial then eligible window',
        numeric_median={method:stats[method]['center'] for method in METHODS}))
    write(destination / 'splits.json', dict(seed=SEED, subjects=mapping, reused_from=str(SPLIT_PATH.relative_to(ROOT)),
        source_sha256=sha(SPLIT_PATH), counts=dict(Counter(m['split'] for m in metadata))))
    write(destination / 'reference_fits.json', reference_fits)
    write(destination / 'normalization.json', stats)
    write(destination / 'label_protocol.json', dict(version=VERSION, outcomes=outcomes, labels=support, clinical_labels=False,
        teacher='Train-reference participant-balanced IQR fences; engineering pseudo-labels',
        window_seconds=WINDOW_SECONDS, stride_seconds=STRIDE_SECONDS, sampled_fps=INPUT_FPS, minimum_reference_participants=5,
        adapter='Named limbs; pelvis=hip midpoint; thorax=shoulder midpoint; spine=mean hips/shoulders; head=eye midpoint approximation; image-centred normalization',
        missing='Finite model placeholders with confidence zero; measured missing values stay NaN',
        quality_gate='Inherited component masks; >=0.8 coverage; L0>=80px; source clip >=2s; retained segment fraction >=0.5; regional failures and dependent normalization excluded',
        coordinate_convention='Input: image x right/y down, image-centred and scaled by half the smaller image dimension. Teacher: x right/y up; fixed segment L0.',
        head_privacy_policy='All five face landmarks masked for documented privacy-blur failures; all-invalid full-body failure windows excluded from models.',
        boundaries='Nearest native timestamps must belong to the current one-second window; no cross-window or cross-segment padding.',
        class_order=list(CLASSES),
        quality_target='Measurement available vs unavailable within applicable activity/view/support only; no reference absence used as a quality negative.'))
    write(destination / 'windows.json', metadata)
    write(destination / 'method_inventory.json', clip_inventory)
    write(destination / 'clip_inventory.json', clip_inventory[:596])
    tmp = destination / 'dataset.npz.tmp'
    with tmp.open('wb') as f:
        np.savez_compressed(f, x=x, values=y, measurement_mask=valid_y, labels=target,
                            split=split_id, method=method_id, quality_scope=quality_scope,
                            explicit_validity=x[...,2]>0,
                            input_eligible=np.array([m['input_eligible'] for m in metadata],bool),
                            input_any_valid=np.array([m['input_any_valid'] for m in metadata],bool))
    tmp.replace(destination / 'dataset.npz')
    with (destination / 'labels.jsonl').open('w') as f:
        from motionbert_gait_common import clean
        for m in metadata:
            f.write(json.dumps(clean(m), allow_nan=False) + '\n')
    summary = dict(clips=596, method_records=609, windows=len(x), window_splits=dict(Counter(m['split'] for m in metadata)),
        pose_methods=dict(Counter(m['method'] for m in metadata)),
        primary_labels_enabled=sum(s['classification_enabled'] for s in support),
        numeric_targets=int(valid_y.sum()), label_states=dict(Counter(l['state'] for m in metadata for l in m['labels'])),
        clip_reasons=dict(Counter(m.get('reason') or 'windows_created' for m in clip_inventory)))
    write(destination / 'label_coverage.json', summary)
    write(destination / 'dataset_manifest.json', dict(created_at=now(), status='partial_pilot' if allow_incomplete else 'frozen',
        code_hashes=code_hashes, input_hashes=file_hashes, dataset_sha256=sha(destination / 'dataset.npz'),
        artifact_hashes={name:sha(destination/name) for name in ['splits.json','reference_fits.json','normalization.json',
            'label_protocol.json','windows.json','clip_inventory.json','method_inventory.json','label_coverage.json','training_baselines.json','labels.jsonl']},
        production_snapshot=production, output_audit_snapshot={k:audit[k] for k in ['updated_at','native_completed','failures']},
        summary=summary, no_clinical_ground_truth=True))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=DESTINATION)
    p.add_argument('--allow-incomplete', action='store_true')
    args = p.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        prepare(args.output, args.allow_incomplete)

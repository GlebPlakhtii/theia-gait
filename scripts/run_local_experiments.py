"""Run fresh, explicitly exploratory experiments using already-local inputs.

Clinical experiment uses archived FEATURES and labels, never saved predictions.
ARED experiment uses newly extracted RGB poses; it has no decision ground truth.
These are separate diagnostics, not a clinical-to-ARED transfer success claim.
"""
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.signal import find_peaks, medfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/local_execution_20261114'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Protocol already exists: do not overwrite a recorded experiment')
    a = ROOT / 'archive/colleague/patient_transfer_v2_results.json'
    b = ROOT / 'archive/colleague/camera_trunk_v2_results.json'
    old, corrected = json.loads(a.read_text()), json.loads(b.read_text())
    datasets = {
        'CTK_arm_elevation': old['targets']['CTK_Error1']['records'],
        'ELK_pelvis_relative': old['targets']['ELK_Error2']['records'],
        'ELK_camera_relative': corrected['result']['records'],
    }
    # Deliberately remove historical predictions, thresholds, metrics and folds.
    data = {k: [{f: r[f] for f in ('id', 'subject', 'cohort', 'y', 'features', 'detail_gap')}
                for r in rows] for k, rows in datasets.items()}
    write(OUT / 'clinical_features.json', data)
    selection = json.loads((ROOT / 'data/poses/sapiens2_5b_full/analysis_selection.json').read_text())
    poses = {'ared_670_779': ROOT / 'data/poses/ared_continuous_development/poses.csv'}
    for r in selection['clips']:
        if r['included_in_provisional_export'] and r['clip_id'].startswith('mendeley'):
            poses[r['clip_id']] = ROOT / 'data/poses/sapiens2_5b_full' / r['clip_id'] / 'poses.csv'
    protocol = {
        'recorded_at': now(), 'status': 'exploratory_protocol_recorded_before_new_fits',
        'authorization': 'User explicitly instructed RUN IT using available data; no additional source acquisition.',
        'scope': 'Fresh clinical feature experiment plus separate in-mission RGB-pose cycle extraction. Neither establishes rehabilitation benefit on an in-mission decision.',
        'prior_exposure': 'All six KERAAL patients and ARED development recording previously explored. This is not an independent confirmatory study or exact reproduction of missing original code.',
        'archive_inputs': {str(p.relative_to(ROOT)): sha(p) for p in (a, b)},
        'clinical_feature_sha256': sha(OUT / 'clinical_features.json'),
        'clinical': {
            'labels': 'Existing KERAAL Error1 (CTK) / Error2 (ELK) annotation-derived record labels retained from archive. Zero means this error code absent, not globally correct.',
            'arms': ['healthy_only', 'healthy_plus_clinical'],
            'split': 'Leave each of R1-R6 out in full; baseline uses only healthy S records; augmented arm adds all other R records. No outer test rows in scaling, fitting or weighting.',
            'model': 'Weighted binary logistic regression with L2 penalty, free intercept; weighted training-only mean/SD scaling.',
            'weights': 'Each training participant initially equal total weight; rescale class totals to 0.5 each; normalize total to 1.',
            'primary_lambda': 0.1, 'sensitivity_lambdas': [0.01, 1.0],
            'threshold': 0.5, 'parameter_selection': 'None; all three predeclared penalties reported. No target-based tuning.',
            'primary_metric': 'Mean per-patient accuracy; also report pooled confusion counts and balanced accuracy.',
            'uncertainty': '10,000 paired patient-cluster bootstrap resamples of accuracy difference; descriptive with only six clusters.',
            'bootstrap_seed': 20261114,
            'limitations': ['Archived numerical features, not new RGB pose validation.',
                           'Clinical augmentation increases data amount and participant diversity; no equal-size causal control.',
                           'ELK healthy training contains only four records from two participants.',
                           'Geometry ablation is already-exposed, post-hoc development evidence.',
                           'These exercises are arm elevation and lateral bending, not squats.'],
        },
        'squat_cycles': {
            'inputs': {k: {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for k, p in poses.items()},
            'angle': 'Mean of left/right projected hip-knee-ankle angles; require all six heatmap scores >=0.3. Not calibrated visibility or anatomical flexion.',
            'smoothing': 'Centered median filter, approximately 0.2 s; offline processing. No gap interpolation.',
            'primary_prominence_deg': 15.0, 'sensitivity_prominences_deg': [10.0, 20.0],
            'candidate_rule': 'Trough with requested prominence and >=1s trough spacing, bounded by nearest local angle maxima (>=5deg prominence; >=0.6s spacing); both excursions >=requested prominence; start-to-end 1-8s; end no more than 10deg below start; no missing-data crossing; nonoverlapping candidates.',
            'endpoint': 'Following angle maximum is a candidate completion timestamp; no independently labeled event metrics.',
            'no_training': 'Same fixed detector on ARED and all five eligible gym clips. Gym health status undocumented. No clinical squat arm invented.',
        },
    }
    write(OUT / 'protocol.json', protocol)
    print('Protocol recorded:', OUT / 'protocol.json', flush=True)


def fit(rows, penalty):
    x = np.asarray([r['features'] for r in rows], float)
    y = np.asarray([r['y'] for r in rows], float)
    assert np.isfinite(x).all() and set(y) == {0., 1.}
    counts = Counter(r['subject'] for r in rows)
    weights = np.array([1 / counts[r['subject']] for r in rows])
    for cls in (0, 1):
        weights[y == cls] *= .5 / weights[y == cls].sum()
    mean = np.average(x, axis=0, weights=weights)
    sd = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    sd[sd < 1e-8] = 1.
    z = np.column_stack([np.ones(len(x)), (x - mean) / sd])

    def objective(beta):
        logits = z @ beta
        loss = np.sum(weights * (np.logaddexp(0, logits) - y * logits))
        loss += penalty * .5 * np.sum(beta[1:] ** 2)
        grad = z.T @ (weights * (expit(logits) - y))
        grad[1:] += penalty * beta[1:]
        return loss, grad

    result = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-9})
    if not result.success or np.max(np.abs(result.jac)) > 1e-5:
        raise RuntimeError(f'Fit failed: {result.message}; gradient {result.jac}')
    return {'mean': mean.tolist(), 'sd': sd.tolist(), 'coefficients': result.x.tolist(),
            'loss': float(result.fun), 'iterations': int(result.nit),
            'max_abs_gradient': float(np.max(np.abs(result.jac))),
            'training_ids': [r['id'] for r in rows], 'training_subjects': sorted(counts)}


def metrics(rows):
    c = {k: 0 for k in ('tp', 'tn', 'fp', 'fn')}
    by_subject = {}
    for r in rows:
        y, p = bool(r['y']), bool(r['prediction'])
        c['tp' if y and p else 'fn' if y else 'fp' if p else 'tn'] += 1
        by_subject.setdefault(r['subject'], []).append(int(y == p))
    sens = c['tp'] / (c['tp'] + c['fn']) if c['tp'] + c['fn'] else None
    spec = c['tn'] / (c['tn'] + c['fp']) if c['tn'] + c['fp'] else None
    acc = {k: float(np.mean(v)) for k, v in by_subject.items()}
    return {**c, 'n': len(rows), 'sensitivity': sens, 'specificity': spec,
            'pooled_accuracy': (c['tp'] + c['tn']) / len(rows),
            'pooled_balanced_accuracy': (sens + spec) / 2 if sens is not None and spec is not None else None,
            'per_patient_accuracy': acc, 'macro_patient_accuracy': float(np.mean(list(acc.values())))}


def clinical(protocol):
    p = OUT / 'clinical_features.json'
    assert sha(p) == protocol['clinical_feature_sha256']
    datasets = json.loads(p.read_text())
    a, b = datasets['ELK_pelvis_relative'], datasets['ELK_camera_relative']
    assert [(r['id'], r['subject'], r['y']) for r in a] == [(r['id'], r['subject'], r['y']) for r in b]
    all_predictions, fits, summaries = [], [], []
    for name, records in datasets.items():
        assert len({r['id'] for r in records}) == len(records)
        assert all(set(r) == {'id', 'subject', 'cohort', 'y', 'features', 'detail_gap'} for r in records)
        subjects = sorted({r['subject'] for r in records if r['cohort'] == 'patient'})
        for penalty in [0.1, 0.01, 1.0]:
            methods = {}
            for arm in ('healthy_only', 'healthy_plus_clinical'):
                predictions = []
                for subject in subjects:
                    train = [r for r in records if r['subject'] != subject and
                             (r['cohort'] == 'healthy' or arm == 'healthy_plus_clinical')]
                    test = [r for r in records if r['subject'] == subject]
                    assert {r['subject'] for r in train}.isdisjoint({r['subject'] for r in test})
                    assert {r['id'] for r in train}.isdisjoint({r['id'] for r in test})
                    model = fit(train, penalty)
                    fits.append({'dataset': name, 'lambda': penalty, 'arm': arm,
                                 'held_out_patient': subject, 'test_ids': [r['id'] for r in test], **model})
                    x = np.asarray([r['features'] for r in test])
                    z = np.column_stack([np.ones(len(x)), (x - model['mean']) / model['sd']])
                    scores = expit(z @ model['coefficients'])
                    for r, score in zip(test, scores):
                        predictions.append({'dataset': name, 'lambda': penalty, 'arm': arm,
                                            'id': r['id'], 'subject': subject, 'y': r['y'],
                                            'detail_gap': r['detail_gap'], 'score': float(score),
                                            'prediction': int(score >= .5)})
                assert len(predictions) == sum(r['cohort'] == 'patient' for r in records)
                methods[arm] = metrics(predictions)
                methods[arm]['exclude_detail_gap_sensitivity'] = metrics([r for r in predictions if not r['detail_gap']])
                all_predictions.extend(predictions)
            delta = np.array([methods['healthy_plus_clinical']['per_patient_accuracy'][s] -
                              methods['healthy_only']['per_patient_accuracy'][s] for s in subjects])
            rng = np.random.default_rng(protocol['clinical']['bootstrap_seed'])
            boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
            summary = {'dataset': name, 'lambda': penalty, 'primary': penalty == .1,
                       'patients': len(subjects), 'healthy_records': sum(r['cohort'] == 'healthy' for r in records),
                       'methods': methods, 'augmentation_minus_baseline_macro_accuracy': float(delta.mean()),
                       'paired_cluster_bootstrap_95_percentile': np.quantile(boot, [.025, .975]).tolist()}
            summaries.append(summary)
            print(name, 'lambda', penalty, {k: (v['fp'], v['fn'], round(v['macro_patient_accuracy'], 4)) for k, v in methods.items()}, flush=True)
    write(OUT / 'clinical_models.json', fits)
    write(OUT / 'clinical_predictions.json', all_predictions)
    write(OUT / 'clinical_results.json', {'new_model_fits': len(fits), 'summaries': summaries})
    # Every changed case is retained, including harm; no selection of favorable examples.
    cases = []
    for name in datasets:
        rows = [r for r in all_predictions if r['dataset'] == name and r['lambda'] == .1]
        base = {r['id']: r for r in rows if r['arm'] == 'healthy_only'}
        aug = {r['id']: r for r in rows if r['arm'] == 'healthy_plus_clinical'}
        for rid, r in base.items():
            s = aug[rid]
            effect = 'unchanged' if r['prediction'] == s['prediction'] else 'helped' if s['prediction'] == s['y'] else 'harmed'
            cases.append({'dataset': name, 'id': rid, 'patient': r['subject'], 'label': r['y'],
                          'healthy_only': r['prediction'], 'healthy_plus_clinical': s['prediction'], 'effect': effect})
    write(OUT / 'clinical_paired_cases.json', cases)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, s in zip(axes, [s for s in summaries if s['primary']]):
        mm = s['methods']; labels = ['Healthy only', '+ Clinical']
        ax.bar(np.arange(2)-.17, [m['fp'] for m in mm.values()], .34, label='False alerts', color='#d88728')
        ax.bar(np.arange(2)+.17, [m['fn'] for m in mm.values()], .34, label='Missed errors', color='#ba4c58')
        ax.set(xticks=np.arange(2), xticklabels=labels, title=s['dataset'].replace('_',' '), ylabel='Records', ylim=(0, max(v[k] for v in mm.values() for k in ('fp','fn'))+3))
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Fresh held-patient-out fits · exploratory archived features · fixed λ=0.1, threshold=0.5')
    fig.savefig(OUT / 'clinical_comparison.png', dpi=160); plt.close(fig)
    return summaries


def load_signal(path):
    frames = {}
    joints = [f'{s}_{j}' for s in ('left', 'right') for j in ('hip', 'knee', 'ankle')]
    for r in csv.DictReader(open(path)):
        if r['joint_id'] not in joints:
            continue
        key = int(r['source_frame_index'])
        frame = frames.setdefault(key, {'time': float(r['timestamp_s']), 'joints': {}})
        assert r['joint_id'] not in frame['joints'], 'Duplicate native frame/joint'
        frame['joints'][r['joint_id']] = [float(r['x']) if r['x'] else np.nan,
                                        float(r['y']) if r['y'] else np.nan, float(r['confidence'] or 0)]
    indices = sorted(frames)
    assert all(b-a == 1 for a,b in zip(indices, indices[1:])), 'Missing native frames'
    times = np.array([frames[i]['time'] for i in indices])
    p = np.array([[frames[i]['joints'][j] for j in joints] for i in indices])
    angles = []
    for offset in (0, 3):
        a,b,c = p[:, offset:offset+3, :2].transpose(1,0,2)
        u,v = a-b,c-b
        denom = np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        cos = np.divide((u*v).sum(1),denom,out=np.full(len(p),np.nan),where=denom>1e-8)
        angles.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
    angle = np.mean(angles,axis=0)
    valid = np.isfinite(angle) & (p[:,:,2].min(1) >= .3)
    return np.array(indices),times,angle,valid


def candidates(t, raw, valid, prominence):
    dt = float(np.median(np.diff(t)))
    smooth = np.full(len(raw),np.nan); events = []
    edges = np.diff(np.r_[False,valid,False].astype(int))
    for start,end in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
        kernel = max(3, int(round(.2/dt)) | 1)
        if end-start < max(kernel, int(1/dt)):
            continue
        pad = kernel//2
        y = medfilt(np.pad(raw[start:end],pad,mode='edge'),kernel)[pad:-pad]
        smooth[start:end] = y
        troughs,_ = find_peaks(-y,prominence=prominence,distance=max(1,round(1/dt)))
        tops,_ = find_peaks(y,prominence=5,distance=max(1,round(.6/dt)))
        last_end = -1
        for bottom in troughs:
            before,after = tops[tops<bottom],tops[tops>bottom]
            if not len(before) or not len(after): continue
            a,b = int(before[-1]),int(after[0])
            duration = t[start+b]-t[start+a]
            if a < last_end or not 1 <= duration <= 8: continue
            if min(y[a]-y[bottom],y[b]-y[bottom]) < prominence or y[b] < y[a]-10: continue
            events.append({'start_s':float(t[start+a]),'bottom_s':float(t[start+bottom]),
                           'completion_s':float(t[start+b]),'duration_s':float(duration),
                           'projected_excursion_deg':float(min(y[a]-y[bottom],y[b]-y[bottom])),
                           'start_index':int(start+a),'bottom_index':int(start+bottom),'end_index':int(start+b)})
            last_end = b
    return smooth,events


def squat(protocol):
    outputs = []
    for clip, info in protocol['squat_cycles']['inputs'].items():
        path = ROOT / info['path']; assert sha(path) == info['sha256']
        indices,t,raw,valid = load_signal(path)
        dt = float(np.median(np.diff(t)))
        result = {'clip_id':clip,'native_frames':len(t),'source_start_s':float(t[0]),
                  'duration_s':float(t[-1]-t[0]+dt),'score_gate_coverage':float(valid.mean()),
                  'reference_labels_available':False,'parameter_results':[]}
        for prominence in [15.,10.,20.]:
            smooth,events = candidates(t,raw,valid,prominence)
            for e in events:
                e['completion_source_frame'] = int(indices[e['end_index']])
            result['parameter_results'].append({'prominence_deg':prominence,'primary':prominence==15,
                                                'candidate_count':len(events),'candidates':events})
        outputs.append(result)
        smooth,events = candidates(t,raw,valid,15.)
        with open(OUT / f'{clip}_signals.csv','w') as f:
            writer=csv.writer(f);writer.writerow(['source_frame','source_time_s','raw_projected_angle_deg','median_filtered_deg','score_gate_passed'])
            for frame,time,angle,filtered,ok in zip(indices,t,raw,smooth,valid):
                writer.writerow([frame,time,angle,filtered if np.isfinite(filtered) else '',int(ok)])
        fig,ax=plt.subplots(figsize=(12,3.7),layout='constrained')
        ax.plot(t,raw,color='#8e9da7',alpha=.6,lw=.6,label='Raw projected angle')
        ax.plot(t,smooth,color='#116c82',lw=1,label='Median-filtered angle')
        for i,e in enumerate(events):
            ax.axvspan(e['start_s'],e['completion_s'],color='#299567',alpha=.1)
            ax.axvline(e['completion_s'],color='#299567',alpha=.7,lw=.8)
            ax.text(e['completion_s'],183,str(i+1),fontsize=7,ha='center')
        ax.set(xlabel='Original source time (s)',ylabel='Projected knee angle (degrees)',ylim=(0,190),
               title=f'{clip}: {len(events)} candidate cycles, unvalidated')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(OUT/f'{clip}_cycles.png',dpi=160);plt.close(fig)
        print(clip,'frames',len(t),'candidate counts (15/10/20deg)',[r['candidate_count'] for r in result['parameter_results']],flush=True)
    write(OUT/'squat_candidates.json',outputs)
    return outputs


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    if (OUT/'complete.json').exists():
        raise RuntimeError('Completed output exists; preserve it')
    started=now()
    results=clinical(protocol)
    cycles=squat(protocol)
    write(OUT/'complete.json',{'started_at':started,'completed_at':now(),
                             'script_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
                             'python':platform.python_version(),'numpy':np.__version__,
                             'clinical_new_fits':108,'clinical_evaluation_configurations':18,
                             'squat_detector_configurations':3,'squat_native_frames':sum(r['native_frames'] for r in cycles),
                             'clinical_to_mission_decision_experiments':0,
                             'artifacts':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}})
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,default=OUT,help='New output directory for a repeat; completed runs are never overwritten')
    args=parser.parse_args()
    OUT=args.output.resolve()
    prepare() if args.action=='prepare' else run()

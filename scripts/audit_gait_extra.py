"""Mechanical artifact and scientific-boundary audit for the additional report."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from gait_extra_data import OUT,BASE
from motionbert_gait_common import ROOT,sha,write,now


def audit():
    read=lambda p:json.loads(Path(p).read_text())
    protocol=read(OUT/'protocol.json');training=read(OUT/'training_summary.json');evaluation=read(OUT/'evaluation.json');demo=read(OUT/'report_data.json')
    assert training['training_phase_complete'] and training['all_requested_runs_trained']
    assert len(training['runs'])==len(protocol['jobs'])==len(evaluation['runs'])==30
    assert {r['id'] for r in evaluation['runs']}=={r['id'] for r in protocol['jobs']}
    assert training['test_evaluations_during_training']==0 and evaluation['training_complete_before_test_evaluation']
    ep=read(OUT/'evaluation_protocol.json');assert datetime.fromisoformat(ep['created_at'])>=datetime.fromisoformat(training['updated_at'])
    assert ep['training_summary_sha256']==sha(OUT/'training_summary.json')
    assert ep['code_sha256']==sha(ROOT/'scripts/evaluate_gait_extra.py')
    implementation=read(OUT/'implementation_manifest.json')
    for name,digest in implementation['code_sha256'].items():assert sha(ROOT/'scripts'/name)==digest
    assert implementation['protocol_sha256']==sha(OUT/'protocol.json')
    assert protocol['plan_sha256']==sha(OUT/'PLAN.txt')
    assert protocol['base_dataset_sha256']==sha(BASE/'dataset.npz')
    splits={name:np.load(OUT/'splits'/name/'targets.npz') for name in protocol['participant_maps']}
    base=np.load(BASE/'dataset.npz');checks=[]
    for r in evaluation['runs']:
        job=next(j for j in protocol['jobs'] if j['id']==r['id']);folder=OUT/'runs'/r['id'];result=read(folder/'training/training_result.json')
        assert r['status']==result['status']=='trained_with_validation' and result['test_data_consumed'] is False
        manifest=read(folder/'run_manifest.json');assert manifest['no_test_result_access'] and manifest['job']==job
        assert manifest['split_manifest_sha256']==sha(OUT/'splits'/r['split']/'manifest.json')
        assert r['checkpoint_sha256']==sha(folder/'training/best.pt')
        if r['architecture']=='motionbert':assert result['encoder_updates']>0 and result['nonzero_encoder_gradient_steps']>0 and result['changed_best_encoder_tensors']>0
        else:assert result['nonzero_gradient_steps']>0 and result['parameter_change_l2']>0
        prediction=np.load(folder/'test_predictions.npz');ids=prediction['indices'];split=splits[r['split']]['split']
        assert np.all(split[ids]==2)
        assert all(np.isfinite(prediction[k]).all() for k in ['probabilities','values','evaluability'])
        assert len(ids)==r['inference_windows']
        if r['condition']=='primary_only':assert r['publisher_method_shift'] is None and not r['publisher_scored']
        for label in r['primary']['labels']:
            if label['average_precision'] is not None:assert label['positive']>0 and label['negative']>0 and label['enabled']
        for o in r['balanced']['outcomes']:
            assert o['participant_count']==len(o['people'])
        checks.append(dict(id=r['id'],best_sha256=r['checkpoint_sha256'],prediction_windows=len(ids),status='passed'))
    mappings=protocol['participant_maps'];test_counts=Counter(p for key,m in mappings.items() if key!='original' for p,s in m.items() if s=='test')
    assert len(test_counts)==15 and set(test_counts.values())=={1}
    root=ROOT/'deliverables/Theia_Squat_Form_Lab_2026-11-14'
    for e in demo['examples']:
        assert mappings[e['split']][e['subject']]=='test'
        assert (OUT/e['predictions']).is_file()
        v=demo['videos'][e['clip_id']];assert (root/'gait_metrics/videos'/v['video']).is_file()
        assert (root/'gait_metrics'/e['pose_prefix']/'clips'/e['clip_id']/'viewer.json').is_file()
        events=read(OUT/e['predictions'])['windows']
        for w in events:
            for event in w['events']:
                if event['prediction_state'] in ('positive','negative'):
                    assert event['quality_gate_passed'] and event['predicted_evaluability_score']>=.5 and event['predicted_value']>=0 and event['reference']
                assert event['clinical_form_label'] is None and event['treatment_or_prosthetic_adjustment'] is None
    for clip in demo['astronaut_clips']:
        assert (root/clip['video']).is_file() and (root/clip['track']).is_file()
        assert clip['assessment']['clinicalVerdict'] is None
        assert len(clip['raw_source_frames'])==len(clip['raw_source_times_s'])==clip['frames']
        assert np.all(np.diff(clip['raw_source_frames'])==1) and np.all(np.diff(clip['raw_source_times_s'])>0)
    p21=next(c for c in demo['astronaut_clips'] if c['clip_id']=='P21_ARED_01_002')
    assert p21['raw_source_frames'][0]==p21['source_first_frame']+30
    for f in demo['figures']:
        path=(root/f['src'].lstrip('/')) if f['src'].startswith('/') else OUT/f['src']
        assert path.is_file()
    cp=demo['clip_prediction_points']
    assert len(cp)==738 and len({p['clip_id'] for p in cp})==75
    assert len({(p['architecture'],p['fold'],p['clip_id'],p['outcome']) for p in cp})==len(cp)
    for p in cp:
        assert mappings[p['fold']][p['subject']]=='test' and p['fold']!='original'
        assert np.isfinite([p['observed'],p['predicted']]).all() and p['windows']>0
    for r in demo['cohort_comparisons']:
        for group,g in r['groups'].items():
            clips={c for u in g['units'] for c in u['clips']}
            rows=[c for c in demo['cohort_clip_values'] if c['mode']==r['mode'] and c['outcome']==r['outcome']['id'] and c['cohort']==group and all(c[k]==v for k,v in r['stratum'].items()) and c['clip_id'] in clips]
            assert len({c['clip_id'] for c in rows})==g['clip_count']
    anatomy=read(OUT/'anatomy/anatomy.json');am=read(OUT/'anatomy/manifest.json')
    assert anatomy['muscle_count']==92 and len(anatomy['meshes'])==19 and len(anatomy['frames'])==46
    assert am['subject_fitted'] is False and am['activation_estimated'] is False
    for frame in anatomy['frames']:
        assert len(frame['muscles'])==92
        for transform in frame['transforms'].values():
            rotation=np.array(transform[:9]).reshape(3,3);assert np.allclose(rotation@rotation.T,np.eye(3),atol=1e-6)
        for muscle in frame['muscles']:assert len(muscle['points'])>=2 and np.isfinite(muscle['points']).all() and muscle['length']>0
    for name,digest in am['files'].items():assert sha(OUT/'anatomy'/name)==digest
    assert demo['clinical_validation'] is False
    write(OUT/'artifact_audit.json',dict(status='passed',created_at=now(),run_checks=checks,known_test_people_covered_once=15,
        actual_finetuning_verified=True,all_30_runs_reported=True,all_example_sources_held_out_in_selected_model=True,
        clinical_verdicts_absent=True,quality_gates_preserved=True,previous_dataset_hash_unchanged=True,
        generic_opensim_anatomy_verified=True,clip_and_person_plot_data_verified=True,clip_prediction_points=len(cp),supported_numeric_example_clips=75,figures=len(demo['figures']),example_model_clip_entries=len(demo['examples']),
        unique_example_clips=len({e['clip_id'] for e in demo['examples']}),astronaut_clips=len(demo['astronaut_clips']),
        astronaut_new_pose_inference=False,browser_verification='separate browser_verification.json',
        deadline_utc=protocol['delivery_deadline_utc'],finished_before_deadline=datetime.fromisoformat(now())<=datetime.fromisoformat(protocol['delivery_deadline_utc'])))
    lines=['THEIA — ADDITIONAL EXPERIMENTS AND VIDEO DEMO','',
        'Open http://localhost:8006/gait_experiments/ with the existing local server.',
        '30 new runs: 10 seed baselines, 8 participant-fold runs, 12 paired ablations.',
        'Frozen 596-clip / 6152-window dataset; no new extraction or labels.',
        'Four-fold test rotation covers 15 known people. One LBMC person stays in train.',
        'Automatic-teacher agreement only. Reused internal tests; no clinical validation.','']
    for s in demo['seed_summary']:
        lines.append(f"{s['architecture']}: five-seed mean AP {s['AP']['mean']:.6f}, range {s['AP']['min']:.6f}–{s['AP']['max']:.6f}; normalized MAE {s['normalized_MAE']['mean']:.6f}.")
    lines+=['',f"{len(demo['figures'])} figures; {len({e['clip_id'] for e in demo['examples']})} unique held-out gait example clips; {len(demo['astronaut_clips'])} existing astronaut/ARED clips.",
        'Plots overlay small clip dots with large person-summary dots; 75 clips support numeric results in the rotated folds.',
        'Interactive generic Gait2392 anatomy: 19 bone meshes and 92 muscle paths, 46 prescribed poses. No subject-specific muscle activation or forces.',
        'Astronaut results reuse Sapiens2-5B, 3D lifting and OpenSim fits from the prior squat lab.',
        'Samantha Cristoforetti source is an unchanged remote video, with no new pose inference.',
        'Matthias Maurer 360-degree timelapse is source context only, excluded from timing.',
        '', 'Files: PLAN.txt, protocol.json, splits/*, training_summary.json, evaluation.json,',
        'runs/*/training/{best.pt,last.pt,training_history.json,training_result.json},',
        'runs/*/test_predictions.npz, report_data.json, figures/*.{png,svg}, artifact_audit.json.',
        '', 'The report depends on existing local video/pose assets through the port-8006 server.',
        'It is not a portable one-file report. No data were uploaded or published.']
    (OUT/'RESULTS_GUIDE.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='passed',runs=len(checks),figures=len(demo['figures']))))


if __name__=='__main__':audit()

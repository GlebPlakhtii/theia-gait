"""Held-out evaluation once, fixed baselines and traceable full-corpus inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from gait_assessment_data import load_dataset
from gait_from_scratch_model import GaitTCN
from gait_from_scratch_evaluation import binary_metrics, numeric_metrics, merge_review_intervals
from gait_from_scratch_benchmark import benchmark_assessment
from motionbert_gait_common import ROOT, METHODS, SPLITS, now, sha, write, classify


def support(data, ids):
    m=[data['windows'][int(i)] for i in ids]
    return dict(windows=len(m),participants=sorted({r['subject'] for r in m if r['subject']}),
        physical_trials=len({r['trial_block'] for r in m}),clips=len({r['clip_id'] for r in m}),
        unresolved_identity_windows=sum(r['subject'] is None for r in m))


def metrics(data, ids, probabilities, values, evaluability):
    ids=np.asarray(ids,dtype=int);outcomes=[];labels=[];normalized=[];aps=[];baseline_aps=[]
    center=np.asarray([data['normalization'][METHODS[m]]['center'] for m in data['method'][ids]]).reshape(-1,10)
    scale=np.asarray([data['normalization'][METHODS[m]]['scale'] for m in data['method'][ids]]).reshape(-1,10)
    for j,o in enumerate(data['protocol']['outcomes']):
        mask=data['regression_mask'][ids,j]&data['input_any_valid'][ids]
        raw=numeric_metrics(data['values'][ids,j],values[ids,j],mask)
        baseline=numeric_metrics(data['values'][ids,j],center[:,j],mask)
        normalized_mae=float(np.mean(abs(values[ids,j][mask]-data['values'][ids,j][mask])/scale[:,j][mask])) if mask.any() else None
        if normalized_mae is not None:normalized.append(normalized_mae)
        gated=mask&(evaluability[ids,j]>=.5)&(values[ids,j]>=0)
        quality=binary_metrics(data['measurement_mask'][ids,j].astype(int),evaluability[ids,j],
            data['quality_scope'][ids,j]&data['input_any_valid'][ids])
        outcomes.append(dict(**o,**raw,n=int(mask.sum()),normalized_mae=normalized_mae,
            support=support(data,ids[mask]),baseline_median=baseline,
            quality_gated=numeric_metrics(data['values'][ids,j],values[ids,j],gated),
            quality_gated_support=support(data,ids[gated]),evaluability=quality))
    for j,label in enumerate(data['protocol']['labels']):
        mask=data['label_mask'][ids,j]&data['input_any_valid'][ids]
        raw=binary_metrics(data['labels'][ids,j],probabilities[ids,j],mask)
        baseline=binary_metrics(data['labels'][ids,j],np.full(len(ids),data['baselines']['class_prevalence'][j]),mask)
        if raw['average_precision'] is not None:aps.append(raw['average_precision']);baseline_aps.append(baseline['average_precision'])
        oi=label['outcome'];gated=mask&(evaluability[ids,oi]>=.5)&(values[ids,oi]>=0)&data['input_eligible'][ids]
        labels.append(dict(id=label['id'],enabled=label['classification_enabled'],**raw,**raw['confusion'],
            n=raw['available_windows'],positive=raw['positive_windows'],negative=raw['negative_windows'],
            support=support(data,ids[mask]),baseline_prevalence=baseline,
            quality_gated=binary_metrics(data['labels'][ids,j],probabilities[ids,j],gated),
            quality_gated_support=support(data,ids[gated])))
    return dict(**support(data,ids),outcomes=outcomes,labels=labels,
        macro_average_precision=float(np.mean(aps)) if aps else None,
        baseline_macro_average_precision=float(np.mean(baseline_aps)) if baseline_aps else None,
        mean_normalized_mae=float(np.mean(normalized)) if normalized else None,
        interpretation='Agreement with automatic engineering labels; correlated windows are not independent people; no clinical validation.')


def export_predictions(data, probabilities, values, evaluability, checkpoint_sha):
    output=[]
    for i,m in enumerate(data['windows']):
        events=[]
        for j,label in enumerate(data['protocol']['labels']):
            oi=label['outcome'];teacher=m['labels'][j];fence=data['reference_fits'].get(teacher['reference_fit'])
            measurement_ok=bool(data['regression_mask'][i,oi]);enabled=label['classification_enabled']
            q=float(evaluability[i,oi]) if np.isfinite(evaluability[i,oi]) else None
            applicable=bool(m['method']=='sapiens' and fence and measurement_ok and m['input_eligible'])
            value=float(values[i,oi]) if np.isfinite(values[i,oi]) and data['regression_mask'][i,oi] else None
            score=float(probabilities[i,j]) if enabled and applicable else None
            regression_flag=classify(value if value is not None else np.nan,fence,label['direction'])
            if not applicable or q is None or q<.5 or value is None or value<0:state='unavailable'
            elif enabled:state='positive' if score>=.5 else 'negative'
            else:state='positive' if regression_flag==1 else 'negative' if regression_flag==0 else 'uncertain'
            events.append(dict(target_id=label['id'],region=label['region'],teacher_state=teacher['state'],
                prediction_state=state,prediction_basis='trained_classifier' if enabled else 'predicted_measurement_against_reference',
                classification_head_enabled=enabled,uncalibrated_model_score=score,
                predicted_value=value,observed_value=teacher['value'],unit=teacher['unit'],
                reference={k:fence[k] for k in ['lower','upper','buffer','participant_count','sha256']} if fence else None,
                predicted_evaluability_score=q,quality_gate_passed=applicable,
                reasons=teacher['reasons'],
                disagreement=state!=teacher['state'] if state in ('positive','negative') and teacher['state'] in ('positive','negative') else None,
                metric_change_direction=('decrease' if label['direction']=='high' else 'increase') if state=='positive' else None,
                clinical_form_label=None,clinical_cause=None,treatment_or_prosthetic_adjustment=None))
        fields=['key','clip_id','source','subject','split','method','activity','environment','view','support_context',
            'start_time_s','end_time_s','first_clip_frame','end_clip_frame_exclusive','metric_path',
            'original_source_first_frame','original_source_end_frame_exclusive','source_sha256',
            'source_time_start_s','source_time_end_s','trial_block','input_eligible']
        output.append(dict(**{k:m[k] for k in fields},events=events))
    return dict(created_at=now(),checkpoint_sha256=checkpoint_sha,windows=output,clinical_validation=False,
        interpretation='Retrospective one-second review windows, not forecasts or diagnoses.')


def interval_summary(windows):
    all_intervals=[];positive=[];class_positive=[]
    for m in windows:
        interval=dict(sequence_id=m['source_sha256']+':'+m['method'],
            start_time_s=m['source_time_start_s'],end_time_s=m['source_time_end_s'])
        all_intervals.append(interval)
        if any(e['prediction_state']=='positive' for e in m['events']):positive.append(interval)
        if any(e['prediction_state']=='positive' and e['classification_head_enabled'] for e in m['events']):class_positive.append(interval)
    total=merge_review_intervals(all_intervals);review=merge_review_intervals(positive)
    classified=merge_review_intervals(class_positive)
    review.update(observed_duration_s=total['review_duration_s'],classifier_only=classified,
        review_events_per_minute=60*review['review_event_count']/total['review_duration_s'] if total['review_duration_s'] else None,
        window_count=len(windows),windows_with_no_available_target=sum(all(e['prediction_state']=='unavailable' for e in m['events']) for m in windows),
        note='Merged original-source intervals within each pose method; camera recordings remain distinct timebases.')
    return review


def evaluate(folder):
    folder=Path(folder)
    if (folder/'evaluation/results.json').exists():raise RuntimeError('Held-out evaluation already recorded; do not tune or rerun')
    torch.set_num_threads(2);data=load_dataset(folder/'dataset')
    checkpoint_path=folder/'training/best.pt';checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert checkpoint['configuration']['input_provenance']['dataset_sha256']==data['manifest']['dataset_sha256']
    result=json.loads((folder/'training/training_result.json').read_text())
    assert result['status']=='trained_with_validation' and result['nonzero_gradient_steps']>0 and result['parameter_change_l2']>0
    model=GaitTCN().cuda();model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
    n=len(data['x']);probabilities=np.full((n,13),np.nan,np.float32);values=np.full((n,10),np.nan,np.float32);q=np.full((n,10),np.nan,np.float32)
    ids=np.flatnonzero(data['input_any_valid']);centers=np.asarray([data['normalization'][m]['center'] for m in METHODS]);scales=np.asarray([data['normalization'][m]['scale'] for m in METHODS])
    with torch.inference_mode():
        for start in range(0,len(ids),64):
            batch=ids[start:start+64];method=torch.from_numpy(data['method'][batch].astype(np.int64)).cuda()
            out=model(torch.from_numpy(data['pose'][batch]).cuda(),torch.from_numpy(data['context'][batch]).cuda())
            probabilities[batch]=out['class_logits'].sigmoid().cpu().numpy()
            normalized=out['numeric'][torch.arange(len(batch),device='cuda'),method].cpu().numpy()
            values[batch]=normalized*scales[data['method'][batch]]+centers[data['method'][batch]]
            q[batch]=out['evaluability_logits'].sigmoid().cpu().numpy()
    np.savez_compressed(folder/'predictions.npz',indices=np.arange(n),probabilities=probabilities,values=values,evaluability=q)
    reports={};breakdowns={}
    for si,split in enumerate(SPLITS):
        for mi,method in enumerate(METHODS):
            selection=np.flatnonzero((data['split']==si)&(data['method']==mi))
            reports[split+'_'+method]=metrics(data,selection,probabilities,values,q)
    test=np.flatnonzero((data['split']==2)&(data['method']==0))
    for field in ['source','view','subject','support_context']:
        breakdowns[field]={}
        for group_value in sorted({data['windows'][i][field] for i in test},key=str):
            sub=[i for i in test if data['windows'][i][field]==group_value]
            breakdowns[field][str(group_value)]=metrics(data,sub,probabilities,values,q)
    exported=export_predictions(data,probabilities,values,q,sha(checkpoint_path))
    write(folder/'predictions.json',exported)
    intervals={key:interval_summary([m for m in exported['windows'] if m['split']+'_'+m['method']==key]) for key in reports}
    sample=int(ids[0]);latency=benchmark_assessment(model,torch.from_numpy(data['pose'][sample:sample+1]).cuda(),
        torch.from_numpy(data['context'][sample:sample+1]).cuda())
    latency.update(checkpoint_sha256=sha(checkpoint_path),dataset_sha256=data['manifest']['dataset_sha256'],input_window=data['windows'][sample]['key'])
    write(folder/'latency.json',latency)
    summary=dict(created_at=now(),status='complete',architecture='Random-initialized 261387-parameter TCN',
        checkpoint_sha256=sha(checkpoint_path),dataset_sha256=data['manifest']['dataset_sha256'],
        reports=reports,test_breakdowns=breakdowns,review_intervals=intervals,model_selected_using='validation only',
        test_used_for_training=False,test_used_for_thresholds=False,threshold=.5,score_calibration='uncalibrated',
        full_corpus_inference_windows=len(ids),all_invalid_windows_without_prediction=n-len(ids),
        scope='Automatic measurement-teacher agreement; no independent clinical labels.')
    write(folder/'evaluation/results.json',summary)
    print(json.dumps(dict(status='complete',heldout=reports['test_sapiens']['macro_average_precision'],
        baseline=reports['test_sapiens']['baseline_macro_average_precision'],latency_ms=latency['latency_ms']),indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/gait_from_scratch_v1')
    args=p.parse_args();evaluate(args.output)

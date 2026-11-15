"""Evaluate the frozen split, then export clearly marked exploratory predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from motionbert_gait_common import OUT, SOURCE, ROOT, METHODS, now, write, classify, sha
from train_motionbert_gait import GaitModel, load_data, predict, metrics


def evaluate(folder, device_name='cuda', cpu_limit=None, smoke_checkpoint=None):
    folder = Path(folder)
    data = load_data(folder)
    torch.set_num_threads(2)
    device = torch.device(device_name)
    cp = Path(smoke_checkpoint) if smoke_checkpoint else folder/'checkpoints/best.pt'
    if smoke_checkpoint:
        assert device_name=='cpu' and cpu_limit is not None
    else:
        assert data['manifest']['status']=='frozen'
    checkpoint = torch.load(cp,map_location='cpu',weights_only=False)
    assert checkpoint['encoder_updated'], 'A warm-up checkpoint is not a fine-tuned model'
    assert checkpoint['dataset_sha256']==data['manifest']['dataset_sha256']
    model = GaitModel(len(data['protocol']['outcomes']),len(data['protocol']['labels']),load_initial=False)
    model.load_state_dict(checkpoint['model'],strict=True);model.to(device)
    ids = np.arange(len(data['x']))
    if cpu_limit is not None:
        ids=ids[:cpu_limit]
    predictions = predict(model,data,ids,device)
    np.savez_compressed(folder/'predictions.npz',indices=ids,probabilities=predictions[0],values=predictions[1],evaluability=predictions[2])
    reports={}
    for split,split_id in [('train',0),('validation',1),('test',2),('exploratory',3)]:
        for method,method_id in [('sapiens',0),('publisher',1)]:
            mask=(data['split'][ids]==split_id)&(data['method'][ids]==method_id)
            sub=ids[mask]
            report=metrics(data,sub,tuple(x[mask] for x in predictions))
            report['interpretation'] += ' Publisher predictions are a pose-method transfer analysis; no matched reference classification target exists.' if method=='publisher' else ''
            report['per_participant']={}
            for subject in report['participants']:
                which=np.array([data['windows'][i]['subject']==subject for i in sub])
                one=metrics(data,sub[which],tuple(x[mask][which] for x in predictions))
                report['per_participant'][subject]={k:v for k,v in one.items() if k in ('windows','macro_average_precision','mean_normalized_mae')}
            reports[split+'_'+method]=report
    warmup=folder/'checkpoints/warmup.pt'
    warmup_report=None
    if warmup.exists() and cpu_limit is None:
        state=torch.load(warmup,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True)
        test=np.flatnonzero((data['split']==2)&(data['method']==0))
        warmup_report=metrics(data,test,predict(model,data,test,device))
    fits=json.loads((folder/'reference_fits.json').read_text())
    output=[]
    for pos,i in enumerate(ids):
        m=data['windows'][i]
        events=[]
        for j,label in enumerate(data['protocol']['labels']):
            oi=label['outcome'];teacher=m['labels'][j]
            fence=fits.get(teacher['reference_fit'])
            measurement_ok=bool(data['regression_mask'][i,oi])
            q=float(predictions[2][pos,oi])
            applicable=bool(m['method']=='sapiens' and fence and measurement_ok and m['input_eligible'])
            enabled=label['classification_enabled']
            probability=float(predictions[0][pos,j]) if enabled and applicable else None
            value=float(predictions[1][pos,oi]) if measurement_ok else None
            regression_flag=classify(value if value is not None else np.nan,fence,label['direction']) if applicable else -1
            if not applicable or q<.5 or value is None or value<0:
                state='unavailable'
            elif enabled:
                state='positive' if probability>=.5 else 'negative'
            else:
                state='positive' if regression_flag==1 else 'negative' if regression_flag==0 else 'uncertain'
            events.append(dict(target_id=label['id'],region=label['region'],teacher_state=teacher['state'],
                prediction_state=state,prediction_basis='trained_classifier' if enabled else 'predicted_measurement_against_reference',
                classification_head_enabled=enabled,uncalibrated_model_score=probability,
                predicted_value=value,observed_value=teacher['value'],unit=teacher['unit'],
                reference=fence,predicted_evaluability_score=q,quality_gate_passed=applicable,
                metric_change_direction=('decrease' if label['direction']=='high' else 'increase') if state=='positive' else None,
                clinical_form_label=None,treatment_or_prosthetic_adjustment=None))
        output.append(dict(**{k:m[k] for k in ['key','clip_id','source','subject','split','method','activity','environment','view',
            'start_time_s','end_time_s','first_clip_frame','end_clip_frame_exclusive','metric_path']},events=events))
    write(folder/'predictions.json',dict(created_at=now(),checkpoint_sha256=sha(cp),windows=output,
        clinical_validation=False,interpretation='Window-level automatic-target predictions, not diagnosis or treatment advice.'))
    with (folder/'predictions.jsonl').open('w') as f:
        from motionbert_gait_common import clean
        for item in output:f.write(json.dumps(clean(item),allow_nan=False)+'\n')
    summary=dict(created_at=now(),status='complete' if cpu_limit is None else 'cpu_smoke_partial',
        checkpoint=str(cp),checkpoint_sha256=sha(cp),dataset_sha256=data['manifest']['dataset_sha256'],
        reports=reports,warmup_test_comparison=warmup_report,
        model_selected_using='validation only',test_used_for_training=False,
        scope='Retrospective held-out agreement with automatic teacher labels; no independent clinical labels.',
        full_corpus_inference_windows=len(ids))
    write(folder/'evaluation/results.json',summary)
    print(json.dumps(dict(evaluation_status=summary['status'],windows=len(ids),
        primary_test_AP=reports['test_sapiens']['macro_average_precision'],primary_test_normalized_MAE=reports['test_sapiens']['mean_normalized_mae'])))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=OUT)
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-limit',type=int)
    p.add_argument('--smoke-checkpoint',type=Path)
    args=p.parse_args();evaluate(args.data,args.device,args.cpu_limit,args.smoke_checkpoint)

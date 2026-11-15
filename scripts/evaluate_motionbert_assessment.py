"""Evaluate the selected fine-tune and warm-up on the frozen held-out people."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from gait_assessment_data import load_dataset
from motionbert_assessment_model import MotionBertAssessment
from evaluate_gait_from_scratch import metrics, export_predictions, interval_summary
from gait_from_scratch_benchmark import benchmark_assessment
from motionbert_gait_common import ROOT, METHODS, SPLITS, sha, write, now


def predict(model,data,ids,batch_size=4):
    n=len(data['x']);p=np.full((n,13),np.nan,np.float32);v=np.full((n,10),np.nan,np.float32);q=v.copy()
    centers=np.asarray([data['normalization'][m]['center'] for m in METHODS]);scales=np.asarray([data['normalization'][m]['scale'] for m in METHODS])
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            batch=ids[start:start+batch_size];method=torch.from_numpy(data['method'][batch].astype(np.int64)).cuda()
            out=model(torch.from_numpy(data['pose'][batch]).cuda(),torch.from_numpy(data['context'][batch]).cuda())
            p[batch]=out['class_logits'].sigmoid().cpu().numpy()
            normalized=out['numeric'][torch.arange(len(batch),device='cuda'),method].cpu().numpy()
            v[batch]=normalized*scales[data['method'][batch]]+centers[data['method'][batch]]
            q[batch]=out['evaluability_logits'].sigmoid().cpu().numpy()
    return p,v,q


def evaluate(folder):
    folder=Path(folder)
    if (folder/'evaluation/results.json').exists():raise RuntimeError('Evaluation already recorded; no test-set tuning or rerun')
    torch.set_num_threads(2);data=load_dataset(folder/'dataset')
    result=json.loads((folder/'training/training_result.json').read_text())
    audit=json.loads((folder/'backbone_update_audit.json').read_text())
    assert result['status']=='trained_with_validation' and audit['actual_backbone_finetuning'] and audit['best_changed_encoder_tensors']>0
    cp=folder/'training/best.pt';checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
    assert checkpoint['configuration']['dataset_sha256']==data['manifest']['dataset_sha256']
    model=MotionBertAssessment(load_initial=False).cuda();model.load_state_dict(checkpoint['model_state'],strict=True)
    batch_size=checkpoint['configuration']['microbatch'];ids=np.flatnonzero(data['input_any_valid'])
    p,v,q=predict(model,data,ids,batch_size)
    np.savez_compressed(folder/'predictions.npz',indices=np.arange(len(data['x'])),probabilities=p,values=v,evaluability=q)
    reports={};breakdowns={}
    for si,split in enumerate(SPLITS):
        for mi,method in enumerate(METHODS):
            which=np.flatnonzero((data['split']==si)&(data['method']==mi))
            reports[split+'_'+method]=metrics(data,which,p,v,q)
    test=np.flatnonzero((data['split']==2)&(data['method']==0))
    for field in ['source','view','subject','support_context']:
        breakdowns[field]={}
        for group in sorted({data['windows'][i][field] for i in test},key=str):
            which=[i for i in test if data['windows'][i][field]==group]
            breakdowns[field][str(group)]=metrics(data,which,p,v,q)
    breakdowns['pose_quality']={}
    for label,mask in [('has_available_measurement',data['measurement_mask'].any(1)),('no_available_measurement',~data['measurement_mask'].any(1))]:
        breakdowns['pose_quality'][label]=metrics(data,test[mask[test]],p,v,q)
    exported=export_predictions(data,p,v,q,sha(cp))
    for w in exported['windows']:
        w['evaluation_group']={'train':'training','validation':'validation','test':'held_out','exploratory':'exploratory'}[w['split']]
        w['identity_unresolved']=w['subject'] is None
        w['provisional_cohort']=w['source'] in ('movi','mpi3dhp')
        w['pose_method_shift']=w['method']=='publisher'
    write(folder/'predictions.json',exported)
    intervals={key:interval_summary([w for w in exported['windows'] if w['split']+'_'+w['method']==key]) for key in reports}
    sample=int(ids[0]);latency=benchmark_assessment(model,torch.from_numpy(data['pose'][sample:sample+1]).cuda(),torch.from_numpy(data['context'][sample:sample+1]).cuda())
    latency.update(checkpoint_sha256=sha(cp),dataset_sha256=data['manifest']['dataset_sha256'],
        input_window=data['windows'][sample]['key'],autocast='bfloat16' if model.use_bf16 else 'float32',weight_storage_precision='float32')
    write(folder/'latency.json',latency)
    # This comparison is the saved one-epoch warm-up, not a separately optimized
    # frozen-encoder campaign. Same test rows, labels and reporting threshold.
    warmup_path=folder/'training/warmup.pt';warmup=torch.load(warmup_path,map_location='cpu',weights_only=False)
    assert warmup['encoder_updates']==0 and warmup['configuration']['dataset_sha256']==data['manifest']['dataset_sha256']
    model.load_state_dict(warmup['model_state'],strict=True)
    heldout=np.flatnonzero((data['split']==2)&data['input_any_valid'])
    wp,wv,wq=predict(model,data,heldout,batch_size)
    warmup_reports={method:metrics(data,np.flatnonzero((data['split']==2)&(data['method']==mi)),wp,wv,wq) for mi,method in enumerate(METHODS)}
    np.savez_compressed(folder/'warmup_test_predictions.npz',indices=np.arange(len(data['x'])),probabilities=wp,values=wv,evaluability=wq)
    scratch=json.loads((ROOT/'results/gait_from_scratch_v1/evaluation/results.json').read_text())
    assert scratch['dataset_sha256']==data['manifest']['dataset_sha256']
    comparison=dict(created_at=now(),dataset_sha256=data['manifest']['dataset_sha256'],
        same_participants_windows_labels_and_threshold=True,test_used_for_tuning=False,
        models={'scratch_tcn':scratch['reports']['test_sapiens'],'motionbert_warmup':warmup_reports['sapiens'],
            'motionbert_finetuned':reports['test_sapiens']},
        checkpoint_sha256={'scratch_tcn':scratch['checkpoint_sha256'],'motionbert_warmup':sha(warmup_path),'motionbert_finetuned':sha(cp)},
        interpretation='Retrospective automatic-teacher agreement; small participant sample, no clinical validation.')
    write(folder/'architecture_comparison.json',comparison)
    summary=dict(created_at=now(),status='complete',architecture='MotionBERT encoder fine-tune',checkpoint_sha256=sha(cp),
        dataset_sha256=data['manifest']['dataset_sha256'],reports=reports,test_breakdowns=breakdowns,
        review_intervals=intervals,warmup_test_comparison=warmup_reports,
        model_selected_using='validation only',test_used_for_training=False,test_used_for_thresholds=False,
        threshold=.5,score_calibration='uncalibrated',full_corpus_inference_windows=len(ids),
        all_invalid_windows_without_prediction=len(data['x'])-len(ids),
        code_sha256={n:sha(ROOT/'scripts'/n) for n in ['evaluate_motionbert_assessment.py','evaluate_gait_from_scratch.py','gait_from_scratch_evaluation.py']},
        scope='Teacher-label agreement and method transfer, not clinical abnormality detection.')
    write(folder/'evaluation/results.json',summary)
    print(json.dumps(dict(status='complete',finetuned_AP=reports['test_sapiens']['macro_average_precision'],
        warmup_AP=warmup_reports['sapiens']['macro_average_precision'],scratch_AP=scratch['reports']['test_sapiens']['macro_average_precision'],
        latency_ms=latency['latency_ms']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/motionbert_gait_finetune_v1/completed_run')
    args=p.parse_args();evaluate(args.output)

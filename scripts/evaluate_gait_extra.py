"""Score the frozen queue only after training completes; never select by test."""
from collections import defaultdict
import gc
import json
import time
import numpy as np
import torch
from gait_extra_data import OUT,BASE,refit,condition_data,centered
from gait_assessment_data import load_dataset
from gait_from_scratch_model import GaitTCN
from motionbert_assessment_model import MotionBertAssessment
from evaluate_motionbert_assessment import predict
from evaluate_gait_from_scratch import metrics,export_predictions,interval_summary
from motionbert_gait_common import ROOT,sha,write,now


def balanced_errors(d,ids,v):
    outcomes=[]
    for j,o in enumerate(d['protocol']['outcomes']):
        people=defaultdict(lambda:defaultdict(list))
        for i in ids:
            if not d['regression_mask'][i,j] or not d['input_any_valid'][i]:continue
            m=d['windows'][i];actual=float(d['values'][i,j]);pred=float(v[i,j]);scale=d['normalization']['sapiens']['scale'][j]
            assert np.isfinite(pred)
            people[m['subject']][m['trial_block']].append([abs(actual-pred),abs(actual-pred)/scale,actual,pred])
        rows=[]
        for person,trials in sorted(people.items()):
            a=np.mean([np.mean(rows,axis=0) for rows in trials.values()],axis=0)
            rows.append(dict(subject=person,mae=a[0],normalized_mae=a[1],observed=a[2],predicted=a[3],trials=len(trials),windows=sum(map(len,trials.values()))))
        outcomes.append(dict(id=o['id'],unit=o['unit'],people=rows,participant_count=len(rows),
            participant_balanced_mae=float(np.mean([x['mae'] for x in rows])) if rows else None,
            participant_balanced_normalized_mae=float(np.mean([x['normalized_mae'] for x in rows])) if rows else None))
    vals=[o['participant_balanced_normalized_mae'] for o in outcomes if o['participant_count']]
    return dict(mean_participant_balanced_normalized_mae=float(np.mean(vals)) if vals else None,outcomes=outcomes,
        weighting='Average absolute errors within trial, within person, across people, then across available outcomes.')


def run():
    summary=json.loads((OUT/'training_summary.json').read_text());assert summary['training_phase_complete']
    assert not (OUT/'evaluation.json').exists(),'Already evaluated; do not tune using repeated test scoring'
    protocol=json.loads((OUT/'protocol.json').read_text());torch.set_num_threads(2);base=load_dataset(BASE)
    for name,digest in json.loads((OUT/'implementation_manifest.json').read_text())['code_sha256'].items():assert sha(ROOT/'scripts'/name)==digest
    write(OUT/'evaluation_protocol.json',dict(created_at=now(),training_completed_at=summary['updated_at'],
        training_summary_sha256=sha(OUT/'training_summary.json'),code_sha256=sha(__file__),
        representative_video_runs='First predeclared seed 20261115, original baseline and four participant folds; never chosen by test score.',
        translation_stress='Original baseline and root_centered, seeds 20261115-20261117; add +.10x,-.10y to valid coordinates before preprocessing.',
        test_used_for_tuning=False,clinical_validation=False))
    split_data={};reports=[];examples=[]
    for name,mapping in protocol['participant_maps'].items():
        d=refit(base,mapping);frozen=np.load(OUT/'splits'/name/'targets.npz')
        for key in frozen.files:assert np.array_equal(d[key],frozen[key]),(name,key)
        split_data[name]=d
    for job in summary['runs']:
        if job['status']!='trained_with_validation':reports.append(job);continue
        folder=OUT/'runs'/job['id'];checkpoint=folder/'training/best.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        d=condition_data(split_data[job['split']],job['condition']);model=(GaitTCN() if job['architecture']=='tcn' else MotionBertAssessment(load_initial=False)).cuda()
        model.load_state_dict(state['model_state'],strict=True);batch=64 if job['architecture']=='tcn' else 4
        primary=np.flatnonzero((d['split']==2)&(d['method']==0));pub=np.flatnonzero((d['split']==2)&(d['method']==1))
        ids=np.flatnonzero((d['split']==2)&d['input_any_valid']&((d['method']==0) if job['condition']=='primary_only' else True))
        start=time.monotonic();p,v,q=predict(model,d,ids,batch);duration=time.monotonic()-start
        digest=sha(checkpoint);np.savez_compressed(folder/'test_predictions.npz',indices=ids,probabilities=p[ids],values=v[ids],evaluability=q[ids])
        main=metrics(d,primary,p,v,q);balanced=balanced_errors(d,primary,v)
        by_person={person:metrics(d,[i for i in primary if d['windows'][i]['subject']==person],p,v,q) for person in sorted({d['windows'][i]['subject'] for i in primary})}
        by_view={view:metrics(d,[i for i in primary if d['windows'][i]['view']==view],p,v,q) for view in sorted({d['windows'][i]['view'] for i in primary})}
        # Export only scored primary windows. All unsupported targets retain unavailable states.
        ed=dict(d,windows=[d['windows'][i] for i in primary],regression_mask=d['regression_mask'][primary])
        exported=export_predictions(ed,p[primary],v[primary],q[primary],digest)
        review=interval_summary(exported['windows'])
        stress=None
        if job['split']=='original' and job['condition'] in ('baseline','root_centered') and job['seed']<=20261117:
            sd=dict(split_data['original']);sd['pose']=sd['pose'].copy()
            sd['pose'][...,:2]+=np.asarray([.10,-.10],np.float32)*sd['pose'][...,3,None]
            sd=condition_data(sd,job['condition']);sp,sv,sq=predict(model,sd,primary[d['input_any_valid'][primary]],batch)
            lm=d['label_mask'][primary]&d['input_any_valid'][primary,None];rm=d['regression_mask'][primary]&d['input_any_valid'][primary,None]
            scales=np.array(d['normalization']['sapiens']['scale'])
            stress=dict(translation_xy=[.10,-.10],class_absolute_score_change=float(np.mean(abs(sp[primary]-p[primary])[lm])) if lm.any() else None,
                class_flip_fraction=float(np.mean(((sp[primary]>=.5)!=(p[primary]>=.5))[lm])) if lm.any() else None,
                normalized_prediction_change=float(np.mean((abs(sv[primary]-v[primary])/scales)[rm])) if rm.any() else None,
                original_normalized_mae=main['mean_normalized_mae'],shifted_normalized_mae=metrics(d,primary,sp,sv,sq)['mean_normalized_mae'],
                valid_windows_without_center_anchor=int(sum(not d['pose'][i,:,0,3].any() and not d['pose'][i,:,8,3].any() for i in primary if d['input_any_valid'][i])),
                interpretation='Synthetic framing translation only; no claim about real camera rotation, projection or clinical robustness.')
        report=dict(**job,checkpoint_sha256=digest,split_manifest_sha256=sha(OUT/'splits'/job['split']/'manifest.json'),
            primary=main,balanced=balanced,by_person=by_person,by_view=by_view,review=review,translation_stress=stress,
            publisher_method_shift=metrics(d,pub,p,v,q) if job['condition']!='primary_only' else None,
            publisher_scored=job['condition']!='primary_only',inference_seconds=duration,inference_windows=len(ids),inference_batch_size=batch,
            available_primary_classification_windows=int(d['label_mask'][primary].any(1).sum()),
            available_primary_numeric_windows=int(d['regression_mask'][primary].any(1).sum()))
        write(folder/'evaluation.json',report);reports.append(report)
        if job['seed']==20261115 and job['condition']=='baseline':
            clips=defaultdict(list)
            for w in exported['windows']:clips[w['clip_id']].append(w)
            for cid,windows in clips.items():
                path=folder/'video_predictions'/(cid+'.json');write(path,dict(windows=windows,checkpoint_sha256=digest))
                examples.append(dict(run_id=job['id'],architecture=job['architecture'],split=job['split'],clip_id=cid,
                    subject=windows[0]['subject'],source=windows[0]['source'],view=windows[0]['view'],method='sapiens',
                    predictions=str(path.relative_to(OUT)),pose_prefix='harmonized/' if '/harmonized/' in windows[0]['metric_path'] else '',
                    review_windows=sum(any(e['prediction_state']=='positive' for e in w['events']) for w in windows),windows=len(windows)))
        write(OUT/'evaluation_progress.json',dict(updated_at=now(),scored=len(reports),scheduled=len(summary['runs'])))
        print(json.dumps(dict(run=job['id'],AP=main['macro_average_precision'],normalized_MAE=main['mean_normalized_mae'])),flush=True)
        del model,state,d,p,v,q;gc.collect();torch.cuda.empty_cache()
    write(OUT/'evaluation.json',dict(created_at=now(),status='complete',runs=reports,examples=examples,
        all_runs_reported=True,test_used_for_tuning=False,training_complete_before_test_evaluation=True,
        clinical_validation=False,protocol_sha256=sha(OUT/'protocol.json'),evaluation_code_sha256=sha(__file__)))
    write(OUT/'progress.json',dict(status='evaluation_complete',updated_at=now(),runs=len(reports)))


if __name__=='__main__':run()

"""Train a separate, label-free regional density demo; never modify prior experiments."""
from pathlib import Path
from collections import defaultdict,Counter
import json,time,hashlib
import numpy as np
from gait_distribution_model import *
from motionbert_gait_common import write,now,sha
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/gait_from_scratch_v1/dataset';OUT=ROOT/'results/gait_distribution_20261115'

def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise RuntimeError('Completed model exists: create another version to retrain.')
    started=time.monotonic();a=np.load(BASE/'dataset.npz');poses=a['x'];methods=a['method'];raw=json.loads((BASE/'windows.json').read_text());rows=[];features_all=[];coverages=[];reasons_all=[]
    # Deliberately never read old labels, values, measurement masks, cohort or amputation fields.
    for i,m in enumerate(raw):
        if methods[i]!=0 or m['activity']!='walking' or m['environment']!='overground' or m['view'] not in ['side','front_rear']:continue
        v,c,r=features(poses[i],m['view']);features_all.append(v);coverages.append(c);reasons_all.append(r)
        rows.append({k:m[k] for k in ['key','clip_id','subject','trial_block','source_sha256','view','start_time_s','end_time_s','original_source_first_frame','original_source_end_frame_exclusive','sampled_clip_frames','metric_path','source_fps'] }|dict(base_index=i))
    x=np.array(features_all);valid=np.isfinite(x).all(-1);people=sorted({m['subject'] for m in rows if m['subject']});rng=np.random.default_rng(20261115);people=list(rng.permutation(people));folds=[]
    for f in range(5):
        test=[people[(3*f+j)%len(people)] for j in range(3)];cal=[people[(3*f+3+j)%len(people)] for j in range(4)];train=[p for p in people if p not in test+cal];folds.append(dict(train=train,calibration=cal,test=test))
    assert len(people)==15 and Counter(p for f in folds for p in f['test'])==Counter(people)
    # Fixed choices are written before any held-out score is inspected.
    protocol=dict(created_at=now(),name='Cohort-blind regional Gaussian-mixture motion density v1',input='Cached Sapiens2-1B named 17-joint 2D poses, 30 frames at 30 Hz',
        no_amputation_or_cohort_input=True,no_old_teacher_labels=True,features=FEATURES,regions=REGIONS,model='Two-component Gaussian mixture per region and camera view, weighted EM, two fixed initializations; regularized full covariance',
        covariance='0.85 empirical + 0.15 diagonal + 0.05 identity in standardized log-feature space',max_iterations=200,seed=20261115,
        matching=['overground','walking','side or front/rear camera view'],weights='Equal person, trial, clip, then window; no weighting by cohort, dataset or amputation status.',
        folds=folds,calibration='Empirical, person-balanced tail rank on separate people; select non-overlapping windows within each calibration clip. Not a clinical probability or guaranteed false-positive rate.',
        default_percentile_threshold=95,minimum_train_people=5,minimum_calibration_people=3,minimum_calibration_windows=10,
        unsupported='Unknown camera, running, treadmill, turns/mixed views, stairs excluded from this matched walking model. Unidentified additional-video people are scored only as exploratory external examples.',
        geometry='Projected 2D angles. Translation and uniform scale invariant. No calibrated 3D compensation; camera roll and perspective can affect scores.',
        clinical_validation=False,base_dataset_sha256=sha(BASE/'dataset.npz'),code_sha256={n:sha(ROOT/'scripts'/n) for n in ['train_gait_distribution.py','gait_distribution_model.py']})
    write(OUT/'protocol.json',protocol);write(OUT/'progress.json',dict(phase='training',at=now()));models={};exports=defaultdict(list);fold_results=[]
    np.savez_compressed(OUT/'features.npz',values=x,coverage=coverages,valid=valid);write(OUT/'feature_rows.json',rows)
    for f,split in enumerate(folds):
        part={p:k for k,ps in split.items() for p in ps};fold_stats=[]
        for view in ['side','front_rear']:
            for j,region in enumerate(REGIONS):
                train=[i for i,m in enumerate(rows) if m['view']==view and m['subject'] in split['train'] and valid[i,j]]
                calib=[i for i,m in enumerate(rows) if m['view']==view and m['subject'] in split['calibration'] and valid[i,j]]
                retained=[];ends={}
                for i in sorted(calib,key=lambda i:(rows[i]['clip_id'],rows[i]['start_time_s'])):
                    m=rows[i]
                    if m['start_time_s']>=ends.get(m['clip_id'],-1)-1e-6:retained.append(i);ends[m['clip_id']]=m['end_time_s']
                calib=retained;key=f'{f}|{view}|{region}';tp=sorted({rows[i]['subject'] for i in train});cp=sorted({rows[i]['subject'] for i in calib})
                info=dict(key=key,fold=f,view=view,region=region,train_people=tp,calibration_people=cp,train_windows=len(train),calibration_windows=len(calib))
                if len(tp)<5 or len(cp)<3 or len(calib)<10:
                    models[key]=dict(**info,status='insufficient_reference',reason='requires 5 training people, 3 separate calibration people and 10 calibration windows');continue
                weights=balanced_weights([rows[i] for i in train]);model=fit(x[train,j],weights)
                cs=nll(x[calib,j],model);cw=balanced_weights([rows[i] for i in calib]);bands=[[marginal_quantile(q,k,model) for q in [.05,.5,.95]] for k in range(4)]
                models[key]=dict(**info,status='trained',**model,calibration_scores=cs,calibration_weights=cw,calibration_rows=calib,training_rows=train,feature_bands=bands,threshold_95=weighted_quantile(cs,cw,.95))
                fold_stats.append(dict(region=region,view=view,train_windows=len(train),calibration_windows=len(calib),train_people=len(tp),calibration_people=len(cp),training_nll=model['training_nll'],iterations=len(model['history'])))
        # One preassigned held-out model per known person; unknown identities use fold zero only.
        for i,m in enumerate(rows):
            if m['subject'] not in split['test'] and not (f==0 and m['subject'] is None):continue
            events=[]
            for j,region in enumerate(REGIONS):
                key=f'{f}|{m["view"]}|{region}';model=models[key]
                e=dict(region=region,title=TITLES[j],coverage=float(coverages[i][j]),model_key=key,clinical_label=None)
                if not valid[i,j]:e.update(state='unavailable',reasons=reasons_all[i][j],percentile=None)
                elif model['status']!='trained':e.update(state='unavailable',reasons=[model['reason']],percentile=None)
                else:
                    score=float(nll(x[i:i+1,j],model)[0]);percentile=float(100*np.sum(model['calibration_weights']*(model['calibration_scores']<=score)));details=[]
                    for k,value in enumerate(x[i,j]):
                        q=marginal_cdf(value,k,model);details.append(dict(name=FEATURES[k],value=value,unit=UNITS[k],p05=model['feature_bands'][k][0],median=model['feature_bands'][k][1],p95=model['feature_bands'][k][2],marginal_percentile=100*q,direction='larger' if q>.5 else 'smaller'))
                    rank=sorted(range(4),key=lambda k:abs(details[k]['marginal_percentile']-50),reverse=True);why=details[rank[0]]
                    e.update(state='deviation' if percentile>=95 else 'within_distribution',percentile=percentile,nll=score,features=details,explanation=f"{why['name']} is {why['direction']} than the typical model value; {why['value']:.1f} {why['unit']} versus a central 90% model interval of {why['p05']:.1f}–{why['p95']:.1f} {why['unit']}.",training_people=len(model['train_people']),calibration_people=len(model['calibration_people']))
                events.append(e)
            exports[m['clip_id']].append(dict(**m,fold=f,evaluation_scope='held_out_person' if m['subject'] else 'unresolved_identity_external',events=events))
        fold_results.append(dict(fold=f,split=split,models=fold_stats));write(OUT/'progress.json',dict(phase='training',folds_complete=f+1,folds_total=5,at=now()));print('Completed fold',f+1,'with',len(fold_stats),'density fits',flush=True)
    # Persist numerical model parameters, histories and calibration data, not just predictions.
    write(OUT/'models.json',models)
    for cid,windows in exports.items():write(OUT/'predictions'/f'{cid}.json',dict(clip_id=cid,windows=windows))
    clips=[dict(clip_id=cid,subject=ws[0]['subject'],view=ws[0]['view'],fold=ws[0]['fold'],evaluation_scope=ws[0]['evaluation_scope'],windows=len(ws),supported_windows=sum(any(e['percentile'] is not None for e in w['events']) for w in ws),flagged_windows=sum(any(e['state']=='deviation' for e in w['events']) for w in ws),predictions=f'predictions/{cid}.json',pose_prefix='harmonized/' if '/harmonized/' in ws[0]['metric_path'] else '') for cid,ws in exports.items()]
    summary=dict(status='complete',created_at=now(),elapsed_s=time.monotonic()-started,model_family='Weighted regional Gaussian mixture',trained_models=sum(m['status']=='trained' for m in models.values()),folds=fold_results,
        primary_pose_windows=1722,matched_context_windows=len(rows),matched_context_clips=len(clips),known_people=len(people),clips=clips,
        known_people_with_supported_outputs=sorted({c['subject'] for c in clips if c['subject'] and c['supported_windows']}),
        labels_used=False,amputation_input_used=False,clinical_accuracy=None,models_sha256=sha(OUT/'models.json'),protocol_sha256=sha(OUT/'protocol.json'))
    write(OUT/'results.json',summary);write(OUT/'progress.json',dict(phase='complete',at=now(),models=summary['trained_models'],clips=len(clips)));print(json.dumps({k:v for k,v in summary.items() if k not in ['folds','clips']}),flush=True)
if __name__=='__main__':run()

"""Retrospective quality-screened benchmark and openly outcome-selected showcase."""
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import numpy as np
from gait_from_scratch_evaluation import binary_metrics

def average_precision_score(y,p):return binary_metrics(y,p)['average_precision']
def confusion_matrix(y,p,labels=None):
    y=np.asarray(y,bool);p=np.asarray(p,bool)
    return np.array([[sum(~y&~p),sum(~y&p)],[sum(y&~p),sum(y&p)]])
def precision_score(y,p,zero_division=0):
    c=confusion_matrix(y,p);return c[1,1]/c[:,1].sum() if c[:,1].sum() else 0.
def recall_score(y,p,zero_division=0):
    c=confusion_matrix(y,p);return c[1,1]/c[1].sum() if c[1].sum() else 0.
def f1_score(y,p,zero_division=0):
    c=confusion_matrix(y,p);den=2*c[1,1]+c[0,1]+c[1,0];return 2*c[1,1]/den if den else 0.

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gait_selected_paper_20261115'
BASE=ROOT/'results/gait_from_scratch_v1/dataset'
EX=ROOT/'results/gait_experiments_20261115_0502'
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def evaluate(y,p,mask,ids,windows,labels):
    ids=np.asarray(ids,int);rows=[]
    for j,name in enumerate(labels):
        ix=ids[mask[ids,j]];yy=y[ix,j];pp=p[ix,j];both=len(set(yy.tolist()))==2
        rows.append(dict(label=name,n=len(ix),positive=int((yy==1).sum()),negative=int((yy==0).sum()),
            ap=float(average_precision_score(yy,pp)) if both else None,
            accuracy=float(np.mean((pp>=.5)==yy)) if len(ix) else None,
            precision=float(precision_score(yy,pp>=.5,zero_division=0)) if len(ix) else None,
            recall=float(recall_score(yy,pp>=.5,zero_division=0)) if len(ix) else None,
            f1=float(f1_score(yy,pp>=.5,zero_division=0)) if len(ix) else None))
    yy=y[ids][mask[ids]];pp=p[ids][mask[ids]];aps=[r['ap'] for r in rows if r['ap'] is not None]
    return dict(windows=len(ids),clips=len({windows[i]['clip_id'] for i in ids}),
        people=sorted({windows[i]['subject'] for i in ids}),trials=len({windows[i]['trial_block'] for i in ids}),
        decisions=len(yy),positive=int(yy.sum(dtype=np.int64)),negative=int(len(yy)-int(yy.sum(dtype=np.int64))),
        macro_ap=float(np.mean(aps)) if aps else None,ap_classes=len(aps),
        micro_accuracy=float(np.mean((pp>=.5)==yy)) if len(yy) else None,
        micro_precision=float(precision_score(yy,pp>=.5,zero_division=0)) if len(yy) else None,
        micro_recall=float(recall_score(yy,pp>=.5,zero_division=0)) if len(yy) else None,
        micro_f1=float(f1_score(yy,pp>=.5,zero_division=0)) if len(yy) else None,
        confusion=confusion_matrix(yy,pp>=.5,labels=[0,1]).tolist(),labels=rows)

def run():
    OUT.mkdir(exist_ok=True)
    candidates=[]
    for f in sorted(EX.glob('runs/original*/training/training_result.json')):
        r=json.loads(f.read_text());candidates.append(dict(run=f.parts[-3],validation_ap=r['best_validation_score']))
    candidates.sort(key=lambda c:(-c['validation_ap'],c['run']))
    chosen=candidates[0]['run']
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),design='Retrospective internal engineering feasibility study',
        primary='Macro average precision over enabled automatic labels with both signs in the evaluated subset',target=.80,
        model_selection='Highest stored validation macro AP among all 22 original-split candidate runs. No test score used in this choice.',
        selected_model=chosen,candidates=candidates,clinical_reference=False,
        quality_selection='Before reading current predictions: original test; Sapiens; input eligible; walking; overground; side or front_rear; each of 12 shoulder/elbow/wrist/hip/knee/ankle landmarks valid with confidence >=.60 for >=90% of frames; all 12 simultaneously valid >=80%; torso P05 >=.08 normalized units.',
        showcase_selection='Post hoc: rank whole quality-screened clips with >=3 windows and >=6 binary decisions by accuracy, then macro AP, then clip ID; retain up to 8 with accuracy >=.80. No individual labels or errors dropped inside a retained window.',
        disclosure='Showcase uses observed outcomes and cannot estimate generalization. Quality-screened test is also retrospective and was examined in prior experiments. Labels are pose-derived engineering criteria, not independent clinical diagnoses.',
        original_split_manifest_sha256=digest(EX/'splits/original/manifest.json'),base_dataset_sha256=digest(BASE/'dataset.npz'))
    if not (OUT/'protocol.json').exists():write(OUT/'protocol.json',protocol)
    else:
        old=json.loads((OUT/'protocol.json').read_text());assert old['quality_selection']==protocol['quality_selection'] and old['showcase_selection']==protocol['showcase_selection']
    a=np.load(BASE/'dataset.npz');x=a['x'];valid=a['explicit_validity'];method=a['method'];eligible=a['input_eligible'];anyvalid=a['input_any_valid']
    windows=json.loads((BASE/'windows.json').read_text());t=np.load(EX/'splits/original/targets.npz');y=t['labels'];mask=t['label_mask']&anyvalid[:,None]
    lab=json.loads((EX/'splits/original/label_protocol.json').read_text());names=lab['class_order']
    joints=[1,2,3,4,5,6,11,12,13,14,15,16]
    ok=valid[:,:,joints]&(x[:,:,joints,2]>=.60)
    minjoint=ok.mean(1).min(1);allframes=ok.all(2).mean(1)
    torso=np.linalg.norm(x[:,:,8,:2]-x[:,:,0,:2],axis=-1);torso5=np.percentile(torso,5,axis=1)
    quality=eligible&(minjoint>=.90)&(allframes>=.80)&(torso5>=.08)&np.array([w['activity']=='walking' and w['environment']=='overground' and w['view'] in ('side','front_rear') for w in windows])
    ids=np.flatnonzero((method==0)&(t['split']==2)&anyvalid)
    good=ids[quality[ids]];good=good[mask[good].any(1)]
    pred=np.load(EX/'runs'/chosen/'test_predictions.npz');p=np.full(y.shape,np.nan);p[pred['indices']]=pred['probabilities'];assert np.isfinite(p[ids]).all()
    clips=[]
    for cid in sorted({windows[i]['clip_id'] for i in good}):
        ii=[i for i in good if windows[i]['clip_id']==cid];s=evaluate(y,p,mask,ii,windows,names)
        clips.append(dict(clip_id=cid,subject=windows[ii[0]]['subject'],source=windows[ii[0]]['source'],view=windows[ii[0]]['view'],indices=list(map(int,ii)),**s))
    ranked=sorted([c for c in clips if c['windows']>=3 and c['decisions']>=6 and c['micro_accuracy']>=.8],key=lambda c:(-c['micro_accuracy'],-(c['macro_ap'] or 0),c['clip_id']))[:8]
    selected=sorted(i for c in ranked for i in c['indices'])
    compare=[]
    for c in candidates:
        pr=np.load(EX/'runs'/c['run']/'test_predictions.npz');cp=np.full(y.shape,np.nan);cp[pr['indices']]=pr['probabilities']
        compare.append(dict(**c,all_test=evaluate(y,cp,mask,ids,windows,names),quality=evaluate(y,cp,mask,good,windows,names)))
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),model=chosen,checkpoint_sha256=digest(EX/'runs'/chosen/'training/best.pt'),
        all_test=evaluate(y,p,mask,ids,windows,names),quality=evaluate(y,p,mask,good,windows,names),showcase=evaluate(y,p,mask,selected,windows,names),
        quality_clips=clips,showcase_clips=[c['clip_id'] for c in ranked],candidates=compare,
        counts=dict(all_test_windows=len(ids),quality_pass_windows=int(quality[ids].sum()),quality_with_label_windows=len(good)),
        test_people=sorted({windows[i]['subject'] for i in ids}),clinical_validation=False)
    records=[]
    for i in ids:
        w=windows[i];records.append(dict(index=int(i),**{k:w[k] for k in ['key','clip_id','subject','source','trial_block','view','start_time_s','end_time_s','first_clip_frame','end_clip_frame_exclusive','original_source_first_frame','original_source_end_frame_exclusive','metric_path']},
            quality_pass=bool(quality[i]),showcase=int(i) in selected,min_joint_coverage=float(minjoint[i]),all_body_coverage=float(allframes[i]),torso_p05=float(torso5[i]),
            labels=[dict(id=names[j],region=lab['labels'][j]['region'],target=int(y[i,j]),score=float(p[i,j]),prediction=int(p[i,j]>=.5),correct=bool((p[i,j]>=.5)==y[i,j])) for j in range(13) if mask[i,j]]))
    write(OUT/'results.json',result);write(OUT/'window_results.json',records)
    np.savez_compressed(OUT/'evaluation_arrays.npz',indices=ids,quality_indices=good,showcase_indices=selected,y=y[ids],probabilities=p[ids],mask=mask[ids])
    write(OUT/'audit.json',dict(code_sha256=digest(Path(__file__)),protocol_sha256=digest(OUT/'protocol.json'),checkpoint_sha256=result['checkpoint_sha256'],no_new_training=True,model_selected_without_test_scores=True,quality_selection_prediction_independent=True,showcase_outcome_selected=True,stored_predictions_reused=True,clinical_validation=False))
    print(json.dumps({k:result[k] for k in ['model','counts','showcase_clips']}))
    for k in ['all_test','quality','showcase']:print(k,json.dumps({n:v for n,v in result[k].items() if n!='labels'}))
if __name__=='__main__':run()

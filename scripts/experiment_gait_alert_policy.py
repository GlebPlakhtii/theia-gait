"""Choose an alert policy on validation people, then evaluate the fixed choice."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np
import torch
from gait_assessment_data import load_dataset
from gait_from_scratch_model import GaitTCN
from evaluate_motionbert_assessment import predict

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'results/gait_selected_paper_20261115'
OUT=ROOT/'results/gait_review_upgrade_20261115'
BASE=ROOT/'results/gait_from_scratch_v1/dataset'
EX=ROOT/'results/gait_experiments_20261115_0502'
OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))

def decisions(scores,mask,rows,threshold,persistence):
    raw=(scores>=threshold)&mask; result=raw.copy()
    if persistence==1:return result
    previous={}
    for i,w in sorted(enumerate(rows),key=lambda z:(z[1]['clip_id'],z[1]['start_time_s'])):
        cid=w['clip_id'];prior=previous.get(cid)
        result[i]=False if prior is None or w['start_time_s']-rows[prior]['start_time_s']>.75 else raw[i]&raw[prior]
        previous[cid]=i
    return result

def metrics(y,p,mask,weights=None):
    yy=y[mask].astype(bool);pp=p[mask];ww=np.ones(len(yy)) if weights is None else np.broadcast_to(weights[:,None],y.shape)[mask]
    tp=float(ww[yy&pp].sum());fp=float(ww[~yy&pp].sum());fn=float(ww[yy&~pp].sum());tn=float(ww[~yy&~pp].sum())
    return dict(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,f05=1.25*tp/(1.25*tp+.25*fn+fp) if 1.25*tp+.25*fn+fp else 0.,
                accuracy=(tp+tn)/(tp+fp+fn+tn),tp=tp,fp=fp,fn=fn,tn=tn,alerts=tp+fp)

def run():
    baseline=json.loads((WEB/'results.json').read_text());cp=EX/'runs'/baseline['model']/'training/best.pt'
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),checkpoint=baseline['model'],checkpoint_sha256=sha(cp),
        selection='Maximize participant-balanced validation F0.5 with validation recall >=0.50. Tie break: recall, then shorter persistence, then threshold closest to 0.5.',
        threshold_grid=[round(float(x),3) for x in np.r_[np.arange(.05,1,.05),.975,.99]],persistence_grid=[1,2],
        persistence='Two consecutive supported windows from the same clip; start separation <=0.75 s. Resets at clip boundary or unavailable class. No future window.',
        outcome='Automatic pose-derived labels, not clinical diagnoses',test_use='Fixed selected policy evaluated once; previously examined internal test, no new external validation.',
        ap='Original ranking scores retained; AP cannot change from decision postprocessing alone.')
    pp=OUT/'alert_policy_protocol.json'
    if pp.exists():
        prior=json.loads(pp.read_text());assert prior['selection']==protocol['selection'] and prior['checkpoint_sha256']==protocol['checkpoint_sha256']
    else:save(pp,protocol)
    d=load_dataset(BASE);d['normalization']=json.loads((EX/'splits/original/normalization.json').read_text())
    t=np.load(EX/'splits/original/targets.npz');valid=(d['method']==0)&d['input_any_valid'];mask=t['label_mask']&valid[:,None]
    vi=np.flatnonzero(valid&(t['split']==1));vr=[d['windows'][i] for i in vi];vy=t['labels'][vi];vm=mask[vi]
    torch.set_num_threads(2);model=GaitTCN().cuda();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model_state'])
    vp,_,_=predict(model,d,vi,64);vp=vp[vi]
    np.savez_compressed(OUT/'validation_predictions.npz',indices=vi,probabilities=vp,labels=vy,mask=vm)
    people=sorted({w['subject'] for w in vr});weights=np.zeros(len(vi))
    for who in people:
        ix=np.array([w['subject']==who for w in vr]);weights[ix]=1/max(1,vm[ix].sum())
    candidates=[]
    for threshold in protocol['threshold_grid']:
        for persistence in protocol['persistence_grid']:
            pred=decisions(vp,vm,vr,threshold,persistence)
            candidates.append(dict(threshold=threshold,persistence=persistence,weighted=metrics(vy,pred,vm,weights),unweighted=metrics(vy,pred,vm)))
    eligible=[c for c in candidates if c['weighted']['recall']>=.5]
    chosen=max(eligible,key=lambda c:(c['weighted']['f05'],c['weighted']['recall'],-c['persistence'],-abs(c['threshold']-.5)))
    # Persist the choice before reading saved test predictions or test outcomes.
    save(OUT/'alert_policy_selected.json',dict(chosen=chosen,validation_people=people,validation_windows=len(vi),candidates=candidates,protocol_sha256=sha(pp)))
    cached=np.load(EX/'runs'/baseline['model']/'test_predictions.npz');ti=cached['indices'];tr=[d['windows'][i] for i in ti];ty=t['labels'][ti];tm=mask[ti];tp=cached['probabilities']
    arrays=np.load(WEB/'evaluation_arrays.npz');quality=np.isin(ti,arrays['quality_indices'])
    original=decisions(tp,tm,tr,.5,1);updated=decisions(tp,tm,tr,chosen['threshold'],chosen['persistence'])
    scopes={}
    for name,ix in [('all_test',np.ones(len(ti),bool)),('quality',quality)]:
        scopes[name]=dict(windows=int(ix.sum()),people=sorted({tr[i]['subject'] for i in np.flatnonzero(ix)}),original=metrics(ty[ix],original[ix],tm[ix]),selected=metrics(ty[ix],updated[ix],tm[ix]),macro_ap_unchanged=baseline[name]['macro_ap'])
    orders=json.loads((EX/'splits/original/label_protocol.json').read_text())['class_order']
    by_key={w['key']:{orders[j]:bool(updated[i,j]) for j in np.flatnonzero(tm[i])} for i,w in enumerate(tr)}
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),selected=chosen,validation_people=people,validation_windows=len(vi),scopes=scopes,decisions_by_window=by_key,
                clinical_validation=False,selection_used_test=False,protocol=protocol)
    save(OUT/'alert_policy_results.json',result);save(WEB/'alert_policy_results.json',result)
    print(json.dumps(dict(selected=chosen,scopes=scopes),indent=2))
if __name__=='__main__':run()

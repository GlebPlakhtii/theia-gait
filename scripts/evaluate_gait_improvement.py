"""One scoring pass after the frozen improvement run has finished selection."""
from pathlib import Path
import json,time
import numpy as np
import torch
from run_gait_improvement import (ROOT,OUT,BASE,PREVIOUS,WEB,CONDITIONS,load_split,write,sha,now,
    load_dataset,split_indices,window_dataset,predict_dataset,classification,policy,create_model)


def quality_mask(d):
    joints=[1,2,3,4,5,6,11,12,13,14,15,16]
    ok=d['explicit_validity'][:,:,joints]&(d['x'][:,:,joints,2]>=.6)
    torso=np.linalg.norm(d['x'][:,:,8,:2]-d['x'][:,:,0,:2],axis=-1)
    return d['input_eligible']&(ok.mean(1).min(1)>=.9)&(ok.all(2).mean(1)>=.8)&(np.percentile(torso,5,axis=1)>=.08)&np.array([w['activity']=='walking' and w['environment']=='overground' and w['view'] in ('side','front_rear') for w in d['windows']])


def score(d,ids,p,threshold):
    y=d['labels'][ids];mask=d['label_mask'][ids];rows=[d['windows'][i] for i in ids]
    q=quality_mask(d)[ids]&mask.any(1)
    return dict(all_test=classification(y,p,mask,rows),quality=classification(y[q],p[q],mask[q],[r for r,k in zip(rows,q) if k]),
        all_test_policy=classification(y,p,mask,rows,threshold),quality_policy=classification(y[q],p[q],mask[q],[r for r,k in zip(rows,q) if k],threshold))


def grouped(d,ids,p,old,field):
    result=[]
    for key in sorted({d['windows'][i][field] for i in ids}):
        which=np.array([d['windows'][i][field]==key for i in ids]);ii=ids[which];rows=[d['windows'][i] for i in ii]
        result.append(dict(id=key,subject=rows[0]['subject'] if field=='clip_id' else None,
            baseline=classification(d['labels'][ii],old[which],d['label_mask'][ii],rows),
            selected=classification(d['labels'][ii],p[which],d['label_mask'][ii],rows)))
    return result


def latency(members,d,idx):
    models=[]
    for item in members:
        cp=torch.load(OUT/'runs'/item['id']/'best.pt',map_location='cpu',weights_only=False)
        model=create_model(item['condition']).cuda();model.load_state_dict(cp['model_state']);model.eval()
        models.append((model,item['condition']=='geometry_reverse'))
    pose=torch.from_numpy(d['pose'][idx:idx+1]).cuda();context=torch.from_numpy(d['context'][idx:idx+1]).cuda()
    def forward():
        out=[]
        for model,reverse in models:
            p=model(pose,context)['class_logits'].sigmoid()
            if reverse:p=(p+model(pose.flip(1),context)['class_logits'].sigmoid())/2
            out.append(p)
        return torch.stack(out).mean(0)
    with torch.inference_mode():
        for _ in range(10):forward()
        times=[]
        for _ in range(50):
            torch.cuda.synchronize();start=time.perf_counter();forward();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    params=sum(m.parameter_count for m,_ in models)
    return dict(batch_size=1,window_frames=30,models=len(models),parameters=params,repeats=50,median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),includes_pose_extraction=False)


def evaluate():
    assert not (OUT/'results.json').exists(),'Already scored; do not iteratively tune on these outcomes.'
    training=json.loads((OUT/'training_summary.json').read_text());assert training['complete']
    selection=json.loads((OUT/'selection.json').read_text());plan=json.loads((OUT/'protocol.json').read_text())
    for f,digest in plan['code_sha256'].items():assert sha(ROOT/'scripts'/f)==digest
    torch.set_num_threads(2);base=load_dataset(BASE)
    splits={name:load_split(base,name) for name in ['original','fold_1','fold_2','fold_3','fold_4']}
    policies={}
    for r in training['runs']:
        d=splits[r['split']];a=np.load(OUT/'runs'/r['id']/'validation_predictions.npz');ids=a['indices']
        policies[r['id']]=policy(d['labels'][ids],a['probabilities'],d['label_mask'][ids],[d['windows'][i] for i in ids])
    write(OUT/'evaluation_protocol.json',dict(created_at=now(),selection_sha256=sha(OUT/'selection.json'),training_summary_sha256=sha(OUT/'training_summary.json'),code_sha256=sha(__file__),validation_policies=policies,test_tuning=False))
    reports=[];predictions={}
    for r in training['runs']:
        d=splits[r['split']];ids=split_indices(d,2,True);ds=window_dataset(d,ids)
        cp=torch.load(OUT/'runs'/r['id']/'best.pt',map_location='cpu',weights_only=False)
        model=create_model(r['condition']).cuda();model.load_state_dict(cp['model_state']);reverse=r['condition']=='geometry_reverse'
        p=predict_dataset(model,ds,reverse);predictions[r['id']]=p
        np.savez_compressed(OUT/'runs'/r['id']/'test_predictions.npz',indices=ids,probabilities=p)
        metrics=score(d,ids,p,policies[r['id']]['threshold'])
        report=dict(**r,**metrics);reports.append(report)
        write(OUT/'runs'/r['id']/'evaluation.json',report)
        print(json.dumps(dict(id=r['id'],AP=metrics['all_test']['macro_ap'],quality_AP=metrics['quality']['macro_ap'])),flush=True)
        del model;torch.cuda.empty_cache()
    d=splits['original'];ids=split_indices(d,2,True)
    cached=np.load(WEB/'evaluation_arrays.npz');assert np.array_equal(cached['indices'],ids)
    assert np.array_equal(cached['y'],d['labels'][ids]) and np.array_equal(cached['mask'],d['label_mask'][ids])
    assert np.array_equal(cached['quality_indices'],ids[quality_mask(d)[ids]&d['label_mask'][ids].any(1)])
    old=cached['probabilities'];ensembles=[];ep={}
    for candidate in selection['candidates']:
        p=np.mean([predictions[m] for m in candidate['members']],0);ep[candidate['condition']]=p
        np.savez_compressed(OUT/(candidate['condition']+'_ensemble_test.npz'),indices=ids,probabilities=p)
        ensembles.append(dict(**candidate,**score(d,ids,p,candidate['alert_policy']['threshold'])))
    old_model=json.loads((WEB/'results.json').read_text())
    old_val=np.load(ROOT/'results/gait_review_upgrade_20261115/validation_predictions.npz')
    vi=old_val['indices'];vp=old_val['probabilities']
    baseline=dict(id=old_model['model'],validation=classification(d['labels'][vi],vp,d['label_mask'][vi],[d['windows'][i] for i in vi]),**score(d,ids,old,.8))
    assert abs(baseline['all_test']['macro_ap']-old_model['all_test']['macro_ap'])<1e-10
    chosen=selection['selected_condition'];chosen_p=ep[chosen]
    points=[]
    for k,i in enumerate(ids):
        w=d['windows'][i]
        for j in np.flatnonzero(d['label_mask'][i]):
            points.append(dict(clip=w['clip_id'],person=w['subject'],label=d['protocol']['class_order'][j],target=int(d['labels'][i,j]),start_s=w['start_time_s'],baseline=float(old[k,j]),selected=float(chosen_p[k,j])))
    write(OUT/'decision_points.json',points)
    selected_members=[r for r in training['runs'] if r['id'] in next(c for c in selection['candidates'] if c['condition']==chosen)['members']]
    bench=latency(selected_members,d,int(ids[0]));write(OUT/'latency.json',bench)
    # Framing invariance is a separate synthetic stress check, not a new test subset.
    stress=[]
    for condition in CONDITIONS:
        r=next(r for r in training['runs'] if r['split']=='original' and r['condition']==condition)
        model=create_model(condition).cuda();model.load_state_dict(torch.load(OUT/'runs'/r['id']/'best.pt',map_location='cpu',weights_only=False)['model_state'])
        ds=window_dataset(d,ids);ds.pose=ds.pose.clone();ds.pose[...,:2]=torch.where(ds.pose[...,3,None].bool(),ds.pose[...,:2]*1.2+torch.tensor([.1,-.1]),ds.pose[...,:2])
        changed=predict_dataset(model,ds,condition=='geometry_reverse');p=predictions[r['id']];mask=d['label_mask'][ids]
        stress.append(dict(condition=condition,mean_score_change=float(abs(changed-p)[mask].mean()),max_score_change=float(abs(changed-p)[mask].max()),decision_flip_fraction=float(((changed>=.5)!=(p>=.5))[mask].mean())))
    result=dict(created_at=now(),status='complete',selected_condition=chosen,selection=selection,baseline=baseline,ensembles=ensembles,runs=reports,
        by_clip=grouped(d,ids,chosen_p,old,'clip_id'),by_person=grouped(d,ids,chosen_p,old,'subject'),framing_stress=stress,latency=bench,
        protocol_sha256=sha(OUT/'protocol.json'),evaluation_protocol_sha256=sha(OUT/'evaluation_protocol.json'),clinical_validation=False,
        interpretation='Agreement with automatic pose-derived labels on reused internal held-out people. Class support varies across folds. No astronaut classifier or clinical validation.',
        existing_demo_and_paper_metrics_changed=False)
    write(OUT/'results.json',result);write(OUT/'progress.json',dict(status='evaluation_complete',updated_at=now(),selected_condition=chosen))
    print('SELECTED '+chosen+' '+json.dumps(next(e['all_test'] for e in ensembles if e['condition']==chosen)),flush=True)

if __name__=='__main__':evaluate()

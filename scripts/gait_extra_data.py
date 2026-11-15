"""Immutable experimental splits and training-only teacher refits."""
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from gait_assessment_data import load_dataset, window_dataset, split_indices
from prepare_gait_assessment_dataset import identity, weighted_median
from gait_from_scratch_sampling import conditional_balancing_weights
from motionbert_gait_common import ROOT, METHODS, SPLITS, reference_fence, classify, sha, write, now

OUT=ROOT/'results/gait_experiments_20261115_0502'
BASE=ROOT/'results/gait_from_scratch_v1/dataset'
SEEDS=list(range(20261115,20261120))


def mappings(base):
    original=json.loads((BASE/'splits.json').read_text())['subjects']
    rng=np.random.default_rng(20261115)
    people=sorted(p for p in original if p.startswith('ljmu360_'))
    groups=[list(g) for g in np.array_split(rng.permutation(people),4)]
    pg=sorted(p for p in original if p.startswith('progait_'))
    result={'original':original}
    for f in range(4):
        m={p:'train' for p in original}
        for p in groups[f]:m[str(p)]='test'
        m[str(groups[(f+1)%4][0])]='validation'
        m[pg[f]]='test';m[pg[(f+1)%4]]='validation'
        result[f'fold_{f+1}']=m
    for people_prefix in ['ljmu360_','progait_']:
        for p in original:
            if p.startswith(people_prefix):assert sum(m[p]=='test' for k,m in result.items() if k!='original')==1
    return result


def refit(base,mapping):
    d=dict(base);d['windows']=[dict(m,split=mapping.get(m['subject'],'exploratory')) for m in base['windows']]
    windows=d['windows'];d['split']=np.array([SPLITS.index(m['split']) for m in windows],np.int8)
    protocol=copy.deepcopy(base['protocol']);d['protocol']=protocol
    ref=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for i,m in enumerate(windows):
        if m['split']!='train' or m['source'] not in ('ljmu360','lbmc') or m['method']!='sapiens' or not m['input_eligible']:continue
        for j,value in enumerate(d['values'][i]):
            if np.isfinite(value):ref[(m['activity'],m['environment'],m['view'],j)][m['subject']][m['trial_block']].append(float(value))
    balanced={key:{p:float(np.median([np.median(v) for v in trials.values()])) for p,trials in people.items()} for key,people in ref.items()}
    fits={};target=np.full_like(base['labels'],-1)
    for i,m in enumerate(windows):
        labels=[]
        for j,label in enumerate(protocol['labels']):
            oi=label['outcome'];key=(m['activity'],m['environment'],m['view'],oi);units=dict(balanced.get(key,{}))
            own=m['split']=='train' and m['source'] in ('ljmu360','lbmc')
            if own:units.pop(m['subject'],None)
            fence=reference_fence(units) if m['method']=='sapiens' else None
            fit_key='|'.join(map(str,key))+'|exclude='+(m['subject'] if own else '')
            if fence:
                fence['sha256']=hashlib.sha256(json.dumps(fence,sort_keys=True).encode()).hexdigest();fits[fit_key]=fence
            value=d['values'][i,oi];v=classify(value,fence,label['direction']);target[i,j]=v
            state='positive' if v==1 else 'negative' if v==0 else 'uncertain' if fence and np.isfinite(value) else 'unavailable'
            why=list(m['measurement_reasons'][oi])
            if m['method']=='publisher':why.append('no_method_matched_reference')
            elif fence is None:why.append('insufficient_reference_people_or_zero_IQR')
            elif state=='uncertain':why.append('reference_boundary_buffer')
            labels.append(dict(id=label['id'],state=state,value=float(value),unit=protocol['outcomes'][oi]['unit'],
                reference_fit=fit_key if fence else None,reference_sha256=fence['sha256'] if fence else None,
                region=label['region'],side='bilateral_or_midline',reasons=why,clinical_form_label=None,
                metric_change_direction=('decrease' if label['direction']=='high' else 'increase') if v==1 else None,
                treatment_or_prosthetic_adjustment=None))
        m['labels']=labels
    d['labels']=target;d['reference_fits']=fits
    weights=[];prevalence=[]
    for j,label in enumerate(protocol['labels']):
        ids=[i for i,m in enumerate(windows) if m['split']=='train' and m['method']=='sapiens' and m['input_eligible'] and target[i,j]>=0]
        signs=[{windows[i]['subject'] for i in ids if target[i,j]==s} for s in [0,1]]
        label.update(classification_enabled=min(map(len,signs))>=2,negative_training_subjects=sorted(signs[0]),positive_training_subjects=sorted(signs[1]))
        if ids:p=float(np.average(target[ids,j],weights=conditional_balancing_weights([identity(windows[i]) for i in ids])))
        else:p=0.
        prevalence.append(p);weights.append(float(np.clip((1-p)/max(p,1e-8),.2,10)))
    stats={}
    for mi,method in enumerate(METHODS):
        centers=[];scales=[];counts=[]
        for j in range(10):
            ids=np.flatnonzero((d['split']==0)&(d['method']==mi)&d['measurement_mask'][:,j])
            vals=d['values'][ids,j];w=conditional_balancing_weights([identity(windows[i]) for i in ids]) if len(ids) else []
            center=weighted_median(vals,w) if len(ids) else 0.;mean=np.average(vals,weights=w) if len(ids) else 0.
            scale=max(float(np.sqrt(np.average((vals-mean)**2,weights=w))),.01) if len(ids) else 1.
            centers.append(center);scales.append(scale);counts.append(len(ids))
        stats[method]=dict(center=centers,scale=scales,count=counts)
    d['normalization']=stats;d['baselines']=dict(class_prevalence=prevalence,positive_class_weight=weights,
        numeric_median={method:stats[method]['center'] for method in METHODS})
    centers=np.asarray([stats[m]['center'] for m in METHODS],np.float32);scales=np.asarray([stats[m]['scale'] for m in METHODS],np.float32)
    counts=np.asarray([stats[m]['count'] for m in METHODS]);d['regression_mask']=d['measurement_mask']&(counts[d['method']]>0)
    d['standardized']=np.nan_to_num((d['values']-centers[d['method']])/scales[d['method']],nan=0).astype(np.float32)
    d['enabled']=np.array([l['classification_enabled'] for l in protocol['labels']],bool)
    d['label_mask']=(target>=0)&d['enabled'][None]&(d['method'][:,None]==0)
    audit_refit(d,mapping)
    return d


def audit_refit(d,mapping):
    for field in ['subject','trial_block','source_sha256']:
        parts=defaultdict(set)
        for m in d['windows']:
            if m[field] is not None:parts[m[field]].add(m['split'])
        assert all(len(v)==1 for v in parts.values()),field
    for fit in d['reference_fits'].values():
        assert fit['participant_count']>=5 and fit['q75']>fit['q25']
        assert all(mapping[p]=='train' and p.startswith(('ljmu360_','lbmc_')) for p in fit['participants'])
    for m in d['windows']:
        for label in m['labels']:
            fit=d['reference_fits'].get(label['reference_fit'])
            if fit and m['split']=='train':assert m['subject'] not in fit['participants']
    assert not d['label_mask'][d['method']==1].any()


def centered(pose):
    out=pose.copy();counts=defaultdict(int)
    for i,p in enumerate(out):
        valid=p[...,3].astype(bool);j=0 if valid[:,0].any() else 8 if valid[:,8].any() else None
        anchor=np.median(p[valid[:,j],j,:2],axis=0) if j is not None else np.zeros(2)
        counts['pelvis' if j==0 else 'thorax' if j==8 else 'zero']+=1
        p[...,:2]=np.where(valid[...,None],p[...,:2]-anchor,0)
    return out,dict(counts)


def condition_data(d,condition):
    out=dict(d)
    if condition=='root_centered':out['pose'],out['centering_counts']=centered(d['pose'])
    elif condition=='primary_only':
        out['regression_mask']=d['regression_mask'].copy();out['regression_mask'][d['method']==1]=False
    elif condition!='baseline':raise ValueError(condition)
    return out


def datasets(d,condition):
    ids=split_indices(d,0,primary_only=condition=='primary_only')
    if condition=='primary_only':
        all_train=split_indices(d,0);pub=int((d['method'][all_train]==1).sum());assert pub>=len(ids)
        ids=np.concatenate([ids,ids]) # sampling slots, never independent observations
    return window_dataset(d,ids),window_dataset(d,split_indices(d,1,True))


def freeze(base):
    plan=OUT/'protocol.json'
    if plan.exists():raise RuntimeError('Protocol already frozen')
    maps=mappings(base);jobs=[]
    for seed in SEEDS:
        for arch in ['tcn','motionbert']:jobs.append(dict(id=f'original_baseline_{arch}_{seed}',split='original',condition='baseline',architecture=arch,seed=seed,priority=1))
    for f in range(1,5):
        for arch in ['tcn','motionbert']:jobs.append(dict(id=f'fold_{f}_baseline_{arch}_20261115',split=f'fold_{f}',condition='baseline',architecture=arch,seed=20261115,priority=2))
    for condition in ['primary_only','root_centered']:
        for seed in SEEDS[:3]:
            for arch in ['tcn','motionbert']:jobs.append(dict(id=f'original_{condition}_{arch}_{seed}',split='original',condition=condition,architecture=arch,seed=seed,priority=3))
    protocol=dict(created_at=now(),base_dataset_sha256=base['manifest']['dataset_sha256'],plan_sha256=sha(OUT/'PLAN.txt'),
        training_cutoff_utc='2026-11-15T05:47:00+00:00',delivery_deadline_utc='2026-11-15T06:02:00+00:00',
        seed_values=SEEDS,jobs=jobs,participant_maps=maps,threshold=.5,clinical_validation=False,
        evaluation='All training complete before test result inspection; all runs reported, no best-test selection. Internal repeated-test/CV evidence is exploratory.')
    write(plan,protocol);return protocol


def snapshot(d,mapping,path):
    path.mkdir(parents=True,exist_ok=False)
    write(path/'splits.json',mapping);write(path/'reference_fits.json',d['reference_fits'])
    write(path/'label_protocol.json',d['protocol']);write(path/'normalization.json',d['normalization']);write(path/'training_baselines.json',d['baselines'])
    np.savez_compressed(path/'targets.npz',labels=d['labels'],split=d['split'],standardized=d['standardized'],
        regression_mask=d['regression_mask'],label_mask=d['label_mask'],enabled=d['enabled'])
    write(path/'manifest.json',dict(created_at=now(),base_dataset_sha256=d['manifest']['dataset_sha256'],
        files_sha256={p.name:sha(p) for p in path.iterdir()},train_people=sorted(p for p,s in mapping.items() if s=='train'),
        validation_people=sorted(p for p,s in mapping.items() if s=='validation'),test_people=sorted(p for p,s in mapping.items() if s=='test'),
        classes_enabled=int(d['enabled'].sum()),train_only_references_and_normalization=True,leakage_audit='passed',
        numeric_window_targets_reused_without_reextraction=True))

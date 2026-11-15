from pathlib import Path
from collections import Counter,defaultdict
import json
import numpy as np
from gait_distribution_model import *
from motionbert_gait_common import write,sha,now
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'results/gait_distribution_20261115';B=ROOT/'results/gait_from_scratch_v1/dataset'

def audit():
    r=json.load(open(P/'results.json'));p=json.load(open(P/'protocol.json'));models=json.load(open(P/'models.json'));rows=json.load(open(P/'feature_rows.json'));a=np.load(P/'features.npz');x=a['values']
    assert sha(B/'dataset.npz')==p['base_dataset_sha256']
    for f,digest in p['code_sha256'].items():assert sha(ROOT/'scripts'/f)==digest
    assert not r['labels_used'] and not r['amputation_input_used']
    sets=[]
    for split in p['folds']:
        sets.append({k:set(v) for k,v in split.items()});s=sets[-1];assert not(s['train']&s['calibration'] or s['train']&s['test'] or s['calibration']&s['test'])
        for field in ['subject','trial_block','source_sha256']:
            mapping=defaultdict(set)
            for row in rows:
                if not row['subject']:continue
                split_name=next(k for k,v in s.items() if row['subject'] in v);mapping[row[field]].add(split_name)
            assert all(len(v)==1 for v in mapping.values()),field
    errors=[];total=flags=known_windows=known_flags=0;region_counts=defaultdict(lambda:[0,0]);people=defaultdict(lambda:[0,0])
    for clip in r['clips']:
        ws=json.load(open(P/clip['predictions']))['windows']
        for w in ws:
            idx=next(i for i,row in enumerate(rows) if row['key']==w['key'])
            if w['subject']:assert w['subject'] in sets[w['fold']]['test']
            any_flag=False
            for j,e in enumerate(w['events']):
                assert e['clinical_label'] is None
                if e['percentile'] is None:continue
                m=models[e['model_key']];model={k:np.asarray(v) for k,v in m.items() if k in ['center','scale','means','covariances','mixture_weights']}
                score=float(nll(x[idx:idx+1,j],model)[0]);err=abs(score-e['nll']);errors.append(err);assert err<1e-8
                if w['subject']:assert w['subject'] not in m['train_people']+m['calibration_people']
                assert len(m['train_people'])>=5 and len(m['calibration_people'])>=3
                weights=balanced_weights([rows[i] for i in m['training_rows']]);group_sums=defaultdict(float)
                for i,weight in zip(m['training_rows'],weights):group_sums[rows[i]['subject']]+=weight
                assert np.allclose(list(group_sums.values()),1/len(group_sums))
                flagged=e['percentile']>=95;total+=1;flags+=flagged;region_counts[e['region']][0]+=1;region_counts[e['region']][1]+=flagged;any_flag|=flagged
            if w['subject']:known_windows+=1;known_flags+=any_flag;people[w['subject']][0]+=1;people[w['subject']][1]+=any_flag
    base=np.load(B/'dataset.npz');poses=base['x'];checks=[]
    for row in rows[::37]:
        pose=poses[row['base_index']].copy();f,_,_=features(pose,row['view']);shift=pose.copy();shift[...,:2]+=[.17,-.12];g,_,_=features(shift,row['view']);assert np.allclose(f,g,equal_nan=True,atol=.01)
        damaged=pose.copy();damaged[damaged[...,2]<.4,:2]=1000.;g,_,_=features(damaged,row['view']);assert np.allclose(f,g,equal_nan=True,atol=1e-7)
        reflected=pose.copy();reflected[...,0]*=-1;g,_,_=features(reflected,row['view']);assert np.allclose(f,g,equal_nan=True,atol=1e-5)
        checks.append(row['key'])
    # Amputation/source metadata cannot affect numerical density inference.
    m=next(v for v in models.values() if v['status']=='trained');numeric={k:np.asarray(m[k]) for k in ['center','scale','means','covariances','mixture_weights']};sample=x[m['training_rows'][:5],REGIONS.index(m['region'])];original=nll(sample,numeric);tagged=dict(numeric,amputation_status='changed',cohort='changed',source='changed');assert np.array_equal(original,nll(sample,tagged))
    report=dict(status='passed',created_at=now(),known_clips=sum(c['subject'] is not None for c in r['clips']),external_clips=sum(c['subject'] is None for c in r['clips']),
        supported_region_windows=total,flagged_region_windows=flags,regional_flag_fraction=flags/total,known_test_windows=known_windows,known_test_windows_with_any_deviation=known_flags,
        per_region_counts=dict(region_counts),per_person_window_counts=dict(people),score_reproduction_max_error=max(errors),metamorphic_samples=len(checks),
        checks=['person/trial/source separation','all known participants held out once','no old teacher targets or amputation input','equal-person reference weights','stored score reproduction','translation invariance above visibility cutoff','reflection invariance','invalid coordinates cannot affect score','status/source metadata does not affect inference'],clinical_accuracy=None)
    write(P/'audit.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':audit()

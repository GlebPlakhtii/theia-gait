"""Replay the frozen pooling ensemble and replace the supervised demo scores."""
from pathlib import Path
import json,shutil,hashlib
from datetime import datetime,timezone
import numpy as np
import torch
import run_gait_improvement as core
from gait_statistical_pool_model import create_refined_model
from evaluate_gait_selected import evaluate

ROOT=Path(__file__).resolve().parents[1];WEB=ROOT/'results/gait_selected_paper_20261115'
STAGE=ROOT/'results/gait_improvement_refinement_20261115'
AUDIT=ROOT/'audit/pooling_demo_update_20261115'
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2)
    AUDIT.mkdir(parents=True,exist_ok=True)
    for name in ['page_data.json','app.js','index.html','README.txt']:
        dest=AUDIT/('before_'+name)
        if not dest.exists():shutil.copy2(WEB/name,dest)
    page=json.loads((AUDIT/'before_page_data.json').read_text())
    results=json.loads((STAGE/'results.json').read_text());selected=next(e for e in results['ensembles'] if e['condition']=='geometry_set')
    d=core.load_split(core.load_dataset(core.BASE),'original');ids=core.split_indices(d,2,True);ds=core.window_dataset(d,ids)
    predictions=[];checkpoints=[]
    for member in selected['members']:
        path=STAGE/'runs'/member/'best.pt';cp=torch.load(path,map_location='cpu',weights_only=False)
        model=create_refined_model('geometry_set').cuda();model.load_state_dict(cp['model_state'])
        predictions.append(core.predict_dataset(model,ds));checkpoints.append(dict(member=member,sha256=sha(path),parameters=model.parameter_count))
        del model
    p=np.mean(predictions,axis=0);stored=np.load(STAGE/'geometry_set_ensemble_test.npz')
    assert np.array_equal(stored['indices'],ids);np.testing.assert_allclose(p,stored['probabilities'],rtol=1e-6,atol=1e-6)
    old=np.load(WEB/'evaluation_arrays.npz');assert np.array_equal(ids,old['indices'])
    names=d['protocol']['class_order'];lookup={int(i):k for k,i in enumerate(ids)};oldrows={w['index']:w for w in page['rows']}
    updated=[];changed=0
    for i in ids:
        w=oldrows[int(i)];k=lookup[int(i)]
        for l in w['labels']:
            j=names.index(l['id']);assert d['label_mask'][i,j] and l['target']==d['labels'][i,j]
            score=float(p[k,j]);changed+=int(abs(score-l['score'])>1e-7)
            l.update(score=score,prediction=int(score>=.5),correct=bool((score>=.5)==l['target']))
        w['prediction_model']='geometry_set_ensemble';updated.append(w)
    rowmap={w['index']:w for w in updated};nclips=0
    for c in page['clips']:
        if c['kind']!='tcn':continue
        c['windows']=[rowmap[w['index']] for w in c['windows']]
        ix=[lookup[i] for i in c['indices']]
        c.update(evaluate(old['y'],p,old['mask'],ix,updated,names));c['window_count']=c.pop('windows')
        c['windows']=[w for w in updated if w['clip_id']==c['clip_id']]
        c['kind']='pooling';c['id']='pooling:'+c['clip_id'];c['model']='geometry_set three-seed ensemble';nclips+=1
    page['rows']=updated
    page['baseline_results']=page['results']
    metadata=dict(name='Statistical-pooling ensemble',id='geometry_set_ensemble',threshold=.5,checkpoints=checkpoints,
        selection='Highest observed internal-test macro AP; not the validation-selected family. Promoted to this experimental demo at user request.',
        test=selected['all_test'],quality=selected['quality'],clinical_validation=False,
        replay='Precomputed frozen ensemble predictions; no browser inference or new pose extraction.',
        legacy_collections='Four density-model cases remain a separate model; ISS examples remain measurements or explicitly simulated alerts.')
    verification=dict(at=datetime.now(timezone.utc).isoformat(),windows=len(ids),supervised_clips=nclips,changed_scores=changed,
        max_probability_difference=float(abs(p-stored['probabilities']).max()),checkpoints=checkpoints,
        source_predictions_sha256=sha(STAGE/'geometry_set_ensemble_test.npz'),all_inputs_and_frozen_split_hashes_verified=True,
        quality_gate_unchanged=True,targets_unchanged=True,threshold=.5,selected_examples='Same historical five clips; no new outcome-based clip selection.')
    assert nclips==14 and changed>0
    page['active_model']=metadata;page['inference']=verification
    # The earlier result object remains explicitly archived for the historical search table.
    write(WEB/'page_data.json',page);write(WEB/'pooling_window_results.json',updated)
    write(WEB/'pooling_model.json',metadata);write(WEB/'pooling_demo_verification.json',verification)
    np.savez_compressed(WEB/'pooling_predictions.npz',indices=ids,probabilities=p)
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()

"""Separately frozen follow-up after the first exploratory comparison.

Motivation: premature early stopping across geometric seeds and a mismatch
between temporal convolutions and order-invariant teacher measurements.
Existing internal test data were already inspected in stage one; results here
remain exploratory. All follow-up choices still use validation predictions.
"""
import argparse,copy,json
import numpy as np
import torch
import run_gait_improvement as core
from gait_statistical_pool_model import create_refined_model

OUT=core.ROOT/'results/gait_improvement_refinement_20261115'
CONDITIONS=('geometry_slow','geometry_set')

def setup():
    core.OUT=OUT
    core.create_model=create_refined_model

def train():
    setup();torch.set_num_threads(2)
    old=core.ROOT/'results/gait_improvement_20261115/protocol.json'
    plan=copy.deepcopy(json.loads(old.read_text()))
    plan.update(created_at=core.now(),stage='follow-up exploratory refinement',previous_protocol_sha256=core.sha(old),
        prior_test_inspection=True,reason='Validation geometry runs often stopped near initialization. Add slower training and a smaller order-invariant frame encoder; report every result.',
        model_inputs='Same 2D joints/confidence/masks and declared context. Statistical model receives per-frame geometry without direction differences. Learned per-frame embeddings are pooled by mean, standard deviation, minimum and maximum; teacher window metrics and thresholds are excluded.',
        reverse='No temporal reversal augmentation or test-time reversal in this stage.',
        code_sha256={f:core.sha(core.ROOT/'scripts'/f) for f in ['gait_kinematic_model.py','gait_statistical_pool_model.py','run_gait_improvement.py','run_gait_improvement_refinement.py']})
    plan['training'].update(lr=.0003,weight_decay=.01,dropout=.2,maximum_epochs=200,minimum_epochs=60,patience=40)
    plan['jobs']=[dict(id=f'original_{c}_{s}',split='original',condition=c,seed=s) for c in CONDITIONS for s in core.SEEDS]
    plan['jobs'] += [dict(id=f'fold_{f}_{c}_{core.SEEDS[0]}',split=f'fold_{f}',condition=c,seed=core.SEEDS[0]) for f in range(1,5) for c in CONDITIONS]
    plan['evaluation']='All 14 follow-up runs and both ensembles reported. Reuse frozen original and four participant-separated folds. Exploratory teacher agreement; no independent clinical reference.'
    if (OUT/'protocol.json').exists():
        prior=json.loads((OUT/'protocol.json').read_text());assert prior['code_sha256']==plan['code_sha256'];plan=prior
    else:core.write(OUT/'protocol.json',plan)
    assert not (OUT/'results.json').exists()
    base=core.load_dataset(core.BASE);splits={s:core.load_split(base,s) for s in ['original','fold_1','fold_2','fold_3','fold_4']}
    runs=[]
    for job in plan['jobs']:
        path=OUT/'runs'/job['id']/'training.json'
        if path.exists():r=json.loads(path.read_text())
        else:
            core.write(OUT/'progress.json',dict(status='training',updated_at=core.now(),current=job,completed=len(runs),scheduled=len(plan['jobs'])))
            r=core.train_one(splits[job['split']],job,plan)
        runs.append(r);core.write(OUT/'training_summary.json',dict(runs=runs,complete=False))
        print(json.dumps({k:r[k] for k in ['id','best_validation_ap','best_epoch','seconds']}),flush=True)
    d=splits['original'];vi=core.split_indices(d,1,True);rows=[d['windows'][i] for i in vi];candidates=[]
    for condition in CONDITIONS:
        members=[r for r in runs if r['split']=='original' and r['condition']==condition]
        a=[np.load(OUT/'runs'/r['id']/'validation_predictions.npz') for r in members]
        assert all(np.array_equal(x['indices'],vi) for x in a)
        p=np.mean([x['probabilities'] for x in a],0)
        np.savez_compressed(OUT/f'{condition}_ensemble_validation.npz',indices=vi,probabilities=p)
        candidates.append(dict(condition=condition,members=[r['id'] for r in members],validation=core.classification(d['labels'][vi],p,d['label_mask'][vi],rows),alert_policy=core.policy(d['labels'][vi],p,d['label_mask'][vi],rows),passes=3))
    chosen=max(candidates,key=lambda c:c['validation']['macro_ap'])
    core.write(OUT/'selection.json',dict(created_at=core.now(),selected_condition=chosen['condition'],candidates=candidates,selection_used_test=False,prior_internal_test_already_examined=True))
    core.write(OUT/'training_summary.json',dict(runs=runs,complete=True,completed_at=core.now(),test_scoring_started=False))
    core.write(OUT/'progress.json',dict(status='training_complete',updated_at=core.now(),completed=len(runs),selected_condition=chosen['condition']))
    print('TRAINING COMPLETE '+chosen['condition'],flush=True)

def evaluate():
    setup()
    import evaluate_gait_improvement as evaluator
    evaluator.OUT=OUT;evaluator.CONDITIONS=CONDITIONS;evaluator.create_model=create_refined_model
    evaluator.evaluate()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','evaluate']);args=p.parse_args()
    train() if args.phase=='train' else evaluate()

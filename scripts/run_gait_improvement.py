"""Frozen participant-separated comparison of raw and geometric sequence models.

Training/validation complete and ensemble choice is persisted before the test
scoring command may run. Existing test data are reused internal evidence.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,time
import numpy as np
import torch

from gait_assessment_data import load_dataset,split_indices,window_dataset
from gait_from_scratch_training import audit_inputs,selection_basis,state_digest,TrainingConfig,validate_model
from gait_from_scratch_sampling import sample_training_epoch
from gait_from_scratch_model import TCNConfig,masked_multitask_loss,CLASSES
from gait_from_scratch_evaluation import binary_metrics
from gait_kinematic_model import create_model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gait_improvement_20261115'
BASE=ROOT/'results/gait_from_scratch_v1/dataset'
PREVIOUS=ROOT/'results/gait_experiments_20261115_0502'
WEB=ROOT/'results/gait_selected_paper_20261115'
CONDITIONS=('raw','geometry','geometry_reverse')
SEEDS=(20261120,20261121,20261122)

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def load_split(base,name):
    folder=PREVIOUS/'splits'/name;manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['base_dataset_sha256']==base['manifest']['dataset_sha256']
    for f,digest in manifest['files_sha256'].items():assert sha(folder/f)==digest,(name,f)
    d=dict(base)
    with np.load(folder/'targets.npz') as a:
        for key in a.files:d[key]=a[key].copy()
    mapping=json.loads((folder/'splits.json').read_text())
    d['windows']=[dict(w,split=mapping.get(w['subject'],'exploratory')) for w in base['windows']]
    for key,file in [('protocol','label_protocol.json'),('normalization','normalization.json'),('baselines','training_baselines.json')]:
        d[key]=json.loads((folder/file).read_text())
    for field in ['subject','trial_block','source_sha256']:
        owners={}
        for w in d['windows']:
            if w[field] is not None:owners.setdefault(w[field],set()).add(w['split'])
        assert all(len(x)==1 for x in owners.values()),field
    d['split_provenance']=manifest
    return d


def freeze(base):
    jobs=[dict(id=f'original_{c}_{s}',split='original',condition=c,seed=s) for c in CONDITIONS for s in SEEDS]
    jobs += [dict(id=f'fold_{f}_{c}_{SEEDS[0]}',split=f'fold_{f}',condition=c,seed=SEEDS[0]) for f in range(1,5) for c in CONDITIONS]
    plan=dict(created_at=now(),dataset_sha256=base['manifest']['dataset_sha256'],jobs=jobs,
        model_inputs='2D pose, confidence, validity and declared view/environment only. No group label, participant/source ID, target window ROM, target asymmetry, teacher thresholds or target flags.',
        geometry='Per-frame normalized joint offsets, bone directions/lengths, bone-direction differences, joint sine/cosine and pelvis/shoulder/trunk directions. Translation and positive isotropic-scale invariant; no camera rotation/depth correction.',
        reverse='geometry_reverse reverses 50% of training windows; inference averages forward/reversed outputs. Targets are order-invariant window statistics; no timing or amplitude rescaling.',
        supervision='Unchanged pose-derived engineering teacher labels; no independent clinical labels.',
        training=dict(maximum_epochs=100,minimum_epochs=12,patience=12,seconds_per_run=90,batch_size=64,lr=.001,weight_decay=.003,dropout=.15,gradient_clip=1.,primary_only=True,sampling='Equal person then trial; 2N primary draws per epoch, matching former primary-only run slots.'),
        selection='Best checkpoint per run by full validation macro AP; average three seeds within each original-split condition; select ensemble by validation macro AP, ties prefer fewer inference passes. Persist before test scoring.',
        thresholds='Report fixed 0.5 and a global threshold maximizing validation participant-balanced F0.5 at recall >=0.5. No test threshold tuning.',
        evaluation='All 21 new runs and all three original ensembles reported. Same original test and quality screen, plus four existing participant-separated folds with training-only references. Reused internal test/CV, not external validation.',
        seed_values=list(SEEDS),clinical_validation=False,old_demo_checkpoint=json.loads((WEB/'results.json').read_text())['model'],
        code_sha256={n:sha(ROOT/'scripts'/n) for n in ['gait_kinematic_model.py','run_gait_improvement.py']},
        research_references=[{'url':'https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html','role':'Bone and joint feature motivation; current model is a TCN, not a reproduction of 2s-AGCN.'},{'url':'https://arxiv.org/abs/1803.01271','role':'Temporal convolution architecture motivation; no clinical performance claim.'}])
    path=OUT/'protocol.json'
    if path.exists():
        prior=json.loads(path.read_text());assert prior['code_sha256']==plan['code_sha256'];return prior
    write(path,plan);return plan


def predict_dataset(model,dataset,reverse=False):
    model.eval();ps=[]
    with torch.inference_mode():
        for start in range(0,len(dataset),64):
            pose=dataset.pose[start:start+64].cuda();context=dataset.context[start:start+64].cuda()
            out=model(pose,context)['class_logits'].sigmoid()
            if reverse:out=(out+model(pose.flip(1),context)['class_logits'].sigmoid())/2
            ps.append(out.cpu().numpy())
    return np.concatenate(ps)


def classification(y,p,mask,rows,threshold=.5):
    classes=[]
    for j,name in enumerate(CLASSES):
        m=binary_metrics(y[:,j],p[:,j],mask[:,j])
        classes.append(dict(id=name,ap=m['average_precision'],positive=m['positive_windows'],negative=m['negative_windows']))
    yy=y[mask].astype(bool);pp=p[mask]>=threshold
    tp=int((yy&pp).sum());fp=int((~yy&pp).sum());fn=int((yy&~pp).sum());tn=int((~yy&~pp).sum())
    aps=[c['ap'] for c in classes if c['ap'] is not None]
    return dict(windows=len(rows),clips=len({w['clip_id'] for w in rows}),people=len({w['subject'] for w in rows}),
        decisions=len(yy),macro_ap=float(np.mean(aps)) if aps else None,ap_classes=len(aps),threshold=float(threshold),
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        accuracy=(tp+tn)/len(yy) if len(yy) else None,tp=tp,fp=fp,fn=fn,tn=tn,per_class=classes)


def policy(y,p,mask,rows):
    weights=np.zeros(len(rows))
    for who in {r['subject'] for r in rows}:
        ix=np.array([r['subject']==who for r in rows]);weights[ix]=1/max(1,mask[ix].sum())
    candidates=[];yy=y[mask].astype(bool);ww=np.broadcast_to(weights[:,None],y.shape)[mask]
    for t in np.r_[np.arange(.05,1,.05),.975,.99]:
        pp=p[mask]>=t;tp=ww[yy&pp].sum();fp=ww[~yy&pp].sum();fn=ww[yy&~pp].sum()
        recall=tp/(tp+fn) if tp+fn else 0.;f05=1.25*tp/(1.25*tp+.25*fn+fp) if tp+fp+fn else 0.
        candidates.append(dict(threshold=float(t),f05=float(f05),recall=float(recall)))
    eligible=[c for c in candidates if c['recall']>=.5]
    return max(eligible,key=lambda c:(c['f05'],c['recall'],-abs(c['threshold']-.5))) if eligible else dict(threshold=.5,f05=None,recall=None)


def train_one(d,job,plan):
    folder=OUT/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    ids=split_indices(d,0,True);train=window_dataset(d,np.r_[ids,ids]);vi=split_indices(d,1,True);val=window_dataset(d,vi)
    enabled=torch.from_numpy(d['enabled']);audit_inputs(train,val,enabled,TCNConfig());basis=selection_basis(val,enabled)
    cfg=plan['training'];seed=job['seed'];torch.manual_seed(seed);np.random.seed(seed)
    model=create_model(job['condition']).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initial_hash=state_digest(initial)
    torch.save(dict(model_state=initial,job=job),folder/'initial.pt')
    positive=torch.tensor(d['baselines']['positive_class_weight'],dtype=torch.float32,device='cuda')
    start=time.monotonic();history=[];best=-float('inf');stale=updates=0;stop='max_epochs'
    for epoch in range(1,cfg['maximum_epochs']+1):
        if time.monotonic()-start>=cfg['seconds_per_run']:stop='time_cap';break
        model.train();ix,_=sample_training_epoch(train.identities,epoch-1,seed);total=0.;batches=0
        for offset in range(0,len(ix),cfg['batch_size']):
            pose,context,targets,masks,method=train.batch(ix[offset:offset+cfg['batch_size']],torch.device('cuda'),enabled)
            if not any(m.any() for m in masks.values()):continue
            if job['condition']=='geometry_reverse':
                reverse=torch.rand(len(pose),device='cuda')<.5
                pose=torch.where(reverse[:,None,None,None],pose.flip(1),pose)
            optimizer.zero_grad(set_to_none=True)
            loss=masked_multitask_loss(model(pose,context),targets,masks,method,positive_class_weight=positive)['total']
            assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert float(norm)>0
            optimizer.step();updates+=1;total+=float(loss.detach());batches+=1
        probabilities=predict_dataset(model,val,job['condition']=='geometry_reverse')
        cm=classification(val.targets['classification'].numpy(),probabilities,val.masks['classification'].numpy(),[d['windows'][i] for i in vi])
        score=cm['macro_ap'];assert score is not None
        history.append(dict(epoch=epoch,loss=total/max(batches,1),validation_ap=score,elapsed_s=time.monotonic()-start))
        if score>best:
            best=score;stale=0
            torch.save(dict(model_state=model.state_dict(),job=job,epoch=epoch,validation_ap=score,protocol_sha256=sha(OUT/'protocol.json')),folder/'best.pt')
        else:stale+=1
        write(folder/'history.json',history)
        if epoch>=cfg['minimum_epochs'] and stale>=cfg['patience']:stop='early_stopping';break
    torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),epoch=epoch,job=job),folder/'last.pt')
    cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(cp['model_state'])
    assert state_digest(cp['model_state'])!=initial_hash
    vp=predict_dataset(model,val,job['condition']=='geometry_reverse')
    np.savez_compressed(folder/'validation_predictions.npz',indices=vi,probabilities=vp)
    result=dict(**job,status='trained',best_epoch=cp['epoch'],best_validation_ap=best,epochs=len(history),updates=updates,
        seconds=time.monotonic()-start,stop_reason=stop,parameters=model.parameter_count,checkpoint_sha256=sha(folder/'best.pt'),
        random_initialization=True,initial_state_sha256=initial_hash,training_people=d['split_provenance']['train_people'],validation_people=d['split_provenance']['validation_people'],test_used_for_training_or_selection=False)
    write(folder/'training.json',result)
    del model,optimizer;torch.cuda.empty_cache()
    return result


def train():
    torch.set_num_threads(2);base=load_dataset(BASE);plan=freeze(base)
    assert not (OUT/'results.json').exists(),'Evaluation already completed; use a new protocol for new experiments.'
    datasets={name:load_split(base,name) for name in ['original','fold_1','fold_2','fold_3','fold_4']}
    runs=[]
    for job in plan['jobs']:
        path=OUT/'runs'/job['id']/'training.json'
        if path.exists():result=json.loads(path.read_text())
        else:
            write(OUT/'progress.json',dict(status='training',updated_at=now(),current=job,completed=len(runs),scheduled=len(plan['jobs'])))
            result=train_one(datasets[job['split']],job,plan)
        runs.append(result);write(OUT/'training_summary.json',dict(runs=runs,complete=False))
        print(json.dumps({k:result[k] for k in ['id','best_validation_ap','best_epoch','seconds']}),flush=True)
    original=datasets['original'];vi=split_indices(original,1,True);y=original['labels'][vi];mask=original['label_mask'][vi];rows=[original['windows'][i] for i in vi]
    candidates=[]
    for condition in CONDITIONS:
        members=[j for j in runs if j['split']=='original' and j['condition']==condition]
        arrays=[]
        for j in members:
            a=np.load(OUT/'runs'/j['id']/'validation_predictions.npz');assert np.array_equal(vi,a['indices']);arrays.append(a['probabilities'])
        p=np.mean(arrays,0);score=classification(y,p,mask,rows)
        np.savez_compressed(OUT/f'{condition}_ensemble_validation.npz',indices=vi,probabilities=p)
        candidates.append(dict(condition=condition,members=[m['id'] for m in members],validation=score,alert_policy=policy(y,p,mask,rows),passes=len(members)*(2 if condition=='geometry_reverse' else 1)))
    chosen=max(candidates,key=lambda c:(c['validation']['macro_ap'],-c['passes']))
    write(OUT/'selection.json',dict(created_at=now(),selected_condition=chosen['condition'],candidates=candidates,selection_used_test=False))
    write(OUT/'training_summary.json',dict(runs=runs,complete=True,completed_at=now(),test_scoring_started=False))
    write(OUT/'progress.json',dict(status='training_complete',updated_at=now(),completed=len(runs),selected_condition=chosen['condition']))
    print('TRAINING COMPLETE; selected by validation: '+chosen['condition'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['train']);args=parser.parse_args();train()

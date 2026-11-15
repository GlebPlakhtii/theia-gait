"""Additional bounded runs; validation-only selection, explicit encoder audits."""
import json
from pathlib import Path
import time
from dataclasses import replace
import numpy as np
import torch
from gait_from_scratch_training import (TrainingConfig,train_model,audit_inputs,selection_basis,validate_model,state_digest)
from gait_from_scratch_model import TCNConfig,masked_multitask_loss
from gait_from_scratch_sampling import sample_training_epoch
from motionbert_assessment_model import MotionBertAssessment
from motionbert_gait_common import CHECKPOINT_SHA,now,sha,write


def train_tcn(train,validation,job,folder,enabled,positive,provenance,deadline):
    remaining=min(60.,deadline-time.time())
    if remaining<=0:raise TimeoutError('Session training cutoff')
    return train_model(train,validation,output_directory=folder,enabled_classes=enabled,
        positive_class_weight=positive,input_provenance=provenance,
        config=TrainingConfig(seed=job['seed'],time_cap_s=remaining),device='cuda')


def train_mb(train,validation,job,folder,enabled,positive,provenance,deadline):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    audit_inputs(train,validation,enabled,TCNConfig());basis=selection_basis(validation,enabled)
    seed=job['seed'];torch.manual_seed(seed);np.random.seed(seed)
    device=torch.device('cuda');model=MotionBertAssessment().to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    encoder_ids={id(p) for p in model.encoder.parameters()}
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=1e-5),
        dict(params=[p for p in model.parameters() if id(p) not in encoder_ids],lr=1e-4)],weight_decay=.01)
    started=time.monotonic();limit=started+min(240.,deadline-time.time())
    if limit<=started:raise TimeoutError('Session training cutoff')
    config=dict(seed=seed,maximum_epochs=20,minimum_encoder_epochs=3,patience=4,microbatch=4,accumulation=4,
        encoder_lr=1e-5,head_lr=1e-4,weight_decay=.01,gradient_clip=1.,warmup_epochs=1,selection=basis,
        time_cap_s=limit-started,parameter_count=model.parameter_count,initial_checkpoint_sha256=CHECKPOINT_SHA,
        input_provenance=provenance,initial_state_sha256=state_digest(initial),test_data_consumed=False,
        precision='BF16 autocast, FP32 weights',pilot='Reused original verified 300-second device pilot; same model/device/batch',
        initial_weight_storage='Official encoder checkpoint plus reproducible seeded heads; no duplicate full initial checkpoint')
    write(folder/'resolved_training.json',config)
    torch.save({k:v for k,v in initial.items() if not k.startswith('encoder.')},folder/'initial_heads.pt')
    positive=torch.as_tensor(positive,dtype=torch.float32,device=device)
    history=[];updates=encoder_updates=nonzero=0;best=best_epoch=None;stale=0;maxgrad=0.;stop='maximum_epochs'
    def save(name,epoch,validation_result=None,include_optimizer=False):
        state=dict(model_state=model.state_dict(),epoch=epoch,updates=updates,encoder_updates=encoder_updates,
            configuration=config,validation=validation_result,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state())
        if include_optimizer:state['optimizer_state']=optimizer.state_dict()
        path=folder/(name+'.pt');tmp=path.with_suffix('.pt.tmp');torch.save(state,tmp);tmp.replace(path)
    for epoch in range(1,21):
        if time.monotonic()>=limit:stop='time_cap';break
        warmup=epoch==1
        for p in model.encoder.parameters():p.requires_grad_(not warmup)
        model.train();indices,sampling=sample_training_epoch(train.identities,epoch-1,seed)
        totals={k:0. for k in ['classification','regression','evaluability','total']};batches=pending=0
        epoch_start=time.monotonic();optimizer.zero_grad(set_to_none=True)
        def step(n):
            nonlocal updates,encoder_updates,nonzero,maxgrad
            if n!=4:
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(4/n)
            if not warmup:
                norm=sum(float(p.grad.detach().float().square().sum()) for p in model.encoder.parameters() if p.grad is not None)
                nonzero+=int(norm>0)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);maxgrad=max(maxgrad,float(grad))
            optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;encoder_updates+=int(not warmup)
        for start in range(0,len(indices),4):
            if time.monotonic()>=limit:stop='time_cap';break
            pose,context,target,masks,method=train.batch(indices[start:start+4],device,enabled)
            if not any(m.any() for m in masks.values()):continue
            loss=masked_multitask_loss(model(pose,context),target,masks,method,positive_class_weight=positive)
            if not all(torch.isfinite(v) for v in loss.values()):raise FloatingPointError('Nonfinite loss')
            (loss['total']/4).backward();pending+=1;batches+=1
            for k in totals:totals[k]+=float(loss[k].detach())
            if pending==4:step(pending);pending=0
        if pending:step(pending)
        val=validate_model(model,validation,enabled,basis,replace(TrainingConfig(),batch_size=4),device,limit,time.monotonic) if time.monotonic()<limit else None
        row=dict(epoch=epoch,warmup=warmup,batches=batches,training_loss={k:v/batches if batches else None for k,v in totals.items()},
            validation=val,elapsed_s=time.monotonic()-started,epoch_seconds=time.monotonic()-epoch_start,sampling=sampling)
        history.append(row);write(folder/'training_history.json',history)
        if val is None:stop='time_cap';break
        score=val['selection_score']
        if not warmup:
            improved=best is None or (score>best if basis['direction']=='maximize' else score<best)
            if improved:best=score;best_epoch=epoch;stale=0;save('best',epoch,val)
            else:stale+=1
        write(folder/'progress.json',dict(updated_at=now(),epoch=epoch,encoder_updates=encoder_updates,
            validation_score=score,elapsed_s=row['elapsed_s']))
        if epoch>=4 and stale>=4:stop='early_stopping';break
    save('last',history[-1]['epoch'] if history else 0,history[-1]['validation'] if history else None,include_optimizer=True)
    if best_epoch is None:raise RuntimeError('No encoder checkpoint with complete validation')
    best_state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model_state']
    assert all(torch.isfinite(v).all() for v in best_state.values())
    changed=[k for k,v in best_state.items() if k.startswith('encoder.') and not torch.equal(initial[k],v)]
    assert changed and nonzero>0 and any('blocks_st' in k for k in changed) and any('blocks_ts' in k for k in changed)
    result=dict(status='trained_with_validation',stop_reason=stop,epochs_entered=len(history),updates=updates,
        encoder_updates=encoder_updates,nonzero_encoder_gradient_steps=nonzero,changed_best_encoder_tensors=len(changed),
        best_epoch=best_epoch,best_validation_score=best,selection=basis,maximum_gradient_norm_before_clipping=maxgrad,
        elapsed_s=time.monotonic()-started,test_data_consumed=False,
        checkpoints={p.name:sha(p) for p in folder.glob('*.pt')})
    write(folder/'training_result.json',result)
    del model,optimizer,initial,best_state;torch.cuda.empty_cache()
    return result

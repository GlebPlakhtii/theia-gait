"""One five-minute device pilot, head warm-up, then full encoder fine-tuning."""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import torch
import numpy as np
from gait_assessment_data import load_dataset, audit_dataset, split_indices, window_dataset
from gait_from_scratch_training import audit_inputs, selection_basis, TrainingConfig, validate_model, state_digest
from gait_from_scratch_model import TCNConfig, masked_multitask_loss
from gait_from_scratch_sampling import sample_training_epoch
from motionbert_assessment_model import MotionBertAssessment, verify_initial_sources
from motionbert_gait_common import ROOT, CHECKPOINT_SHA, SEED, now, sha, write


def train(folder):
    folder=Path(folder);scratch=ROOT/'results/gait_from_scratch_v1'
    assert json.loads((scratch/'final_verification.json').read_text())['status']=='scratch_experiment_complete'
    assert json.loads((ROOT/'results/gait_metrics_20261114/production_status.json').read_text())['phase']=='pose_complete'
    folder.mkdir(parents=True,exist_ok=True)
    training=folder/'training';training.mkdir(exist_ok=False)
    (folder/'dataset').symlink_to(scratch/'dataset',target_is_directory=True)
    torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
    source=verify_initial_sources();data=load_dataset(folder/'dataset');audit=audit_dataset(data,folder/'dataset')
    trainset=window_dataset(data,split_indices(data,0));valset=window_dataset(data,split_indices(data,1,True))
    enabled=torch.from_numpy(data['enabled']);audit_inputs(trainset,valset,enabled,TCNConfig());basis=selection_basis(valset,enabled)
    write(folder/'dataset_audit.json',audit)
    device=torch.device('cuda');model=MotionBertAssessment().to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    positive=torch.tensor(data['baselines']['positive_class_weight'],dtype=torch.float32,device=device)
    microbatch=4;accumulation=4
    # Profile on training windows only. Backpropagation is exercised but no
    # optimizer exists and no parameter updates occur during this timing pilot.
    ids,_=sample_training_epoch(trainset.identities,0,SEED)
    usable=[int(i) for i in ids if any(mask[i].any() for mask in trainset.masks.values())]
    pilot_start=time.monotonic();durations=[];pilot_steps=0;ooms=[];last_progress=pilot_start
    model.train();torch.cuda.reset_peak_memory_stats(device)
    while time.monotonic()-pilot_start<300:
        batch=usable[:microbatch]
        try:
            model.zero_grad(set_to_none=True)
            pose,context,target,masks,method=trainset.batch(batch,device,enabled)
            torch.cuda.synchronize();start=time.monotonic();out=model(pose,context)
            loss=masked_multitask_loss(out,target,masks,method,positive_class_weight=positive)['total']
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pilot loss')
            loss.backward();torch.cuda.synchronize();durations.append(time.monotonic()-start);pilot_steps+=1
            if time.monotonic()-last_progress>=30:
                write(folder/'training_status.json',dict(status='running',phase='throughput_vram_pilot',updated_at=now(),
                    elapsed_s=time.monotonic()-pilot_start,pilot_steps=pilot_steps,microbatch=microbatch))
                last_progress=time.monotonic()
        except torch.OutOfMemoryError:
            ooms.append(dict(microbatch=microbatch,elapsed_s=time.monotonic()-pilot_start))
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            if microbatch==1:raise
            microbatch//=2;durations=[];pilot_steps=0
    peak=torch.cuda.max_memory_allocated(device);model.zero_grad(set_to_none=True)
    assert all(torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items()),'Pilot changed weights'
    pilot=dict(status='passed',created_at=now(),duration_s=time.monotonic()-pilot_start,steps=pilot_steps,
        microbatch=microbatch,accumulation=accumulation,mean_forward_backward_s=float(np.mean(durations)),
        estimated_epoch_training_s=len(ids)/microbatch*float(np.mean(durations)),peak_allocated_bytes=peak,
        oom_adjustments=ooms,optimizer_updates=0,weights_unchanged=True,training_data_only=True,
        device=torch.cuda.get_device_name(),autocast='bfloat16' if model.use_bf16 else 'float32')
    write(folder/'device_pilot.json',pilot);print('Device pilot:',pilot,flush=True)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    encoder_ids={id(p) for p in model.encoder.parameters()}
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=1e-5),
        dict(params=[p for p in model.parameters() if id(p) not in encoder_ids],lr=1e-4)],weight_decay=.01)
    started=time.monotonic();cutoff=datetime(2026,11,15,5,30,tzinfo=timezone.utc).timestamp()
    remaining=min(7200,cutoff-time.time());assert remaining>0,'Training deadline has passed'
    deadline=started+remaining
    config=dict(seed=SEED,maximum_epochs=20,minimum_encoder_epochs=3,patience=4,
        microbatch=microbatch,accumulation=accumulation,encoder_lr=1e-5,head_lr=1e-4,weight_decay=.01,
        gradient_clip=1.,warmup_epochs=1,warmup_cap_s=300,training_cap_s=remaining,
        deadline_utc='2026-11-15T05:30:00+00:00',selection=basis,parameter_count=model.parameter_count,
        initial_checkpoint_sha256=CHECKPOINT_SHA,vendor_revision=source['revision'],
        dataset_sha256=data['manifest']['dataset_sha256'],dataset_manifest_sha256=sha(folder/'dataset/dataset_manifest.json'),
        code_sha256={n:sha(ROOT/'scripts'/n) for n in ['train_motionbert_assessment.py','motionbert_assessment_model.py','gait_assessment_data.py']},
        autocast='bfloat16' if model.use_bf16 else 'float32',input_context=list(TCNConfig().context_names),
        input_channels=list(TCNConfig().channels),test_data_consumed=False,
        unused_original_head='Pretrained 3D coordinate head is retained in state but outside get_representation; all encoder blocks train.')
    write(training/'resolved_training.json',config)
    history=[];updates=encoder_updates=nonzero_encoder_steps=0;best=None;best_epoch=None;stale=0;maxgrad=0.
    stop='maximum_epochs'
    def save(name,epoch,val=None):
        path=training/(name+'.pt');tmp=path.with_suffix('.pt.tmp')
        torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),epoch=epoch,
            updates=updates,encoder_updates=encoder_updates,configuration=config,validation=val,
            torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),tmp);tmp.replace(path)
    save('initial',0)
    for epoch in range(1,21):
        if time.monotonic()>=deadline:stop='time_cap';break
        warmup=epoch==1
        for parameter in model.encoder.parameters():parameter.requires_grad_(not warmup)
        model.train();indices,sampling=sample_training_epoch(trainset.identities,epoch-1,SEED)
        totals={k:0. for k in ['classification','regression','evaluability','total']};batches=0;pending=0
        epoch_start=time.monotonic();optimizer.zero_grad(set_to_none=True)
        def step(pending_count):
            nonlocal updates,encoder_updates,nonzero_encoder_steps,maxgrad
            if pending_count!=accumulation:
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(accumulation/pending_count)
            if not warmup:
                norm=sum(float(p.grad.detach().float().square().sum()) for p in model.encoder.parameters() if p.grad is not None)
                nonzero_encoder_steps+=int(norm>0)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);maxgrad=max(maxgrad,float(grad))
            optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;encoder_updates+=int(not warmup)
        for start in range(0,len(indices),microbatch):
            if time.monotonic()>=deadline:stop='time_cap';break
            if warmup and time.monotonic()-epoch_start>=300:break
            pose,context,target,masks,method=trainset.batch(indices[start:start+microbatch],device,enabled)
            if not any(mask.any() for mask in masks.values()):continue
            out=model(pose,context);loss=masked_multitask_loss(out,target,masks,method,positive_class_weight=positive)
            if not all(torch.isfinite(v) for v in loss.values()):raise FloatingPointError('Nonfinite training loss')
            (loss['total']/accumulation).backward();pending+=1;batches+=1
            for k in totals:totals[k]+=float(loss[k].detach())
            if pending==accumulation:step(pending);pending=0
        if pending:step(pending)
        save('last',epoch)
        if warmup:save('warmup',epoch)
        vc=replace(TrainingConfig(),batch_size=microbatch)
        validation=validate_model(model,valset,enabled,basis,vc,device,deadline,time.monotonic) if time.monotonic()<deadline else None
        row=dict(epoch=epoch,warmup=warmup,batches=batches,training_loss={k:v/batches if batches else None for k,v in totals.items()},
            validation=validation,elapsed_s=time.monotonic()-started,epoch_seconds=time.monotonic()-epoch_start,
            encoder_optimizer_steps=encoder_updates,sampling=sampling)
        history.append(row);write(training/'training_history.json',history)
        if validation is None:stop='time_cap';break
        score=validation['selection_score']
        if not warmup:
            improved=best is None or (score>best if basis['direction']=='maximize' else score<best)
            if improved:best=score;best_epoch=epoch;stale=0;save('best',epoch,validation)
            else:stale+=1
        write(folder/'training_status.json',dict(status='running',phase='head_warmup' if warmup else 'encoder_finetuning',
            updated_at=now(),epoch=epoch,encoder_updates=encoder_updates,validation_score=score,elapsed_s=row['elapsed_s']))
        print(json.dumps({k:v for k,v in row.items() if k not in ['sampling','validation']}),flush=True)
        if epoch>=4 and stale>=4:stop='early_stopping';break
    changed=[k for k,v in model.state_dict().items() if k.startswith('encoder.') and not torch.equal(initial[k],v.detach().cpu())]
    assert best_epoch is not None and encoder_updates>0 and nonzero_encoder_steps>0 and changed
    best_state=torch.load(training/'best.pt',map_location='cpu',weights_only=False)
    best_changed=[k for k,v in best_state['model_state'].items() if k.startswith('encoder.') and not torch.equal(initial[k],v)]
    assert best_changed and best_state['encoder_updates']>0
    audit=dict(created_at=now(),actual_backbone_finetuning=True,encoder_optimizer_steps=encoder_updates,
        nonzero_encoder_gradient_steps=nonzero_encoder_steps,changed_encoder_tensors=len(changed),
        changed_parameter_names=changed,best_changed_encoder_tensors=len(best_changed),
        best_encoder_parameter_change_l2=float(sum((best_state['model_state'][k]-initial[k]).double().square().sum() for k in best_changed).sqrt()))
    write(folder/'backbone_update_audit.json',audit)
    result=dict(status='trained_with_validation',stop_reason=stop,epochs_entered=len(history),updates=updates,
        encoder_updates=encoder_updates,best_epoch=best_epoch,best_validation_score=best,selection=basis,
        maximum_gradient_norm_before_clipping=maxgrad,elapsed_s=time.monotonic()-started,test_data_consumed=False,
        checkpoints={p.name:sha(p) for p in training.glob('*.pt')})
    write(training/'training_result.json',result);write(folder/'training_status.json',dict(**result,updated_at=now()))
    print('Fine-tuning complete:',result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/motionbert_gait_finetune_v1/completed_run')
    args=p.parse_args();train(args.output)

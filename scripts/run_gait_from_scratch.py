"""One real-data TCN experiment, freshly initialized, on the audited split."""
import argparse
from pathlib import Path
import torch
from gait_assessment_data import load_dataset, audit_dataset, split_indices, window_dataset
from gait_from_scratch_training import train_model, audit_inputs, selection_basis
from gait_from_scratch_model import TCNConfig
from motionbert_gait_common import ROOT, sha, write


def run(folder, device='cuda'):
    folder=Path(folder); torch.set_num_threads(2)
    data=load_dataset(folder/'dataset')
    audit=audit_dataset(data,folder/'dataset')
    train=window_dataset(data,split_indices(data,0))
    validation=window_dataset(data,split_indices(data,1,primary_only=True))
    enabled=torch.from_numpy(data['enabled'])
    audit_inputs(train,validation,enabled,TCNConfig())
    audit.update(train_windows=len(train),validation_windows=len(validation),
                 selection=selection_basis(validation,enabled))
    write(folder/'dataset_audit.json',audit)
    provenance=dict(dataset_manifest_sha256=sha(folder/'dataset/dataset_manifest.json'),
        dataset_sha256=data['manifest']['dataset_sha256'], dataset_audit_sha256=sha(folder/'dataset_audit.json'),
        code_sha256={name:sha(ROOT/'scripts'/name) for name in ['gait_from_scratch_model.py',
            'gait_from_scratch_training.py','gait_from_scratch_sampling.py','gait_assessment_data.py','run_gait_from_scratch.py']},
        clinical_labels=False, pretrained_assessment_weights=False)
    print('Frozen input audit passed:',audit,flush=True)
    result=train_model(train,validation,output_directory=folder/'training',enabled_classes=enabled,
        positive_class_weight=data['baselines']['positive_class_weight'],input_provenance=provenance,device=device)
    print(result,flush=True)
    assert result['status']=='trained_with_validation'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/gait_from_scratch_v1')
    p.add_argument('--device',default='cuda');args=p.parse_args();run(args.output,args.device)

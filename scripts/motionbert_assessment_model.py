"""Official MotionBERT encoder with new assessment heads and explicit masks."""
from contextlib import nullcontext
import json
import sys
import torch
from torch import nn
from motionbert_gait_common import ROOT, CHECKPOINT_SHA, sha

sys.path.insert(0,str(ROOT/'vendor/motionbert'))
from lib.model.DSTformer import DSTformer


def verify_initial_sources():
    manifest=json.loads((ROOT/'vendor/motionbert/source_manifest.json').read_text())
    assert manifest['revision']=='705d3a95354db8bdb696b3492e47a3b5537174ff'
    for name,digest in manifest['files'].items():assert sha(ROOT/'vendor/motionbert'/name)==digest,name
    assert sha(ROOT/'models/motionbert/pose3d_h36m.bin')==CHECKPOINT_SHA
    return manifest


class MotionBertAssessment(nn.Module):
    def __init__(self,load_initial=True):
        super().__init__()
        self.encoder=DSTformer(dim_feat=512,dim_rep=512,depth=5,num_heads=8,mlp_ratio=2,num_joints=17,maxlen=243)
        if load_initial:
            verify_initial_sources()
            original=torch.load(ROOT/'models/motionbert/pose3d_h36m.bin',map_location='cpu',weights_only=True)['model_pos']
            self.encoder.load_state_dict({k.removeprefix('module.'):v for k,v in original.items()},strict=True)
        self.readout=nn.Sequential(nn.LayerNorm(520),nn.Linear(520,128),nn.GELU(),nn.Dropout(.1))
        self.classifier=nn.Linear(128,13)
        self.numeric_heads=nn.ModuleList([nn.Linear(128,10) for _ in range(2)])
        self.evaluability=nn.Linear(128,10)
        self.use_bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    def forward(self,pose,context):
        if pose.shape[1:]!=(30,17,4) or context.shape!=(len(pose),8):raise ValueError('Frozen adapter/context schema mismatch')
        valid=pose[...,3].bool()
        if not valid.any(dim=(1,2)).all():raise ValueError('All-invalid windows cannot enter encoder')
        x=torch.where(valid[...,None],pose[...,:3],torch.zeros_like(pose[...,:3]))
        with torch.autocast('cuda',dtype=torch.bfloat16) if pose.is_cuda and self.use_bf16 else nullcontext():
            representation=self.encoder.get_representation(x)
            confidence=x[...,2:3].clamp(0,1)
            pooled=(representation*confidence).sum((1,2))/confidence.sum((1,2)).clamp_min(1)
            z=self.readout(torch.cat([pooled,context],-1))
            return dict(class_logits=self.classifier(z).float(),
                numeric=torch.stack([head(z) for head in self.numeric_heads],1).float(),
                evaluability_logits=self.evaluability(z).float())

    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters())

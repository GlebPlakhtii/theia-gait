"""Pose-only geometric sequence encoder; no window statistics or teacher labels.

Features are computed per frame from H36M-17 2D joints. They do not recover
depth, compensate camera rotation, or establish clinical normality.
"""
import torch
from torch import nn
from torch.nn import functional as F
from gait_from_scratch_model import GaitTCN, TCNConfig

PARENTS=(0,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15)
TRIPLES=((1,2,3),(4,5,6),(11,12,13),(14,15,16),(0,1,2),(0,4,5),(0,8,11),(0,8,14))
LINES=((1,4),(14,11),(0,8))
FEATURE_WIDTH=17*5+17*3+17*3+8*3+3*3


def geometric_sequence(pose):
    valid=pose[...,3].bool()
    xy=torch.where(valid[...,None],pose[...,:2],torch.zeros_like(pose[...,:2]))
    confidence=torch.where(valid,pose[...,2],torch.zeros_like(pose[...,2]))
    delta=xy-xy[:,:,PARENTS]
    length=delta.norm(dim=-1)
    bone_valid=valid&valid[:,:,PARENTS]&(length>1e-6)
    unit=delta/length.clamp_min(1e-6)[...,None]
    unit=torch.where(bone_valid[...,None],unit,torch.zeros_like(unit))
    torso=(xy[:,:,8]-xy[:,:,0]).norm(dim=-1)
    root_valid=valid[:,:,0]&valid[:,:,8]&(torso>1e-6)
    coord_valid=valid&root_valid[...,None]
    relative=(xy-xy[:,:,:1])/torso.clamp_min(1e-6)[...,None,None]
    relative=torch.where(coord_valid[...,None],relative.clamp(-5,5),torch.zeros_like(relative))
    lengths=torch.where(bone_valid&root_valid[...,None],(length/torso.clamp_min(1e-6)[...,None]).clamp(0,5),torch.zeros_like(length))
    velocity=torch.zeros_like(unit); motion_valid=torch.zeros_like(bone_valid)
    motion_valid[:,1:]=bone_valid[:,1:]&bone_valid[:,:-1]
    velocity[:,1:]=unit[:,1:]-unit[:,:-1]
    velocity=torch.where(motion_valid[...,None],velocity,torch.zeros_like(velocity))
    features=[unit.flatten(2),lengths,confidence,valid.float(),relative.flatten(2),coord_valid.float(),velocity.flatten(2),motion_valid.float()]
    for a,b,c in TRIPLES:
        u,v=xy[:,:,a]-xy[:,:,b],xy[:,:,c]-xy[:,:,b]
        lu,lv=u.norm(dim=-1),v.norm(dim=-1)
        good=valid[:,:,a]&valid[:,:,b]&valid[:,:,c]&(lu>1e-6)&(lv>1e-6)
        den=(lu*lv).clamp_min(1e-12)
        cos=(u*v).sum(-1)/den;sin=(u[:,:,0]*v[:,:,1]-u[:,:,1]*v[:,:,0])/den
        features.append(torch.stack([torch.where(good,cos.clamp(-1,1),0),torch.where(good,sin.clamp(-1,1),0),good.float()],-1))
    for a,b in LINES:
        v=xy[:,:,b]-xy[:,:,a];length=v.norm(dim=-1)
        good=valid[:,:,a]&valid[:,:,b]&(length>1e-6)
        v=torch.where(good[...,None],v/length.clamp_min(1e-6)[...,None],0)
        features.append(torch.cat([v,good[...,None]],-1))
    result=torch.cat(features,-1)
    assert result.shape[-1]==FEATURE_WIDTH
    return result,valid.any(-1)


class KinematicTCN(GaitTCN):
    def __init__(self,config=None):
        super().__init__(config)
        self.frame_embedding=nn.Linear(FEATURE_WIDTH,self.config.width)

    def forward(self,pose,context):
        if pose.ndim!=4 or tuple(pose.shape[1:])!=(30,17,4):
            raise ValueError('Expected B x 30 x 17 x 4')
        if not torch.isfinite(pose).all() or not torch.isfinite(context).all():
            raise ValueError('Inputs must be finite')
        if not ((pose[...,3]==0)|(pose[...,3]==1)).all():
            raise ValueError('Validity must be binary')
        features,mask=geometric_sequence(pose)
        if not mask.any(1).all():raise ValueError('All-invalid sequence')
        x=F.gelu(self.frame_embedding(features))*mask[...,None]
        for block in self.blocks:x=block(x,mask)
        mean=x.sum(1)/mask.sum(1,keepdim=True)
        maximum=x.masked_fill(~mask[...,None],-torch.inf).amax(1)
        z=self.readout(torch.cat([mean,maximum,context],-1))
        return dict(class_logits=self.classifier(z),numeric=torch.stack([h(z) for h in self.numeric_heads],1),evaluability_logits=self.evaluability(z))


def create_model(condition):
    config=TCNConfig(dropout=.15)
    return GaitTCN(config) if condition=='raw' else KinematicTCN(config)

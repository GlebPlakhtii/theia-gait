"""Learned frame features with permutation-invariant statistical pooling.

No target ROM, asymmetry value, reference threshold or cohort label is input.
The heads approximate existing automatic criteria; this is not diagnosis.
"""
import torch
from torch import nn
from torch.nn import functional as F
from gait_from_scratch_model import GaitTCN,TCNConfig
from gait_kinematic_model import KinematicTCN,geometric_sequence,FEATURE_WIDTH

class StatisticalPoseModel(GaitTCN):
    def __init__(self):
        config=TCNConfig(dropout=.2)
        super().__init__(config)
        # Remove direction-change features: every remaining input is per-frame.
        self.frame_embedding=nn.Linear(FEATURE_WIDTH-51,96)
        self.frame_net=nn.Sequential(nn.LayerNorm(96),nn.GELU(),nn.Linear(96,96),nn.GELU())
        self.blocks=nn.ModuleList()
        self.readout=nn.Sequential(nn.Linear(96*4+8,128),nn.GELU(),nn.Dropout(.2))

    def forward(self,pose,context):
        if not torch.isfinite(pose).all() or not torch.isfinite(context).all():raise ValueError('Nonfinite input')
        features,mask=geometric_sequence(pose)
        if not mask.any(1).all():raise ValueError('All-invalid window')
        features=torch.cat([features[:,:,:136],features[:,:,187:]],-1)
        x=self.frame_net(self.frame_embedding(features));n=mask.sum(1,keepdim=True)
        mean=(x*mask[...,None]).sum(1)/n
        variance=((x-mean[:,None]).square()*mask[...,None]).sum(1)/n
        std=(variance+1e-8).sqrt()
        maximum=x.masked_fill(~mask[...,None],-torch.inf).amax(1)
        minimum=x.masked_fill(~mask[...,None],torch.inf).amin(1)
        z=self.readout(torch.cat([mean,std,minimum,maximum,context],-1))
        return dict(class_logits=self.classifier(z),numeric=torch.stack([h(z) for h in self.numeric_heads],1),evaluability_logits=self.evaluability(z))

def create_refined_model(condition):
    if condition=='geometry_set':return StatisticalPoseModel()
    if condition=='geometry_slow':return KinematicTCN(TCNConfig(dropout=.2))
    raise ValueError(condition)

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(31)
    p=torch.randn(2,30,17,4);p[...,2]=.9;p[...,3]=1;p[:,5,:,3]=0
    c=torch.zeros(2,8);c[:,0]=1;c[:,5]=1
    m=StatisticalPoseModel().eval()
    with torch.no_grad():
        a=m(p,c);b=m(p[:,torch.randperm(30)],c)
        q=p.clone();q[...,:2]=q[...,:2]*1.4+torch.tensor([.2,-.3]);d=m(q,c)
    for k in a:
        torch.testing.assert_close(a[k],b[k],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(a[k],d[k],atol=2e-6,rtol=2e-5)
    print({'permutation_invariance':'passed','translation_scale_invariance':'passed','parameters':m.parameter_count})

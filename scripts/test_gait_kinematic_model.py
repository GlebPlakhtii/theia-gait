"""Behavioral checks for geometric inputs and masked frame handling."""
import torch
from gait_kinematic_model import KinematicTCN,geometric_sequence

def run():
    torch.manual_seed(13);torch.set_num_threads(2)
    pose=torch.randn(3,30,17,4)
    pose[...,2]=.8;pose[...,3]=1
    pose[:,5:8,3,3]=0;pose[:,15,:,3]=0
    context=torch.zeros(3,8);context[:,0]=1;context[:,5]=1
    model=KinematicTCN().eval()
    shifted=pose.clone();shifted[...,:2]=pose[...,:2]*2.4+torch.tensor([.43,-.72])
    invalid=pose.clone();invalid[...,:3][pose[...,3]==0]=12345.
    with torch.no_grad():
        a=model(pose,context);b=model(shifted,context);c=model(invalid,context)
    for key in a:
        torch.testing.assert_close(a[key],b[key],rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(a[key],c[key],rtol=0,atol=0)
    features,mask=geometric_sequence(pose)
    assert torch.isfinite(features).all() and not mask[:,15].any()
    empty=pose.clone();empty[0,...,3]=0
    try:model(empty,context)
    except ValueError:pass
    else:raise AssertionError('All-invalid input was accepted')
    model.train();loss=model(pose,context)['class_logits'].square().mean();loss.backward()
    assert model.frame_embedding.weight.grad.norm()>0
    print({'translation_scale_invariance':'passed','invalid_placeholder_invariance':'passed','missing_frame_gate':'passed','all_invalid_rejection':'passed','nonzero_gradient':'passed','parameters':model.parameter_count})

if __name__=='__main__':run()

"""Offline squat-form research pipeline: native 2D -> stabilization -> 3D lifting.

Existing doctor annotations are read-only. MotionBERT depth is an estimate,
not calibrated motion capture; downstream OpenSim is a kinematic consistency fit.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/squat_form_lab_20261114'
DELIVERY = ROOT / 'deliverables/Theia_Squat_Form_Lab_2026-11-14'
sys.path.insert(0, str(ROOT / 'scripts'))
from train_existing_data import NAMES, load_csv_pose

H36 = ['pelvis','right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle',
       'spine','thorax','nose','head','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist']
EDGES = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
PACKETS = [
    ('initial','Theia_Doctor_Clips_2026-11-14','audit/doctor_clip_source_mapping_20261114.json'),
    ('gerst','Theia_Doctor_Gerst_Clips_2026-11-14','audit/gerst_processing_20261114/short_clip_source_mapping.json'),
    ('extra','Theia_Doctor_Extra_Clips_2026-11-14','audit/new_video_intake_20261114/doctor_source_mapping.json')]


def now(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def status(stage, **kw):
    d=dict(stage=stage,at=now(),**kw);write(OUT/'status.json',d);print(json.dumps(d),flush=True)


def joint_angle(a,b,c):
    u,v=a-b,c-b
    den=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
    return np.degrees(np.arccos(np.clip(np.divide((u*v).sum(-1),den,out=np.full(den.shape,np.nan),where=den>1e-10),-1,1)))


def stabilize(p):
    """Foot-reference rigid alignment. Rotation/translation only, no fake depth."""
    xy=p[...,:2].copy();valid=np.isfinite(p).all(-1)&(p[...,2]>=.3)
    foot_ids=[15,16,17,18,19,20,21,22]
    core=[5,6,11,12,13,14,15,16]
    coverage=valid[:,foot_ids].sum(1)
    ext=(joint_angle(xy[:,11],xy[:,13],xy[:,15])+joint_angle(xy[:,12],xy[:,14],xy[:,16]))/2
    eligible=(coverage>=6)&valid[:,core].all(1)&np.isfinite(ext)
    if not eligible.any():
        reference=int(np.argmax(coverage))
    else:
        scores=np.where(eligible,ext,-np.inf);reference=int(np.argmax(scores))
    result=xy.copy();rotation=np.zeros(len(p));residual=np.full(len(p),np.nan);supported=np.zeros(len(p),bool)
    ref=xy[reference,foot_ids]
    for i in range(len(p)):
        use=valid[i,foot_ids]&valid[reference,foot_ids]
        if use.sum()<4:continue
        a,b=xy[i,foot_ids][use],ref[use];ac,bc=a.mean(0),b.mean(0)
        u,_,vt=np.linalg.svd((a-ac).T@(b-bc));r=u@vt
        if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@vt
        transformed=(xy[i]-ac)@r+bc
        result[i]=transformed;rotation[i]=np.arctan2(r[0,1],r[0,0]);supported[i]=True
        residual[i]=np.sqrt(np.mean(np.sum((transformed[foot_ids][use]-b)**2,axis=1)))
    # Define a source-relative upright axis using the most extended visible frame.
    up=(xy[reference,11]+xy[reference,12]-xy[reference,15]-xy[reference,16])/2
    theta=float(np.arctan2(up[0],-up[1]))
    rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    anchor=ref[np.isfinite(ref).all(1)].mean(0)
    aligned=(result-anchor)@rot
    return aligned, dict(reference_frame=reference, upright_rotation_deg=float(np.degrees(theta)),
        foot_reference_assumption='Feet remain attached to a common support; actual foot movement may be absorbed. Inspect residual.',
        axis_assumption='Most-extended observed pose defines reference up; not a calibrated gravity direction.'), rotation, residual, supported


def h36m_from_named(p, xy):
    x=np.zeros((len(p),17,3),np.float32)
    for target,name in [(1,'right_hip'),(2,'right_knee'),(3,'right_ankle'),(4,'left_hip'),(5,'left_knee'),(6,'left_ankle'),
                         (9,'nose'),(11,'left_shoulder'),(12,'left_elbow'),(13,'left_wrist'),(14,'right_shoulder'),(15,'right_elbow'),(16,'right_wrist')]:
        j=NAMES.index(name);x[:,target,:2]=xy[:,j];x[:,target,2]=np.nan_to_num(p[:,j,2])
    for target,ids in [(0,[11,12]),(8,[5,6]),(10,[1,2])]:
        x[:,target,:2]=np.mean(xy[:,ids],axis=1);x[:,target,2]=np.min(np.nan_to_num(p[:,ids,2]),axis=1)
    x[:,7,:2]=(x[:,0,:2]+x[:,8,:2])/2;x[:,7,2]=np.minimum(x[:,0,2],x[:,8,2])
    # Head is approximated from eyes/nose because the input has no measured H36M head-top marker.
    eye_mid=xy[:,[1,2]].mean(1);x[:,10,:2]=eye_mid+.5*(eye_mid-xy[:,0])
    x[...,2]=np.clip(x[...,2],0,1)
    good=np.isfinite(x).all(-1)&(x[...,2]>=.3)
    for j in range(17):
        ids=np.where(good[:,j])[0]
        if len(ids):
            for dim in range(2):x[:,j,dim]=np.interp(np.arange(len(x)),ids,x[ids,j,dim])
        else:x[:,j,:2]=0
    x[...,2][~good]=0
    valid=x[good,:2]
    if not len(valid):raise ValueError('No valid model input')
    lo,hi=np.quantile(valid,[.005,.995],axis=0);scale=float(max(hi-lo))
    center=(hi+lo)/2
    x[...,:2]=np.clip((x[...,:2]-center)/(scale/2),-1,1)
    return x, dict(crop_center=center.tolist(),crop_scale_pixels=scale,
                   missing_input_coordinates='Interpolated for model input only; confidence stays zero and decisions remain gated',
                   h36m_mapping='Named limbs; pelvis=hip midpoint; thorax=shoulder midpoint; spine=midpoint; head=eye/nose approximation')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);DELIVERY.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'results/training_all_data_20261114/data_manifest.json').read_text())['records']
    by_sha={r['pose_sha256']:r for r in manifest if r['task']!='gait'}
    sources=[];clips=[]
    for packet,name,mapping in PACKETS:
        folder=ROOT/'deliverables'/name
        rows=json.loads((folder/'automatic_labels/clips.json').read_text())
        for s in json.loads((ROOT/mapping).read_text()):
            rec=by_sha[s['pose_csv_sha256']];path=ROOT/rec['pose_path'];t,p=load_csv_pose(path)
            video=Path(s['original_cut_input']);probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames','-of','json',str(video)]))['streams'][0]
            assert len(p)==s['frames'] and int(probe['nb_frames'])==len(p)
            assert digest(path)==s['pose_csv_sha256']
            stabilized,stab,rot,residual,available=stabilize(p)
            x,mapping_info=h36m_from_named(p,stabilized)
            folder_out=OUT/'sources'/s['id'];folder_out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder_out/'inputs.npz',time=t,pose2d=p,stabilized=stabilized,model_input=x,
                                foot_rotation=rot,foot_residual=residual,foot_supported=available)
            row=dict(**s,pose_path=rec['pose_path'],width=int(probe['width']),height=int(probe['height']),
                     packet=packet,stabilization=stab,mapping=mapping_info,
                     publisher_label=('bad' if '_bad_' in rec['id'] else 'good' if '_good_' in rec['id'] else None),
                     subject=rec['subject'],model_input_sha256=digest(folder_out/'inputs.npz'))
            sources.append(row)
        for c in rows:
            clips.append(dict(**c,packet=packet,original_path=str((folder/c['original_video']).relative_to(ROOT)),
                              overlay_path=str((folder/c['pose_video']).relative_to(ROOT))))
    write(OUT/'sources.json',sources);write(OUT/'clips_input.json',clips)
    write(OUT/'protocol.json',dict(at=now(),scope='Offline research form screening on ground and in-mission ARED squats',
        source_sequences=len(sources),clips=len(clips),native_frames=sum(s['frames'] for s in sources),
        pose='Existing Sapiens2-5B tracks, held quality exclusions preserved',
        projection='Foot-reference 2D rigid stabilization + MotionBERT monocular lifting + body-relative analysis',
        opensim='Custom lower-body/trunk kinematic model fitted to estimated 3D markers; no force/muscle simulation',
        truth='No clinician-approved thresholds or independent ARED good/bad verdicts available',
        evaluation='Agreement with gym publisher file labels is exploratory; labels are not per-repetition expert ground truth',
        causal=False,notes=['Centered temporal processing uses future frames; retrospective rep assessment, not forecasting.',
        'No rule is a validated injury predictor. Good means no active rule exceeded, not medical clearance.',
        'A body-relative 3D reconstruction is not metric ground truth; yaw/depth are model estimates.',
        'No prescribed squat depth or measured ARED load is supplied. ARED uses within-source reference changes.']))
    status('prepared',sources=len(sources),clips=len(clips),frames=sum(s['frames'] for s in sources))


def lift():
    import torch
    sys.path.insert(0,str(ROOT/'vendor/motionbert'))
    from lib.model.DSTformer import DSTformer
    from lib.utils.utils_data import flip_data
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    model=DSTformer(dim_feat=512,dim_rep=512,depth=5,num_heads=8,mlp_ratio=2,num_joints=17,maxlen=243).cuda().eval()
    weights=torch.load(ROOT/'models/motionbert/pose3d_h36m.bin',map_location='cpu',weights_only=True)['model_pos']
    model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True)
    for s in json.loads((OUT/'sources.json').read_text()):
        folder=OUT/'sources'/s['id'];target=folder/'lifted.npz'
        if target.exists():continue
        a=np.load(folder/'inputs.npz');x=a['model_input'];n=len(x)
        starts=list(range(0,max(1,n-243+1),162))
        if n>243 and starts[-1]!=n-243:starts.append(n-243)
        result=np.zeros((n,17,3),np.float64);count=np.zeros(n);disagree=np.zeros((n,17),np.float64)
        started=time.monotonic()
        for start in starts:
            stop=min(n,start+243);batch=torch.from_numpy(x[start:stop][None]).cuda()
            with torch.inference_mode():
                first=model(batch);second=flip_data(model(flip_data(batch)))
                first=first-first[:,:,:1];second=second-second[:,:,:1]
                mean=((first+second)/2)[0].cpu().numpy();delta=torch.linalg.vector_norm(first-second,dim=-1)[0].cpu().numpy()
            result[start:stop]+=mean;disagree[start:stop]+=delta;count[start:stop]+=1
        assert (count>0).all()
        result/=count[:,None,None];disagree/=count[:,None]
        # Camera coordinates -> right-handed OpenSim axes. Absolute orientation remains arbitrary.
        world=result[:,:,[2,1,0]].copy();world[:,:,1]*=-1
        lengths=[]
        for h,k,ank in [(1,2,3),(4,5,6)]:lengths.append(np.linalg.norm(world[:,h]-world[:,k],axis=-1)+np.linalg.norm(world[:,k]-world[:,ank],axis=-1))
        nominal_scale=float(np.median(lengths))
        world/=nominal_scale;disagree/=nominal_scale
        np.savez_compressed(target,camera=result.astype(np.float32),world=world.astype(np.float32),
                            flip_disagreement=disagree.astype(np.float32),window_overlap=count,
                            nominal_scale=nominal_scale)
        write(folder/'lift_complete.json',dict(at=now(),frames=n,windows=len(starts),elapsed_s=time.monotonic()-started,
            checkpoint_sha256=digest(ROOT/'models/motionbert/pose3d_h36m.bin'),input_sha256=digest(folder/'inputs.npz'),
            output_sha256=digest(target),units='Nominal leg length = 1 model metre; subject height is unknown. Angles are scale-invariant.',
            depth_status='Estimated by Earth-trained MotionBERT H36M model; not validated for ARED microgravity',
            left_right_flip_difference='Model sensitivity indicator, not a calibrated confidence interval'))
        status('lifting',source=s['id'],frames=n,elapsed_s=round(time.monotonic()-started,2))
    status('lifting_complete')


def tests():
    a=np.array([0.,1.]);b=np.array([0.,0.]);c=np.array([1.,0.])
    assert abs(joint_angle(a,b,c)-90)<1e-8
    r=Rotation.from_euler('xyz',[.2,.5,.3]).as_matrix()
    a,b,c=[np.r_[p,0] for p in [a,b,c]]
    assert abs(joint_angle(a@r,b@r,c@r)-90)<1e-8
    print('Joint-angle rotation-invariance check passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','lift','test']);args=parser.parse_args()
    if args.action=='prepare':tests();prepare()
    elif args.action=='lift':lift()
    else:tests()

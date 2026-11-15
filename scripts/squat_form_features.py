"""Compute traceable repetition measurements and build portable viewer data."""
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from squat_form_pipeline import ROOT,OUT,DELIVERY,NAMES,H36,EDGES,joint_angle,write,digest,status

EVIDENCE=[
 dict(id='NASA2014',title='Newby et al. — ISS squat and deadlift kinematics on ARED',
      url='https://ntrs.nasa.gov/citations/20140013188',kind='Primary NASA conference study; four crew members',
      supports='Foot-fixed visualization can remove apparatus motion. Reported in-flight squat kinematics varied across crew and differed from ground ARED.',
      does_not_support='A universal ARED squat depth, automatic load adjustment, or validated good/bad cutoff.'),
 dict(id='GRABER2023',title='Graber et al. — Trunk and shank position and hip-to-knee moment ratio',
      url='https://pmc.ncbi.nlm.nih.gov/articles/PMC10518215/',kind='Primary laboratory study; 100 participants',
      supports='Trunk and shank positions both affect the distribution of hip/knee moments. Different squat strategies serve different purposes.',
      does_not_support='Calling forward trunk lean inherently bad, estimating forces from our videos, or prescribing one depth for everyone.'),
 dict(id='GWYNNE2014',title='Gwynne & Curran — Two-dimensional frontal-plane knee assessment',
      url='https://pmc.ncbi.nlm.nih.gov/articles/PMC4275194/',kind='Primary validation study; 18 participants, single-leg squat',
      supports='Standardized frontal video can quantify projected knee alignment; the reported measurement error was several degrees.',
      does_not_support='Transferring a single-leg protocol to ARED as validated, or treating an oblique 2D angle as true 3D valgus.'),
 dict(id='MOTIONBERT',title='MotionBERT — pretrained 3D motion reconstruction',
      url='https://github.com/Walter0807/MotionBERT',kind='Official implementation / ICCV 2023',
      supports='Lift 17-joint 2D sequences using an H36M-fine-tuned model, with horizontal-flip averaging.',
      does_not_support='Calibrated anatomical 3D, clinical diagnoses, or ARED-domain accuracy.'),
 dict(id='OPENSIM',title='OpenSim — inverse kinematics and model consistency',
      url='https://github.com/opensim-org/opensim-core',kind='Official OpenSim 4.6 engine',
      supports='An articulated model can fit supplied 3D marker trajectories and expose fit residuals.',
      does_not_support='Recovering missing camera depth by itself or inferring actual forces without appropriate measurements and a validated model.')]

DEFAULTS=dict(minCoverage=.80,maxFlip=.10,maxIk=.06,maxReprojection=.12,
              maxAsymmetry=15.,maxPhaseFraction=.15,groundDepth=90.,depthTolerance=10.,
              minExcursion=60.,depthDrop=15.,trunkChange=15.,frontalChange=10.,maxFrontalYaw=30.,
              minDepthDropFraction=.20,angleDeadband=5.)


def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.floating,float)):return round(float(x),5) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x


def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)
def smooth(v,fps):
    n=max(3,int(round(.12*fps))|1)
    return median_filter(v,size=(n,)+(1,)*(v.ndim-1),mode='nearest')
def percentile(x,q):
    a=np.asarray(x);a=a[np.isfinite(a)];return float(np.percentile(a,q)) if len(a) else None


def signals(source):
    folder=OUT/'sources'/source['id'];a=np.load(folder/'inputs.npz');b=np.load(folder/'lifted.npz');o=np.load(folder/'opensim/fit.npz')
    fps=source['fps'];p=a['pose2d'];raw=a['pose2d'][...,:2];st=a['stabilized'];world=smooth(b['world'],fps);fit=smooth(o['world'],fps)
    good=np.isfinite(p).all(-1)&(p[...,2]>=.3)
    core=good[:,[5,6,11,12,13,14,15,16]].all(1)
    flex0=((180-joint_angle(world[:,1],world[:,2],world[:,3]))+(180-joint_angle(world[:,4],world[:,5],world[:,6])))/2
    standing=np.argsort(flex0)[:max(3,len(world)//10)]
    up=unit(np.median(world[standing,0]-world[standing][:,[3,6]].mean(1),axis=0))
    # Source-up is defined by the most extended poses, not gravity or a force plate.
    lateral=unit(world[:,1]-world[:,4]);lateral=unit(lateral-(lateral@up)[:,None]*up)
    forward=unit(np.cross(lateral,np.broadcast_to(up,lateral.shape)))
    yaw=np.degrees(np.arctan2(np.abs(lateral[:,0]),np.abs(lateral[:,2])))
    out={}
    for side,coco,hh in [('left',(11,13,15),(4,5,6)),('right',(12,14,16),(1,2,3))]:
        h,k,ank=coco;H,K,A=hh
        for mode,points,ids in [('raw',raw,coco),('lift',world,hh),('ik',fit,hh)]:
            i,j,l=ids;v=180-joint_angle(points[:,i],points[:,j],points[:,l]);out[mode+'_'+side]=smooth(v,fps)
        # Signed frontal-plane projection in an estimated pelvis-relative plane.
        v1=world[:,H]-world[:,K];v2=world[:,A]-world[:,K]
        c1=np.stack([(v1*lateral).sum(-1),v1@up],-1);c2=np.stack([(v2*lateral).sum(-1),v2@up],-1)
        deviation=180-joint_angle(c1,np.zeros_like(c1),c2)
        hipx=(world[:,H]*lateral).sum(-1);kneex=(world[:,K]*lateral).sum(-1);ankx=(world[:,A]*lateral).sum(-1)
        hy,ky,ay=world[:,H]@up,world[:,K]@up,world[:,A]@up
        frac=np.divide(hy-ky,hy-ay,out=np.ones(len(p))*.5,where=np.abs(hy-ay)>1e-4)
        inward=(kneex-(hipx+frac*(ankx-hipx)))*(1 if side=='left' else -1)
        out['frontal_'+side]=smooth(deviation*np.sign(inward),fps)
    trunk=unit(world[:,8]-world[:,0]);trunk_angle=np.degrees(np.arctan2((trunk*forward).sum(-1),trunk@up))
    # Sign is relative to an estimated body forward axis; compare magnitudes for lean.
    out['trunk']=np.abs(smooth(trunk_angle,fps))
    shank=unit((world[:,2]-world[:,3]+world[:,5]-world[:,6])/2)
    out['shank']=np.abs(smooth(np.degrees(np.arctan2((shank*forward).sum(-1),shank@up)),fps))
    out['trunk_tibia']=out['trunk']-out['shank']
    out['yaw']=yaw
    out['ik_residual']=np.sqrt(np.mean(o['residual']**2,axis=1))
    out['flip']=np.mean(b['flip_disagreement'][:,[1,2,3,4,5,6]],axis=1)
    # Weak-perspective reprojection consistency (no estimated focal length or physical camera calibration).
    reproj=[]
    target=a['model_input'];cam=b['camera']
    for x,y in zip(cam,target):
        use=(y[:,2]>=.3);u=x[use,:2];v=y[use,:2]
        if len(u)<6:reproj.append(np.nan);continue
        uc,vc=u.mean(0),v.mean(0);scale=np.sum((u-uc)*(v-vc))/max(np.sum((u-uc)**2),1e-8)
        e=(u-uc)*scale+vc-v;reproj.append(np.sqrt(np.mean(np.sum(e**2,axis=1)))/2)
    out['reprojection']=np.array(reproj)
    for k,v in out.items():out[k]=np.asarray(v)
    out['valid']=core;out['time']=a['time'];out['foot_residual']=a['foot_residual'];out['foot_supported']=a['foot_supported']
    return out,dict(raw=raw,score=p[...,2],stabilized=st,world=world,fit=fit),up


def features_for_clip(c,s,g):
    # Exact native-frame lineage, not a re-decoded timestamp guess.
    start=c['source_first_frame']-s['source_frame_first'];end=start+c['frames']
    ta=g['time'][start:end]-c['source_start_s'];mask=(ta>=c['core_start_s']-1e-5)&(ta<=c['core_end_s']+1e-5)
    ids=np.arange(start,end)[mask];valid=g['valid'][ids];v=ids[valid]
    f=dict(coverage=float(valid.mean()),flip=percentile(g['flip'][ids],90),ikResidual=percentile(g['ik_residual'][ids],90),
        reprojection=percentile(g['reprojection'][ids],90),yaw=percentile(g['yaw'][ids],50),
        footSupport=float(g['foot_supported'][ids].mean()),footResidualPx=percentile(g['foot_residual'][ids],90),
        completeRep=c['automatic_type']=='candidate_repetition',nativeFrames=len(ids),duration=c.get('cycle_duration_s'))
    if not len(v):return f,start,end,ids
    f['depth']=(percentile(g['ik_left'][v],95)+percentile(g['ik_right'][v],95))/2
    f['rawDepth']=(percentile(g['raw_left'][v],95)+percentile(g['raw_right'][v],95))/2
    f['liftDepth']=(percentile(g['lift_left'][v],95)+percentile(g['lift_right'][v],95))/2
    for side in ['left','right']:
        f[side+'Rom']=percentile(g['ik_'+side][v],95)-percentile(g['ik_'+side][v],5)
        f[side+'Depth']=percentile(g['ik_'+side][v],95)
    f['romDifference']=abs(f['leftRom']-f['rightRom'])
    f['asymmetry']=percentile(np.abs(g['ik_left'][v]-g['ik_right'][v]),90)
    f['rawAsymmetry']=percentile(np.abs(g['raw_left'][v]-g['raw_right'][v]),90)
    f['meanRom']=(f['leftRom']+f['rightRom'])/2
    f['modelAngleDifference']=percentile(np.abs((g['ik_left'][v]+g['ik_right'][v]-g['lift_left'][v]-g['lift_right'][v])/2),90)
    mean=(g['ik_left'][v]+g['ik_right'][v])/2
    f['trunkAtDepth']=float(g['trunk'][v[np.argmin(np.abs(mean-60))]]) if np.min(np.abs(mean-60))<=5 else None
    f['trunkMax']=percentile(g['trunk'][v],95);f['trunkTibiaAtBottom']=float(g['trunk_tibia'][v[np.argmax(mean)]])
    tl=g['time'][v[np.argmax(g['ik_left'][v])]];tr=g['time'][v[np.argmax(g['ik_right'][v])]]
    f['phaseDifference']=float(abs(tl-tr));f['phaseFraction']=float(abs(tl-tr)/max(c.get('cycle_duration_s') or (g['time'][v[-1]]-g['time'][v[0]]),.1))
    depth_mask=mean>=np.quantile(mean,.75);top_mask=mean<=np.quantile(mean,.25)
    frontal=[]
    for side in ['left','right']:
        value=float(np.median(g['frontal_'+side][v[depth_mask]])-np.median(g['frontal_'+side][v[top_mask]]))
        frontal.append(value)
    f['frontalChange']=max(frontal);f['frontalSide']='left' if frontal[0]>=frontal[1] else 'right'
    f['depthDrop']=None;f['depthDropFraction']=None;f['trunkChange']=None;f['referenceIds']=[]
    return f,start,end,ids


def build_features():
    sources=json.loads((OUT/'sources.json').read_text());clips=json.loads((OUT/'clips_input.json').read_text())
    all_rows=[];source_rows=[]
    for s in sources:
        g,pose,up=signals(s);sc=sorted([c for c in clips if c['source_id']==s['id']],key=lambda c:c['source_first_frame'])
        rows=[];reference=[]
        for c in sc:
            f,start,end,ids=features_for_clip(c,s,g)
            if f['completeRep'] and f['coverage']>=DEFAULTS['minCoverage'] and (f['ikResidual'] or 0)<=DEFAULTS['maxIk']:
                if len(reference)<3:reference.append((c['clip_id'],f))
                else:
                    baseline=float(np.median([r['depth'] for _,r in reference]));f['depthDrop']=baseline-f['depth'];f['depthDropFraction']=(baseline-f['depth'])/max(baseline,1)
                    values=[r['trunkAtDepth'] for _,r in reference if r.get('trunkAtDepth') is not None]
                    if len(values)>=2 and f['trunkAtDepth'] is not None:f['trunkChange']=f['trunkAtDepth']-float(np.median(values))
                    f['referenceIds']=[cid for cid,_ in reference];f['baselineDepth']=baseline
            row={k:v for k,v in c.items() if k not in ['automatic_measurements','doctor_conclusion','doctor_notes']}
            row.update(features=f,publisherLabel=s['publisher_label'],subject=s['subject'],sourceDomain=s['domain'],
                       width=s['width'],height=s['height'],nativePoseStartIndex=start,
                       provisional=True,profile='ared' if s['domain']=='in_mission_ared' else 'ground')
            rows.append(row)
            # Per-clip pose/series assets preserve actual native frames; no added observations.
            d=dict(times=(g['time'][start:end]-c['source_start_s']),pose2d=pose['raw'][start:end],scores=pose['score'][start:end],
                   stabilized=pose['stabilized'][start:end],world=pose['world'][start:end],opensim=pose['fit'][start:end],
                   signals={k:g[k][start:end] for k in ['raw_left','raw_right','lift_left','lift_right','ik_left','ik_right','trunk','shank','yaw','ik_residual','valid']})
            target=DELIVERY/'tracks'/f'{c["clip_id"]}.js';target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('window.THEIA_TRACKS=window.THEIA_TRACKS||{};window.THEIA_TRACKS['+json.dumps(c['clip_id'])+']='+json.dumps(clean(d),separators=(',',':'))+';\n')
            media=DELIVERY/'videos'/f'{c["clip_id"]}.mp4';media.parent.mkdir(parents=True,exist_ok=True)
            original=ROOT/c['original_path']
            if not media.exists():os.link(original,media)
            row['video']='videos/'+media.name;row['track']='tracks/'+target.name
        all_rows.extend(rows)
        fit_info=json.loads((OUT/'sources'/s['id']/'opensim/complete.json').read_text())
        exported=DELIVERY/'opensim'/s['id'];exported.mkdir(parents=True,exist_ok=True)
        for name in ['model.osim','estimated_markers.trc','kinematics.mot','ik_setup.xml','complete.json']:
            shutil.copy2(OUT/'sources'/s['id']/'opensim'/name,exported/name)
        # Portable IK setup uses paths relative to its containing source directory.
        setup=exported/'ik_setup.xml';text=setup.read_text()
        text=text.replace(str(OUT/'sources'/s['id']/'opensim')+'/', '').replace(str(OUT/'sources'/s['id']/'opensim'),'.')
        setup.write_text(text)
        source_rows.append(dict(id=s['id'],domain=s['domain'],subject=s['subject'],publisherLabel=s['publisher_label'],
            frames=s['frames'],fps=s['fps'],width=s['width'],height=s['height'],poseSource=s['pose_path'],
            stabilization=s['stabilization'],referenceIds=[cid for cid,_ in reference],
            medianYaw=percentile(g['yaw'],50),opensim=fit_info,
            opensimModel='opensim/'+s['id']+'/model.osim',opensimMotion='opensim/'+s['id']+'/kinematics.mot',opensimMarkers='opensim/'+s['id']+'/estimated_markers.trc'))
    payload=clean(dict(createdAt=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        defaults=DEFAULTS,clips=all_rows,sources=source_rows,evidence=EVIDENCE,jointNames=NAMES,h36mNames=H36,edges3d=EDGES,
        scope='Research screening, not clinical approval. Thresholds below are engineering defaults, not published diagnostic cutoffs.',
        thresholdProvenance='Features are literature-motivated. Numerical thresholds are explicit prototype settings, frozen before publisher-label comparison.',
        protocol=json.loads((OUT/'protocol.json').read_text())))
    write(OUT/'features.json',payload);write(DELIVERY/'data.json',payload)
    write(OUT/'rule_defaults.json',dict(thresholds=DEFAULTS,evidence=EVIDENCE))
    status('features_complete',clips=len(all_rows),sources=len(source_rows))


if __name__=='__main__':build_features()

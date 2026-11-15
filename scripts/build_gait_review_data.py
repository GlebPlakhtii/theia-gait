"""Cached pose trajectories, conservative tracking QC and exploratory cycle references."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
from scipy.signal import savgol_filter,find_peaks

ROOT=Path(__file__).resolve().parents[1];WEB=ROOT/'results/gait_selected_paper_20261115';OUT=ROOT/'results/gait_review_upgrade_20261115'
OUT.mkdir(exist_ok=True)
REGIONS=['trunk','hips','shoulders','left_knee','right_knee','left_arm','right_arm']
GROUPS={'trunk':['left_hip','right_hip','left_shoulder','right_shoulder'],'hips':['left_hip','right_hip'],'shoulders':['left_shoulder','right_shoulder']}
for side in ['left','right']:
    GROUPS[side+'_knee']=[side+'_hip',side+'_knee',side+'_ankle'];GROUPS[side+'_arm']=[side+'_hip',side+'_shoulder',side+'_elbow']
def clean(a):
    if isinstance(a,np.ndarray):return clean(a.tolist())
    if isinstance(a,list):return [clean(x) for x in a]
    if isinstance(a,dict):return {k:clean(v) for k,v in a.items()}
    if isinstance(a,(np.floating,float)):return round(float(a),4) if np.isfinite(a) else None
    if isinstance(a,np.integer):return int(a)
    return a
def save(p,d):p.write_text(json.dumps(clean(d),separators=(',',':'),allow_nan=False))
def runs(ok,t):
    ii=np.flatnonzero(ok)
    return np.split(ii,np.flatnonzero((np.diff(ii)>1)|(np.diff(t[ii])>.12))+1)
def angle(a,b):
    n=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1)/np.maximum(n,1e-9),-1,1)))

def analyze(path,view):
    raw=json.loads(path.read_text());names=raw['joint_names'];p=np.array(raw['xy_pixels'],float);t=np.array(raw['timestamp_s']);q=np.array(raw['quality_mask'],bool)&(np.array(raw['confidence'])>=.5)&np.isfinite(p).all(2)
    idx={name:i for i,name in enumerate(names)};get=lambda n:p[:,idx[n]]
    hip=(get('left_hip')+get('right_hip'))/2;shoulder=(get('left_shoulder')+get('right_shoulder'))/2
    torso=np.linalg.norm(shoulder-hip,axis=1);scale=np.nanmedian(torso);scale=max(scale,1.)
    spikes=np.zeros(q.shape,bool)
    for j in range(len(names)):
        if len(t)<3:continue
        before=np.linalg.norm(p[1:-1,j]-p[:-2,j],axis=1);after=np.linalg.norm(p[1:-1,j]-p[2:,j],axis=1);bridge=np.linalg.norm(p[2:,j]-p[:-2,j],axis=1)
        spikes[1:-1,j]=(before>.25*scale)&(after>.25*scale)&(bridge<.15*scale)&q[:-2,j]&q[1:-1,j]&q[2:,j]&(np.diff(t)[:-1]<.12)&(np.diff(t)[1:]<.12)
    q&=~spikes
    # Invalidate grossly collapsed/elongated segments, without imposing bilateral symmetry.
    for side in ['left','right']:
        for aa,bb in [('hip','knee'),('knee','ankle'),('shoulder','elbow'),('elbow','wrist')]:
            a,b=idx[side+'_'+aa],idx[side+'_'+bb];length=np.linalg.norm(p[:,a]-p[:,b],axis=1);valid=q[:,a]&q[:,b]
            if valid.sum()<5:continue
            med=np.median(length[valid]);bad=valid&((length<max(1.,med*.3))|(length>med*2.5))
            q[bad,a]=False;q[bad,b]=False
    series={};cycles={}
    for region in REGIONS:
        joints=[idx[n] for n in GROUPS[region]];ok=q[:,joints].all(1)
        if region=='trunk':d=shoulder-hip;curve=np.degrees(np.arctan2(d[:,0],-d[:,1]));ok&=torso>3
        elif region in ['hips','shoulders']:
            a,b=[get(n) for n in GROUPS[region]];d=b-a;curve=np.degrees(np.arctan2(d[:,1],np.abs(d[:,0])));ok&=np.abs(d[:,0])>.15*scale
            if view!='front_rear':ok[:]=False
        elif region.endswith('_knee'):
            a,b,c=[get(n) for n in GROUPS[region]];curve=180-angle(a-b,c-b)
        else:
            a,b,c=[get(n) for n in GROUPS[region]];curve=angle(a-b,c-b)
        rawcurve=np.where(ok,curve,np.nan);smoothed=rawcurve.copy()
        for run in runs(ok,t):
            if len(run)>=5:smoothed[run]=savgol_filter(curve[run],5,2)
        if region.endswith(('_knee','_arm')):smoothed=np.clip(smoothed,0,180)
        center=np.mean(p[:,joints],axis=1);span=np.max(np.ptp(p[:,joints],axis=1),axis=1)
        crop_size=float(np.median(span[ok])*1.35) if ok.any() else scale*2
        series[region]=dict(raw=rawcurve,smooth=smoothed,ok=ok.tolist(),coverage=float(ok.mean()),crop_size=max(crop_size,scale*.85),joints=GROUPS[region])
        region_cycles=[]
        if view=='side' and region.endswith('_knee'):
            for run in runs(ok,t):
                if len(run)<10 or t[run[-1]]-t[run[0]]<1.2:continue
                fps=1/np.median(np.diff(t[run]));z=smoothed[run];amp=np.percentile(z,95)-np.percentile(z,5)
                if amp<8:continue
                peaks,_=find_peaks(z,distance=max(1,int(.5*fps)),prominence=max(6.,amp*.2))
                for aa,bb in zip(peaks[:-1],peaks[1:]):
                    a,b=int(run[aa]),int(run[bb]);period=t[b]-t[a]
                    if not .65<=period<=2.2:continue
                    values=np.interp(np.linspace(t[a],t[b],101),t[a:b+1],smoothed[a:b+1])
                    region_cycles.append(dict(start=float(t[a]),end=float(t[b]),period=float(period),curve=values,definition='Successive projected knee-bend peaks; not verified heel strikes.'))
        cycles[region]=region_cycles
    return dict(t=t,joint_names=names,joint_ok=q.tolist(),regions=series,cycles=cycles,torso_pixels=float(scale),spike_joint_frames=int(spikes.sum()),frames=len(t),pose_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def run():
    page=json.loads((WEB/'page_data.json').read_text());rows=json.loads((ROOT/'results/gait_from_scratch_v1/dataset/windows.json').read_text())
    protocol=dict(confidence=.5,isolated_jump='Both adjacent displacements >0.25 median torso, neighbors <0.15 torso apart; adjacent timestamps <0.12s',segment_length='Mask lengths below 0.3 or above 2.5 times within-clip median, or below 1px',smoothing='Savitzky-Golay 5 frames/order2, only contiguous reliable runs; never bridge missing observations',
        cycle='Side-view knee-bend peak to next peak; prominence max(6 degrees,0.2 robust range), robust range>=8 degrees, separation0.65–2.2s. Exploratory cycle markers, not clinical gait-phase labels.',
        reference='Original TCN training participants only; same anatomical side and camera view; period within20%; one median curve per participant, then10th–90th participant percentiles; minimum3people and6cycles. No test cases or amputation labels in fit.',normal_band=False)
    save(OUT/'trajectory_protocol.json',protocol)
    clips={};seen={}
    for c in page['clips']:
        if c['clip_id'] in clips:continue
        path=ROOT/'deliverables/Theia_Squat_Form_Lab_2026-11-14'/c['pose_url'].lstrip('/')
        clips[c['clip_id']]=analyze(path.resolve(),c['view'])
    training={}
    for row in rows:
        if row['split']=='train' and row['method']=='sapiens' and row['subject'] and row['view']=='side' and row['activity']=='walking' and row['environment']=='overground':
            training.setdefault(row['clip_id'],row)
    references=defaultdict(list)
    for cid,row in training.items():
        path=ROOT/row['metric_path'];path=path.with_name('viewer.json')
        if not path.exists():continue
        a=analyze(path,'side')
        for region,cc in a['cycles'].items():
            for cycle in cc:references[region].append(dict(**cycle,subject=row['subject'],clip_id=cid))
    supported=0;total=0
    for c in page['clips']:
        if c['kind']!='tcn':continue
        a=clips[c['clip_id']]
        for region,cc in a['cycles'].items():
            for cycle in cc:
                total+=1;ref=[x for x in references[region] if abs(x['period']/cycle['period']-1)<=.2]
                groups=defaultdict(list)
                for x in ref:groups[x['subject']].append(x['curve'])
                assert c['subject'] not in groups
                if len(groups)>=3 and len(ref)>=6:
                    medians=np.array([np.median(vals,axis=0) for vals in groups.values()]);lo,median,hi=np.percentile(medians,[10,50,90],axis=0)
                    cycle['reference']=dict(low=lo,median=median,high=hi,people=len(groups),cycles=len(ref),participant_ids=sorted(groups),period_match_fraction=.2)
                    cycle['outside_reference_fraction']=float(np.mean((cycle['curve']<lo)|(cycle['curve']>hi)));supported+=1
    lookup={w['key']:w for w in rows};outcomes=json.loads((ROOT/'results/gait_experiments_20261115_0502/splits/original/label_protocol.json').read_text())['outcomes']
    observations={}
    for c in page['clips']:
        if c['kind']=='tcn':
            for w in c['windows']:
                base=lookup[w['key']];observations[w['key']]=[{**outcomes[j],'value':value} for j,value in enumerate(base['value'])]
    summary=dict(unique_clips=len(clips),candidate_cycles=total,cycles_with_training_reference=supported,reference_training_clips=len(training),reference_people=sorted({x['subject'] for rr in references.values() for x in rr}),spike_joint_frames=sum(a['spike_joint_frames'] for a in clips.values()),clinical_validation=False)
    result=dict(protocol=protocol,summary=summary,clips=clips,tcn_observations=observations)
    save(WEB/'review_data.json',result);save(OUT/'trajectory_summary.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':run()

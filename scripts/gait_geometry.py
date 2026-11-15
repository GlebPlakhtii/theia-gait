"""View-gated image-plane gait observations; no diagnostic thresholds.

Missing anatomical axes, ground calibration, manual event validation and
prosthetic joint verification are explicit gates, never imputed measurements.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

VERSION = 'gait-observations-0.3.0'
NAMES = ['nose','left_eye','right_eye','left_ear','right_ear',
         'left_shoulder','right_shoulder','left_elbow','right_elbow',
         'left_wrist','right_wrist','left_hip','right_hip','left_knee',
         'right_knee','left_ankle','right_ankle','left_big_toe',
         'left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']
LABELS = ['Pelvic obliquity / hiking / contralateral drop','Lateral trunk lean',
 'Forward trunk lean / posterior pelvis proxy','Swing circumduction',
 'Knee flexion','Knee hyperextension / rapid extension','Vaulting / heel rise',
 'Foot contact pattern','Swing foot clearance','Foot progression / ankle motion',
 'Stance and step-time asymmetry','Step length / step-to pattern','Base of support',
 'Gait-cycle irregularity','Shoulder asymmetry / hiking','Arm swing','Forward head position']


def cross(a,b): return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def bend(hip,knee,ankle,direction=None):
    a=knee-hip;b=ankle-knee
    c=cross(a,b);d=np.sum(a*b,axis=-1)
    deg=np.degrees(np.arctan2(c,d))
    out=np.abs(deg) if direction is None else -direction*deg
    valid=(np.linalg.norm(a,axis=-1)>1e-6)&(np.linalg.norm(b,axis=-1)>1e-6)
    return np.where(valid,out,np.nan)


def unsigned_bend_3d(a,b):
    valid=(np.linalg.norm(a,axis=-1)>1e-6)&(np.linalg.norm(b,axis=-1)>1e-6)
    val=np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))
    return np.where(valid,val,np.nan)


def stats(x):
    x=np.asarray(x);good=x[np.isfinite(x)]
    if len(good)<3:return None
    lo,hi=np.percentile(good,[5,95])
    return dict(n=int(len(good)),valid_fraction=float(len(good)/len(x)),mean=float(np.mean(good)),
                median=float(np.median(good)),p05=float(lo),p95=float(hi),rom95=float(hi-lo),sd=float(np.std(good)))


def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def filtered(x,t):
    """Filter only contiguous valid runs; never bridge an occlusion or cut."""
    out=x.copy();dt=np.diff(t)
    if len(dt)<2 or not np.all(dt>0):raise ValueError('Non-monotonic native timestamps')
    fps=1/np.median(dt)
    # Retain raw observations for short or irregularly timed runs.
    if np.max(np.abs(dt-np.median(dt)))>0.05*np.median(dt):return out
    sos=butter(4,min(5.0,fps*.20),fs=fps,output='sos')
    for j in range(x.shape[1]):
        for lo,hi in runs(np.isfinite(x[:,j]).all(-1)):
            if hi-lo>=24:out[lo:hi,j]=sosfiltfilt(sos,x[lo:hi,j],axis=0)
    return out


def view_of(c):
    if c.get('context_review'):
        return c['context_review']['nominal_view'],'sampled_visual_context_review'
    label=c.get('view','').lower()
    if 'oblique' in label:return 'oblique','curation_manifest_nominal_view'
    if label=='side' or label.startswith('side '):return 'side','curation_manifest_nominal_view'
    if label in ('front','rear','front/rear'):return 'front_rear','curation_manifest_nominal_view'
    # Filenames provide a tentative label, never calibrated camera pose.
    source=c.get('original','').lower()
    if c['source']=='additional_input' and 'gait_side' in source:return 'side','filename_hint_unverified'
    if c['source']=='additional_input' and 'gait_front' in source:return 'front_rear','filename_hint_unverified'
    return 'unknown','no_verified_view_metadata'


def stable_segments(x,t):
    """Fixed scale per provisional segment; cuts at large image-scale changes.

    Engineering 15% tolerance is a QC parameter, not clinical evidence.
    It does not compensate perspective or establish straight walking.
    """
    get=lambda n:x[:,NAMES.index(n)]
    lengths=[]
    for s in ['left','right']:
        lengths.append(np.linalg.norm(get(s+'_hip')-get(s+'_knee'),axis=-1)+
                       np.linalg.norm(get(s+'_knee')-get(s+'_ankle'),axis=-1))
    lengths=np.array(lengths).T
    with np.errstate(invalid='ignore'):
        count=np.isfinite(lengths).sum(1)
        scale=np.divide(np.nansum(lengths,axis=1),count,out=np.full(len(t),np.nan),where=count>0)
    # One-second median reduces apparent scale fluctuations due to knee flexion.
    width=max(1,round(.5/np.median(np.diff(t))))
    smooth=np.array([np.nanmedian(scale[max(0,i-width):min(len(t),i+width+1)])
                     if np.isfinite(scale[max(0,i-width):min(len(t),i+width+1)]).any() else np.nan for i in range(len(t))])
    boundaries=[0];anchor=None
    for i,v in enumerate(smooth):
        if not np.isfinite(v) or v<=1:continue
        if anchor is None:anchor=v
        if abs(v/anchor-1)>.15 and t[i]-t[boundaries[-1]]>=.5:
            boundaries.append(i);anchor=v
    boundaries.append(len(t));out=[]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        z=scale[lo:hi];z=z[np.isfinite(z)&(z>1)]
        out.append((lo,hi,float(np.median(z)) if len(z) else None))
    return out


def candidate_cycles(signal,t,direction,scale):
    """Relative-ankle extrema are proposals, not validated foot contacts."""
    if direction is None or scale is None:return []
    out=[];fps=1/np.median(np.diff(t))
    for lo,hi in runs(np.isfinite(signal)):
        if hi-lo<max(20,round(.8*fps)):continue
        peaks,_=find_peaks(direction*signal[lo:hi],distance=max(1,round(.35*fps)),prominence=.06*scale)
        peaks=peaks+lo
        for a,b in zip(peaks[:-1],peaks[1:]):
            if .4<=t[b]-t[a]<=3.5:
                out.append(dict(start_frame=int(a),end_frame=int(b),period_s=float(t[b]-t[a]),
                    status='unvalidated_extremum_cycle',contact_event_validated=False))
    return out


def measure_segment(x,t,c,l0):
    view,basis=view_of(c);get=lambda n:x[:,NAMES.index(n)]
    hip=(get('left_hip')+get('right_hip'))/2;shoulder=(get('left_shoulder')+get('right_shoulder'))/2
    torso=shoulder-hip;series={};proposals={}
    # Image y is converted to upward y by the caller.
    degree=lambda a:np.degrees(np.arctan2(a[:,0],a[:,1]))
    norm=lambda v:v/l0 if l0 is not None else np.full_like(v,np.nan)
    series['trunk_image_inclination_deg']=degree(torso)
    series['trunk_horizontal_offset_L0']=norm(torso[:,0])
    delta=get('left_hip')-get('right_hip')
    series['left_up_hip_line_deg']=np.degrees(np.arctan2(delta[:,1],np.abs(delta[:,0])))
    delta_sh=get('left_shoulder')-get('right_shoulder')
    series['left_up_shoulder_line_deg']=np.degrees(np.arctan2(delta_sh[:,1],np.abs(delta_sh[:,0])))
    series['ankle_horizontal_separation_L0']=norm(np.abs(get('left_ankle')[:,0]-get('right_ankle')[:,0]))
    direction=None
    fd=np.concatenate([((get(s+'_big_toe')+get(s+'_small_toe'))/2-get(s+'_heel'))[:,0] for s in ('left','right')])
    good=fd[np.isfinite(fd)]
    if view=='side' and l0 and len(good)>=10 and abs(np.median(good))>.025*l0:
        proposed=int(np.sign(np.median(good)))
        if np.mean(np.sign(good)==proposed)>=.85:direction=proposed
    for side in ('left','right'):
        h,k,a=[get(side+'_'+n) for n in ('hip','knee','ankle')]
        e=get(side+'_heel');f=(get(side+'_big_toe')+get(side+'_small_toe'))/2
        sh,el,wr,ear=[get(side+'_'+n) for n in ('shoulder','elbow','wrist','ear')]
        series[side+'_unsigned_projected_knee_bend_deg']=bend(h,k,a)
        series[side+'_heel_minus_forefoot_height_L0']=norm(e[:,1]-f[:,1])
        series[side+'_ankle_minus_hip_horizontal_L0']=norm(a[:,0]-h[:,0])
        series[side+'_shoulder_minus_hip_height_L0']=norm(sh[:,1]-h[:,1])
        series[side+'_wrist_minus_shoulder_horizontal_L0']=norm(wr[:,0]-sh[:,0])
        series[side+'_ear_minus_shoulder_horizontal_L0']=norm(ear[:,0]-sh[:,0])
        upper=el-sh;down=hip-shoulder
        series[side+'_upper_arm_to_trunk_unsigned_deg']=np.degrees(np.arctan2(np.abs(cross(upper,down)),np.sum(upper*down,axis=-1)))
        if direction is not None:
            series[side+'_candidate_signed_knee_bend_deg']=bend(h,k,a,direction)
            series[side+'_projected_foot_pitch_deg']=np.degrees(np.arctan2(f[:,1]-e[:,1],direction*(f[:,0]-e[:,0])))
            proposals[side]=candidate_cycles(a[:,0]-hip[:,0],t,direction,l0)
    summary={k:stats(v) for k,v in series.items()}
    paired={}
    if view=='side':
        for family,suffix in [('knee','_unsigned_projected_knee_bend_deg'),('arm','_wrist_minus_shoulder_horizontal_L0')]:
            left,right=[series[s+suffix] for s in ('left','right')];common=np.isfinite(left)&np.isfinite(right)
            if common.sum()>=3:
                ls=stats(np.where(common,left,np.nan));rs=stats(np.where(common,right,np.nan))
                paired[family]=dict(valid_fraction=float(common.mean()),common_frames=int(common.sum()),
                    left_rom95=ls['rom95'],right_rom95=rs['rom95'],
                    mean_rom95=(ls['rom95']+rs['rom95'])/2,
                    mean_p95=(ls['p95']+rs['p95'])/2,mean_sd=(ls['sd']+rs['sd'])/2,
                    signed_left_minus_right_rom95=ls['rom95']-rs['rom95'],absolute_rom95_difference=abs(ls['rom95']-rs['rom95']),
                    side_meaning='Anatomical pose-label left/right; prosthetic side not inferred',
                    unit='deg' if family=='knee' else 'L0')
    walking=c.get('activity') in ('walk','walking','walking_with_walker','walking_with_weights','curb_walking')
    steady=walking and c.get('environment') in ('overground','treadmill','parallel_bars') and not c.get('contains_turns_or_transitions',False)
    metrics=[]
    for i,label in enumerate(LABELS,1):
        m=dict(metric_id=f'M{i:02d}',name=label,measurement_method='none',measurement_status='not_evaluable',
               detection_status='not_evaluable',value=None,unit=None,reason_codes=[],uncertainty=None,
               cause='unknown',valid_cycle_count=0,repeated_cycle_count=0,observations={})
        if not steady:
            m.update(measurement_status='not_applicable',detection_status='not_applicable',
                     reason_codes=['not_straight_level_walking_scope']);metrics.append(m);continue
        required='front_rear' if i in (1,2,4,13,15) else 'side' if i in (3,5,6,7,8,9,12,16,17) else None
        if required and view!=required:
            m['reason_codes']=['view_incompatible_or_unverified'];metrics.append(m);continue
        fields={1:['left_up_hip_line_deg'],2:['trunk_image_inclination_deg','trunk_horizontal_offset_L0'],
            3:['trunk_image_inclination_deg','trunk_horizontal_offset_L0'],
            4:[s+'_ankle_minus_hip_horizontal_L0' for s in ('left','right')],
            5:[s+'_unsigned_projected_knee_bend_deg' for s in ('left','right')],
            6:[s+'_candidate_signed_knee_bend_deg' for s in ('left','right')],
            7:[s+'_heel_minus_forefoot_height_L0' for s in ('left','right')],
            8:[s+'_projected_foot_pitch_deg' for s in ('left','right')],
            13:['ankle_horizontal_separation_L0'],
            15:['left_up_shoulder_line_deg']+[s+'_shoulder_minus_hip_height_L0' for s in ('left','right')],
            16:[s+'_wrist_minus_shoulder_horizontal_L0' for s in ('left','right')],
            17:[s+'_ear_minus_shoulder_horizontal_L0' for s in ('left','right')]}.get(i,[])
        m['observations']={k:summary[k] for k in fields if summary.get(k) and summary[k]['valid_fraction']>=.7}
        m['observation_units']={k:('deg' if k.endswith('_deg') else 'L0') for k in m['observations']}
        blockers={1:['gait_phase_unvalidated','hip_landmarks_not_ASIS'],4:['swing_phase_unvalidated'],
            6:['signed_anatomical_axis_unvalidated','prosthetic_joint_geometry_unverified'],
            7:['ground_plane_missing','flat_foot_baseline_missing','midstance_unvalidated'],
            8:['contact_events_unvalidated','ground_plane_missing'],9:['ground_plane_missing','sole_clearance_unvalidated'],
            10:['ground_homography_missing','anatomical_axes_unvalidated'],11:['contact_events_unvalidated'],
            12:['contact_events_unvalidated','ground_scale_missing'],13:['double_support_unvalidated'],
            14:['valid_cycles_unavailable'],15:['phase_specific_hiking_unvalidated'],
            16:['arm_gait_phase_unvalidated'],17:['head_rotation_not_verified']}
        m['reason_codes']=blockers.get(i,[])+['clinical_thresholds_and_error_floor_unvalidated']
        if m['observations']:
            m['measurement_method']='2d_projected'
            # Only phase-independent image-plane statistics are numeric outcomes.
            if i in (2,3,5,15,16,17):
                m['measurement_status']='uncertain';m['value']=m['observations']
                m['unit']='degrees and/or fixed projected leg-length units; see field names'
                m['reason_codes']+=['nominal_view_only','camera_roll_unverified','projection_uncorrected']
                m['uncertainty']={'type':'unquantified','bounds':None,'reason':'Pose-coordinate error and camera/anatomical calibration not validated for this video'}
            else:m['reason_codes']+=['raw_proxy_available_but_requested_phase_metric_unavailable']
        else:m['reason_codes']+=['required_observations_missing_or_quality_insufficient']
        if i==5 and c['cohort']=='with_amputation':m['reason_codes']+=['prosthetic_mechanical_knee_correspondence_unverified']
        if i==16 and (c.get('environment')=='parallel_bars' or c.get('context_review',{}).get('visible_support','none')!='none'):
            m['reason_codes']+=['assisted_or_loaded_arm_movement_not_free_arm_swing']
        metrics.append(m)
    return dict(view=view,view_basis=basis,direction_candidate=direction,
                direction_basis='toe-to-heel image vector; anatomical sign not validated',
                metrics=metrics,series=series,cycle_proposals=proposals,paired_geometry=paired)

"""Extract all 17 metric-family records, reusing publisher/native pose output."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import warnings
import argparse
import time
import fcntl
import numpy as np
from gait_geometry import NAMES, LABELS, VERSION, filtered, stable_segments, measure_segment

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/prepared/gait_curated_20261114'
OUT=ROOT/'results/gait_metrics_20261114'
POSE_ROOT=OUT
FORCE_SAPIENS=False
SPEC=Path('/home/hlak/.codex/attachments/4fbf179e-f7e0-4a13-95af-1a390a221113/Pasted text.txt')


def input_signature(c,code_hash):
    paths=[]
    if c.get('publisher_pose') and not FORCE_SAPIENS:
        paths=[BASE/c['publisher_pose']]
    else:
        for variant in ['1b','5b','0.4b']:
            folder=POSE_ROOT/f'poses_sapiens2_{variant}'/c['clip_id']
            if (folder/'complete.json').exists():
                paths=[folder/'raw_308.npz',folder/'complete.json'];break
    stamps=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps(dict(clip=c,code=code_hash,poses=stamps,force_sapiens=FORCE_SAPIENS),sort_keys=True).encode()).hexdigest()


def clean(obj):
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,(np.floating,float)):return round(float(obj),6) if np.isfinite(obj) else None
    if isinstance(obj,np.integer):return int(obj)
    return obj


def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(obj),separators=(',',':'),allow_nan=False));tmp.replace(path)


def load_pose(c):
    if c.get('publisher_pose') and not FORCE_SAPIENS:
        path=BASE/c['publisher_pose'];a=np.load(path,allow_pickle=False)
        assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==c['publisher_pose_sha256']
        names=a['joint_names'].tolist();idx=[names.index(n) for n in NAMES]
        assert np.array_equal(a['source_frame'],np.arange(c['source_first_frame'],c['source_end_frame_exclusive']))
        raw=a['keypoints'][:,idx];t=a['source_time_s']-a['source_time_s'][0]
        return raw[:,:,:2],raw[:,:,2],t,dict(name='ProGait publisher RTMW-assisted manually corrected 23-keypoint annotations',
            method='publisher_annotation',path=str(path),sha256=c['publisher_pose_sha256'],independent_manual_ground_truth=False)
    for variant in ['1b','5b','0.4b']:
        base=POSE_ROOT/f'poses_sapiens2_{variant}';path=base/c['clip_id']/'raw_308.npz'
        if not (path.parent/'complete.json').exists():continue
        a=np.load(path);schema=json.loads((base/'joint_schema.json').read_text())['body_feet_23']
        assert np.array_equal(a['source_frame_index'],np.arange(c['clip_frames']))
        idx={n:int(i) for i,n in schema.items()};indices=[idx[n] for n in NAMES]
        for attempt in range(5):
            try:run=json.loads((base/'run.json').read_text());break
            except json.JSONDecodeError:
                if attempt==4:raise
                time.sleep(.1)
        return a['keypoints'][:,indices],a['scores'][:,indices],a['timestamp_s'],dict(name=run['model'],
            method='sapiens_inference',path=str(path),sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
            weight_sha256=run['weight_sha256'],vendor_revision=run['vendor_revision'],precision=run['precision'],flip_test=run['flip_test'])
    return None


def apply_visual_review(segment,review,provenance):
    if (review.get('status')!='regional_pose_failure'
        or review.get('weight_sha256')!=provenance.get('weight_sha256')):return
    affected=set(review.get('affected_metric_ids',[]))
    bad_leg_scale=review.get('normalization_scale_unreliable',bool(affected & {f'M{i:02d}' for i in range(4,15)}))
    for m in segment['metrics']:
        if m['metric_id'] in affected:
            m['measurement_status']='not_evaluable';m['value']=None
            m['reason_codes']+=[review['reason_code'],'raw_observations_retained_for_debug_not_comparison']
        if bad_leg_scale:
            components={k:'leg_scale_unreliable_after_visual_pose_failure'
                        for k,unit in m.get('observation_units',{}).items() if unit=='L0'}
            if not components:continue
            m['excluded_observation_components']=components
            m['reason_codes']+=['leg_scale_unreliable_after_visual_pose_failure','normalized_raw_observations_retained_for_debug_only']
            if isinstance(m.get('value'),dict):
                m['value']={k:v for k,v in m['value'].items() if k not in components}
                if not m['value']:m['measurement_status']='not_evaluable';m['value']=None


def apply_visibility_masks(valid,review,provenance):
    """Mask documented occlusion intervals without changing raw pose coordinates."""
    if review.get('weight_sha256')!=provenance.get('weight_sha256'):return 0
    before=int(valid.sum())
    for mask in review.get('manual_visibility_masks',[]):
        lo,hi=mask['first_clip_frame'],mask['end_clip_frame_exclusive']
        assert isinstance(lo,int) and isinstance(hi,int) and 0<=lo<hi<=len(valid)
        assert mask['joint_names'] and all(n in NAMES for n in mask['joint_names'])
        valid[lo:hi,[NAMES.index(n) for n in mask['joint_names']]]=False
    return before-int(valid.sum())


def review_evidence_frames(review,provenance,frame_count):
    """Display exact failure frames and mask boundaries for this pose model."""
    if review.get('weight_sha256')!=provenance.get('weight_sha256'):return []
    evidence=list(review.get('failure_clip_frames',[]))
    for mask in review.get('manual_visibility_masks',[]):
        evidence.extend([mask['first_clip_frame'],mask['end_clip_frame_exclusive']-1])
    assert all(isinstance(f,int) and 0<=f<frame_count for f in evidence),'Visual review evidence frame outside clip'
    return sorted(set(evidence))


def process(c):
    raw=load_pose(c)
    metadata={k:c.get(k) for k in ['clip_id','source','cohort','cohort_basis','cohort_evidence_level','needs_cohort_confirmation',
        'subject','split_group','video','video_absolute','sha256','original','source_sha256','source_first_frame','source_end_frame_exclusive',
        'activity','environment','view','width','height','fps','clip_frames','clip_duration_s','context_review','pose_visual_review']}
    result=dict(**metadata,feature_version=VERSION,clinical_form_label=None,threshold_version=None,reference_group=None,
        clinical_detection_available=False,metric_families=17,created_at=datetime.now(timezone.utc).isoformat())
    if raw is None:
        result.update(status='waiting_for_pose',segments=[],metrics=[dict(metric_id=f'M{i:02d}',name=label,
            measurement_method='none',measurement_status='not_evaluable',detection_status='not_evaluable',value=None,
            reason_codes=['pose_inference_pending'],cause='unknown') for i,label in enumerate(LABELS,1)])
        return result
    xy,score,t,provenance=raw
    assert xy.shape==(c['clip_frames'],23,2) and score.shape==xy.shape[:2]
    assert len(t)==len(xy) and np.all(np.diff(t)>0)
    valid=np.isfinite(xy).all(-1)&np.isfinite(score)&(score>=.3)&(xy[:,:,0]>=0)&(xy[:,:,0]<c['width'])&(xy[:,:,1]>=0)&(xy[:,:,1]<c['height'])
    manual_masked=apply_visibility_masks(valid,c.get('pose_visual_review') or {},provenance)
    x=xy.copy().astype(float);x[~valid]=np.nan;x[:,:,1]*=-1
    # Gross one-frame discontinuities are explicit missing observations. This
    # engineering screen is not proof of correct tracking or limb visibility.
    hip=(x[:,11]+x[:,12])/2
    leg_lengths=np.stack([np.linalg.norm(x[:,11]-x[:,13],axis=-1)+np.linalg.norm(x[:,13]-x[:,15],axis=-1),
        np.linalg.norm(x[:,12]-x[:,14],axis=-1)+np.linalg.norm(x[:,14]-x[:,16],axis=-1)],axis=1)
    reference_scale=float(np.nanmedian(leg_lengths)) if np.isfinite(leg_lengths).any() else None
    jump_mask=np.zeros(valid.shape,dtype=bool)
    if reference_scale and reference_scale>1:
        speed=np.linalg.norm(np.diff(x,axis=0),axis=-1)/reference_scale/np.diff(t)[:,None]
        jump_mask[1:]|=speed>12;jump_mask[:-1]|=speed>12
        valid&=~jump_mask;x[~valid]=np.nan
    smoothed=filtered(x,t)
    spans=stable_segments(smoothed,t);segments=[];timeseries={};curves=[]
    for si,(lo,hi,scale) in enumerate(spans):
        if hi-lo<3:continue
        context=dict(c);flags=[]
        view=c.get('context_review',{}).get('nominal_view',c.get('view',''))
        if view in ('front','rear','front/rear','front_rear') and scale:
            delta=smoothed[lo:hi,NAMES.index('left_hip'),0]-smoothed[lo:hi,NAMES.index('right_hip'),0]
            strong=delta[np.isfinite(delta)&(np.abs(delta)>.1*scale)]
            if len(strong)>=10 and min(np.mean(strong>0),np.mean(strong<0))>.1:
                context['contains_turns_or_transitions']=True;flags.append('frontal_hip_order_reversal_possible_turn_or_side_swap')
        d=measure_segment(smoothed[lo:hi],t[lo:hi],context,scale)
        review=c.get('pose_visual_review') or {}
        apply_visual_review(d,review,provenance)
        series=d.pop('series')
        for name,values in series.items():
            if name not in timeseries:timeseries[name]=np.full(len(x),np.nan)
            timeseries[name][lo:hi]=values
        # Curves use candidate extremum-to-extremum cycles, explicitly not IC-to-IC gait phases.
        for side,cycles in d['cycle_proposals'].items():
            key=side+'_unsigned_projected_knee_bend_deg'
            for cycle in cycles:
                a,b=cycle['start_frame'],cycle['end_frame']
                y=series[key][a:b+1];tt=t[lo+a:lo+b+1]
                if np.isfinite(y).all():
                    curves.append(dict(segment=si,side=side,start_frame=lo+a,end_frame=lo+b,
                        phase_basis='unvalidated_ankle_extremum_cycle_not_contact_gait_phase',
                        percent=np.linspace(0,100,101),knee_bend_deg=np.interp(np.linspace(tt[0],tt[-1],101),tt,y)))
        segments.append(dict(segment_id=si,first_clip_frame=lo,end_clip_frame_exclusive=hi,
            start_time_s=float(t[lo]),end_time_s=float(t[hi-1]),fixed_L0_pixels=scale,
            L0_basis='Fixed median projected left/right hip-knee-ankle chain length in this segment; intact side and metric scale unknown',
            segmentation_basis='15% smoothed projected leg-scale change; view stability and turns need review',
            quality_flags=flags,**d))
    folder=OUT/'clips'/c['clip_id'];folder.mkdir(parents=True,exist_ok=True)
    temporary=folder/'observations.npz.tmp'
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,timestamp_s=t,source_frame=np.arange(len(t))+c['source_first_frame'],
            xy_pixels=xy.astype(np.float32),confidence=score.astype(np.float32),quality_mask=valid,
            filtered_xy_up=smoothed.astype(np.float32),**{k:v.astype(np.float32) for k,v in timeseries.items()})
    temporary.replace(folder/'observations.npz')
    step=max(1,round(c['fps']/10));idx=np.unique(np.r_[np.arange(0,len(t),step),len(t)-1])
    review=c.get('pose_visual_review') or {}
    evidence=review_evidence_frames(review,provenance,len(t))
    if evidence:idx=np.unique(np.r_[idx,evidence]).astype(int)
    write(folder/'viewer.json',dict(joint_names=NAMES,timestamp_s=t[idx],clip_frame=idx,native_timestamp_s=t,
        xy_pixels=xy[idx],confidence=score[idx],quality_mask=valid[idx],
        observations={k:v[idx] for k,v in timeseries.items()},display_only_sampling='approximately 10 Hz plus exact reviewed failure frames and occlusion boundaries; calculations use all native frames'))
    write(folder/'candidate_curves.json',curves)
    result.update(status='observations_extracted',pose=provenance,segments=segments,
        processed_frames=len(t),valid_observation_fraction=float(valid.mean()),
        jump_masked_joint_observations=int(jump_mask.sum()),
        manual_visibility_masked_joint_observations=manual_masked,
        valid_contact_gait_cycles=0,candidate_extremum_cycles=len(curves),
        curve_path=f'clips/{c["clip_id"]}/candidate_curves.json',viewer_path=f'clips/{c["clip_id"]}/viewer.json',
        observation_path=f'clips/{c["clip_id"]}/observations.npz',
        limitations=['No ground or camera calibration supplied; no cm, ground-relative clearance or foot-progression angle.',
            'No validated foot-contact annotations; extremum cycles are proposals, not stance/swing ground truth.',
            'Pose confidence is uncalibrated; the 0.3 score gate is an engineering filter.',
            'Clinical detection thresholds, prosthetic mechanical-joint correspondence and error bounds are not validated.',
            'Native source timing is retained; real-time playback speed is not independently verified.',
            'Image-plane angular change does not eliminate out-of-plane movement or depth ambiguity.'])
    return result


def main():
    global OUT,FORCE_SAPIENS
    parser=argparse.ArgumentParser();parser.add_argument('--harmonized',action='store_true');args=parser.parse_args()
    if args.harmonized:
        OUT=OUT/'harmonized';FORCE_SAPIENS=True
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'clips').mkdir(exist_ok=True)
    lock=(OUT/'.extract.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    shutil.copyfile(SPEC,OUT/'user_specification.txt')
    clips=json.loads((BASE/'manifest.json').read_text())['clips'];records=[]
    contexts={}
    for name in ['additional_context_review.json','progait_subset_context_review.json']:
        contexts.update(json.loads((POSE_ROOT/name).read_text())['clips'])
    reviews=json.loads((POSE_ROOT/'pose_visual_review.json').read_text())['clips'] if (POSE_ROOT/'pose_visual_review.json').exists() else {}
    code_hash=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('gait_geometry.py').read_bytes()).hexdigest()
    if args.harmonized:
        selected=set(json.loads((POSE_ROOT/'harmonization_subset.json').read_text())['clips']);clips=[c for c in clips if c['clip_id'] in selected]
    for c in clips:
        if c['clip_id'] in reviews:c['pose_visual_review']=reviews[c['clip_id']]
        if c['clip_id'] in contexts:
            c['context_review']=contexts[c['clip_id']]
            c['contains_turns_or_transitions']=c['context_review']['contains_turns_or_transitions'] or c['context_review']['scene_edit_visible']
            c['view']={'front_rear':'front/rear'}.get(c['context_review']['nominal_view'],c['context_review']['nominal_view'])
    for i,c in enumerate(clips):
        folder=OUT/'clips'/c['clip_id'];folder.mkdir(exist_ok=True)
        signature=input_signature(c,code_hash)
        path=folder/'metrics.json';r=json.loads(path.read_text()) if path.exists() else None
        artifacts_ok=r and (r['status']=='waiting_for_pose' or all((folder/n).exists() for n in ['observations.npz','viewer.json','candidate_curves.json']))
        if not artifacts_ok or r.get('input_signature')!=signature:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='All-NaN slice encountered')
                r=process(c)
            r['input_signature']=signature;r['extraction_code_sha256']=code_hash
            write(path,r)
        records.append({k:v for k,v in r.items() if k not in ('segments','metrics','limitations')})
        if (i+1)%50==0:print(f'Metrics {i+1}/{len(clips)}; completed {sum(x["status"]=="observations_extracted" for x in records)}',flush=True)
    totals=Counter(r['status'] for r in records)
    catalog=dict(version=VERSION,extraction_code_sha256=code_hash,updated_at=datetime.now(timezone.utc).isoformat(),counts=dict(totals),clips=records)
    write(OUT/'catalog.json',catalog)
    write(OUT/'processing_summary.json',dict(**dict(totals),total_clips=len(records),
        total_processed_frames=sum(r.get('processed_frames',0) for r in records),
        specification_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        clinically_validated_metrics=0,ground_calibrated_metrics=0,contact_validated_cycles=0,
        note='Complete output records include explicit unavailability. Extracted observations do not mean every requested anatomical metric is measurable.'))
    print(json.dumps(dict(totals)),flush=True)


if __name__=='__main__':main()

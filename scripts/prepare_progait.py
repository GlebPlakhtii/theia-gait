"""Decode every ProGait RGB, align publisher poses, and cut short doctor clips.

Publisher poses are RTMW-assisted annotations, NOT new Sapiens predictions.
Reference-derived crops are explicitly identified in the separate inference queue.
"""
import concurrent.futures as cf
from collections import defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw/ProGait'
PREP = ROOT/'data/prepared/progait_20261114'
OUT = ROOT/'deliverables/Theia_Doctor_ProGait_2026-11-14'
AUDIT = ROOT/'audit/progait_20261114'
PACKAGE_ID = 'theia-doctor-progait-20261114-v1'
NAMES = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
         'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
         'left_knee','right_knee','left_ankle','right_ankle','left_big_toe','left_small_toe',
         'left_heel','right_big_toe','right_small_toe','right_heel']


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def angles(a):
    output={}
    for side, ids in [('left',[11,13,15]),('right',[12,14,16])]:
        x=a[:,ids,:];u=x[:,0,:2]-x[:,1,:2];v=x[:,2,:2]-x[:,1,:2]
        denom=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        good=(x[:,:,2].min(axis=1)>=.3)&(denom>0)&np.isfinite(x).all(axis=(1,2))
        result=np.full(len(x),np.nan)
        result[good]=np.degrees(np.arccos(np.clip((u[good]*v[good]).sum(axis=1)/denom[good],-1,1)))
        output[side]=result
    return output


def process(meta):
    cid='PG_'+meta['subset']+'_'+meta['ID']
    done=PREP/cid/'complete.json'
    if done.exists():return json.loads(done.read_text())
    video=RAW/meta['file_name']
    kp_path=RAW/'annotations'/meta['subset']/(meta['ID']+'_keypoints.npy.gz')
    while not video.exists() or not kp_path.exists():
        status=json.loads((AUDIT/'acquisition_status.json').read_text())
        if status['status'] in ['complete','failed']:
            raise ValueError('Missing required download: '+cid)
        time.sleep(3)
    with gzip.open(kp_path,'rb') as f:kp=np.load(f,allow_pickle=False)
    fmap=[]
    with av.open(str(video)) as container:
        stream=container.streams.video[0];stream.thread_count=1
        width,height=stream.width,stream.height;rate=Fraction(stream.average_rate)
        for i,frame in enumerate(container.decode(video=0)):
            pts=float(frame.pts*frame.time_base)
            if i==0:first_pts=pts
            fmap.append(dict(source_frame_index=i,source_pts=frame.pts,source_pts_s=pts,timestamp_s=pts-first_pts))
    n=len(fmap)
    reference_shape=list(kp.shape)
    aligned=kp.shape==(n,23,3) and np.isfinite(kp).all()
    reference_problem=None if aligned else f'Publisher pose shape {reference_shape} does not align with {n} decoded RGB frames, or contains nonfinite values. Overlay withheld; no offset assumed.'
    if not aligned:
        # Internal invalid sentinel only; never export these zeros as observed poses.
        kp=np.zeros((n,23,3),dtype=np.float32)
    if not np.allclose([r['timestamp_s'] for r in fmap],np.arange(n)/float(rate),atol=1e-5,rtol=0):
        raise ValueError('Nonconstant native cadence: '+cid)
    save_csv(PREP/cid/'frames.csv',fmap)
    # Pose scores are model outputs; they are not clinical certainty or visibility probabilities.
    valid=(kp[:,:,2]>=.3)&(kp[:,:,0]>=0)&(kp[:,:,0]<width)&(kp[:,:,1]>=0)&(kp[:,:,1]<height)
    roi={};missing_roi_frames=[]
    for i in range(n):
        xy=kp[i,valid[i],:2]
        if len(xy)<8:
            missing_roi_frames.append(i)
            continue
        low=xy.min(axis=0);high=xy.max(axis=0);pad=np.maximum((high-low)*.15,20)
        box=np.r_[np.maximum(low-pad,0),np.minimum(high+pad,[width,height])]/[width,height,width,height]
        roi[str(i)]=box.tolist()
    max_frames=math.floor(float(rate)*8)
    cuts=list(range(0,n,max_frames))
    folder=OUT/'clips_original';folder.mkdir(parents=True,exist_ok=True)
    expected=[folder/(cid+f'_{i+1:03d}.mp4') for i in range(len(cuts))]
    for start,path in zip(cuts,expected):
        end=min(start+max_frames,n)
        # Explicit native-frame trim avoids keyframe-dependent segment boundaries.
        part=path.with_suffix('.part.mp4')
        args=['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(video),'-map','0:v:0',
              '-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
              '-frames:v',str(end-start),'-an','-c:v','h264_nvenc','-preset','p4',
              '-rc','vbr','-cq','20','-b:v','0','-bf','0','-pix_fmt','yuv420p',
              '-fps_mode','passthrough','-movflags','+faststart','-y',str(part)]
        subprocess.run(args,check=True,capture_output=True)
        part.replace(path)
    angle_input=kp.copy();angle_input[:,:,2]=np.where(valid,kp[:,:,2],0)
    signals=angles(angle_input);records=[]
    for number,(start,path) in enumerate(zip(cuts,expected),1):
        end=min(start+max_frames,n);clip_id=cid+f'_{number:03d}'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,avg_frame_rate,width,height','-of','json',str(path)]))['streams'][0]
        if int(probe['nb_read_frames'])!=end-start or Fraction(probe['avg_frame_rate'])!=rate:
            raise ValueError('Encoded cut frame count/rate mismatch: '+clip_id)
        metrics={}
        for side in ['left','right']:
            a=signals[side][start:end];a=a[np.isfinite(a)]
            for name,v in [('min',a.min() if len(a) else None),('max',a.max() if len(a) else None),
                           ('excursion',np.ptp(a) if len(a) else None)]:
                metrics[f'{side}_projected_knee_{name}_deg']=round(float(v),1) if v is not None else None
        metrics['score_gate_coverage']=round(float(valid[start:end,11:17].all(axis=1).mean()),4)
        pose_data=dict(clip_id=clip_id,width=width,height=height,fps=float(rate),joint_names=NAMES,
                       provenance='Publisher RTMW-assisted pose annotations; manually corrected subset; not new Sapiens2 output',
                       frames=np.round(kp[start:end],3).reshape(end-start,69).tolist())
        pose_file=OUT/'pose_data'/(clip_id+'.js');pose_file.parent.mkdir(exist_ok=True)
        if aligned:pose_file.write_text('window.THEIA_POSE='+json.dumps(pose_data,separators=(',',':'),allow_nan=False)+';\n')
        records.append(dict(clip_id=clip_id,source_id=cid,domain='clinical_prosthetic_gait',
            exercise='walking',automatic_type='gait_window',source_start_s=start/float(rate),
            source_last_s=(end-1)/float(rate),source_first_frame=start,source_last_frame=end-1,
            frames=end-start,fps=float(rate),video_duration_s=(end-start)/float(rate),
            core_start_s=0,core_end_s=(end-start-1)/float(rate),candidate_bottom_s=None,
            cycle_duration_s=None,descent_s=None,ascent_s=None,automatic_measurements=metrics,
            original_video=str(path.relative_to(OUT)),pose_video=None,pose_data=str(pose_file.relative_to(OUT)) if aligned else None,
            pose_provenance=pose_data['provenance'] if aligned else reference_problem,doctor_conclusion=None,doctor_notes='',
            subject_id=meta['ID'].split('_')[0],trial_id='_'.join(meta['ID'].split('_')[:3]),
            view='frontal' if '_f' in meta['ID'] else 'sagittal',subset=meta['subset'],
            source_url='https://huggingface.co/datasets/ericyxy98/ProGait',
            context='Prosthesis walking trial. Source privacy blur preserved. Gait reference text is withheld from this review page.'+(' '+reference_problem if reference_problem else '')))
    with video.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    queue=dict(clip_id=cid,local_path=str(video),sha256=sha,start_s=0,end_s=n/float(rate),
        frame_map=str(PREP/cid/'frames.csv'),status='reference_roi_ready_for_independent_model_comparison',
        transformations='native RGB; no rotation/crop/resampling; reference-derived target ROI, not a detection benchmark')
    result=dict(clip_id=cid,frames=n,rate=str(rate),width=width,height=height,source_sha256=sha,
        records=records,queue=queue,roi={'bbox_by_source_frame':roi},reference_frame_alignment='native RGB count matches publisher pose array' if aligned else 'unverified; overlay withheld',
        publisher_reference_shape=reference_shape,reference_problem=reference_problem,
        annotation_source='Publisher RTMW-assisted annotations; not entirely manual ground truth',
        missing_reference_roi_frames=missing_roi_frames)
    write(done,result)
    return result


def main():
    PREP.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    metadata=[json.loads(line) for line in (AUDIT/'metadata.jsonl').read_text().splitlines()]
    assert len(metadata)==412 and len({(m['subset'],m['ID']) for m in metadata})==412
    strata=defaultdict(list)
    for m in metadata:strata[(m['ID'].split('_')[0],m['subset'],'frontal' if '_f' in m['ID'] else 'sagittal')].append(m)
    metadata=[group[i] for i in range(max(map(len,strata.values()))) for _,group in sorted(strata.items()) if i<len(group)]
    results=[];failures=[]
    def update():
        status=dict(status='processing',complete_sources=len(results),total_sources=len(metadata),
                    review_clips=sum(len(r['records']) for r in results),native_frames=sum(r['frames'] for r in results),
                    failures=failures,updated_at=datetime.now(timezone.utc).isoformat())
        write(AUDIT/'preparation_status.json',status)
        write(OUT/'automatic_labels/clips.json',[c for r in results for c in r['records']])
        return status
    update()
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(process,m):m for m in metadata}
        for future in cf.as_completed(futures):
            try:results.append(future.result())
            except Exception as e:failures.append(dict(source=futures[future]['file_name'],error=type(e).__name__,message=str(e)[:400]))
            update()
            if (len(results)+len(failures))%10==0:print('Prepared',len(results),'/412; failures',len(failures),flush=True)
    results.sort(key=lambda r:r['clip_id']);status=update()
    complete_crops=[r['queue'] for r in results if not r.get('missing_reference_roi_frames')]
    hold_path=AUDIT/'model_tracking_holds.json'
    model_holds=json.loads(hold_path.read_text()) if hold_path.exists() else {}
    save_csv(PREP/'pose_queue_reference_roi_complete.csv',complete_crops)
    save_csv(PREP/'pose_queue_ready.csv',[r for r in complete_crops if r['clip_id'] not in model_holds])
    write(PREP/'pose_queue_needs_tracking_review.json',[dict(clip_id=r['clip_id'],missing_frames=r['missing_reference_roi_frames']) for r in results if r.get('missing_reference_roi_frames')])
    write(PREP/'pose_rois_all.json',{'provenance':'Reference-derived crops, not independent person detection','clips':{r['clip_id']:r['roi'] for r in results}})
    write(OUT/'automatic_labels/blank_doctor_review.json',dict(package_id=PACKAGE_ID,schema_version=2,reviews={}))
    write(OUT/'provenance/sources.json',[{k:v for k,v in r.items() if k not in ['roi','records']} for r in results])
    (OUT/'START_HERE.txt').write_text('ProGait gait review clips. Use the unified Theia review page.\nEach source is cut into consecutive windows of at most 8 seconds, covering every native frame.\nThe pose overlay uses publisher RTMW-assisted annotations with manual correction, not new Sapiens2 inference.\nNo form conclusion is prefilled. Published gait descriptions remain in the research folder, outside this review package.\nWalking with a transfemoral prosthesis is a separate clinical gait domain, not ARED squat evidence or a healthy cohort.\nSource privacy masking is preserved. CC BY-NC-SA 4.0; credit Yin et al., ProGait, ICCV 2025.\nhttps://huggingface.co/datasets/ericyxy98/ProGait\n')
    status['status']='failed' if failures else 'complete';write(AUDIT/'preparation_status.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()

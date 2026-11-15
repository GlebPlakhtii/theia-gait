"""A stratified 192-frame Sapiens cross-check; not a fully manual accuracy test."""
import csv
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'audit/progait_20261114'
PREP=ROOT/'data/prepared/progait_comparison_20261114'
POSE=ROOT/'data/poses/sapiens2_5b_progait_comparison'
RAW=ROOT/'data/raw/ProGait'
NAMES=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)

def status(phase,**extra):
    data=dict(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**extra);write(AUDIT/'comparison_status.json',data);print(data,flush=True)

def main():
    PREP.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for line in (AUDIT/'metadata.jsonl').read_text().splitlines():
        m=json.loads(line);groups[(m['ID'].split('_')[0],m['subset'],'f' if '_f' in m['ID'] else 's')].append(m)
    chosen=[]
    for _,group in sorted(groups.items()):
        for m in sorted(group,key=lambda m:m['ID']):
            with gzip.open(RAW/'annotations'/m['subset']/(m['ID']+'_keypoints.npy.gz'),'rb') as f:kp=np.load(f,allow_pickle=False)
            with av.open(str(RAW/m['file_name'])) as con:w,h=con.streams.video[0].width,con.streams.video[0].height
            good=(kp[:,:,2]>=.3)&(kp[:,:,0]>=0)&(kp[:,:,0]<w)&(kp[:,:,1]>=0)&(kp[:,:,1]<h)
            if np.all(good.sum(axis=1)>=8):chosen.append(m);break
    assert len(chosen)==15  # Subject 3 has no inside-bars frontal clip in the release.
    expected_frames=len(chosen)*12
    rows=[];rois={'provenance':'Publisher pose-derived subject crops. Conditional pose comparison, not independent person detection.','clips':{}}
    for m in chosen:
        cid='PG_'+m['subset']+'_'+m['ID'];video=RAW/m['file_name'];folder=PREP/cid;folder.mkdir(exist_ok=True)
        with gzip.open(RAW/'annotations'/m['subset']/(m['ID']+'_keypoints.npy.gz'),'rb') as f:kp=np.load(f,allow_pickle=False)
        fmap=[]
        with av.open(str(video)) as con:
            stream=con.streams.video[0];stream.thread_count=1;w,h=stream.width,stream.height;rate=float(stream.average_rate)
            for i,frame in enumerate(con.decode(video=0)):
                t=float(frame.pts*frame.time_base)
                if i==0:t0=t
                fmap.append(dict(source_frame_index=i,source_pts=frame.pts,source_pts_s=t,timestamp_s=t-t0))
        assert kp.shape==(len(fmap),23,3)
        with (folder/'frames.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(fmap[0]));wr.writeheader();wr.writerows(fmap)
        boxes={}
        for i,a in enumerate(kp):
            valid=(a[:,2]>=.3)&(a[:,0]>=0)&(a[:,0]<w)&(a[:,1]>=0)&(a[:,1]<h);xy=a[valid,:2]
            assert len(xy)>=8
            low,high=xy.min(axis=0),xy.max(axis=0);pad=np.maximum((high-low)*.15,20)
            boxes[str(i)]=(np.r_[np.maximum(low-pad,0),np.minimum(high+pad,[w,h])]/[w,h,w,h]).tolist()
        rois['clips'][cid]={'bbox_by_source_frame':boxes}
        with video.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        rows.append(dict(clip_id=cid,local_path=str(video),sha256=sha,start_s=0,end_s=len(fmap)/rate,
                         frame_map=str(folder/'frames.csv'),status='stratified_cross_model_comparison',transformations='native RGB; reference-derived ROI'))
    with (PREP/'queue.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    write(PREP/'rois.json',rois)
    write(AUDIT/'comparison_protocol.json',dict(recorded_before_inference=datetime.now(timezone.utc).isoformat(),
        selection='Lexicographically first source with >=8 valid reference joints on every frame per existing subject × inside/outside × frontal/sagittal stratum; 12 seeded stratified frames per source. High-quality reference coverage selection; not representative of tracking failures. Subject 3 has no inside-bars frontal source.',
        selected_sources=[r['clip_id'] for r in rows],expected_frames=expected_frames,model='Sapiens2-5B BF16 flip-average',
        reference='Publisher RTMW-assisted annotations with manual corrections; not entirely independent manual ground truth',
        crops='Publisher keypoints-derived target ROIs; conditional pose comparison, not a detection benchmark',
        metrics='Euclidean pixel distance and distance normalized by reference body-box diagonal; PCK@0.05 diagonal, reference score >=0.3',
        exclusions='No clinical form accuracy or ARED transfer inferred. No training or threshold tuning on this sample.'))
    # Snapshot the runner: avoid changing the executing ARED program's provenance.
    runner=PREP/'run_sapiens2_pose.py'
    source=(ROOT/'scripts/run_sapiens2_pose.py').read_text().replace("ROOT = Path(__file__).resolve().parents[1]",'ROOT = Path('+repr(str(ROOT))+')')
    source=source.replace('manual_ROI;','publisher_reference_ROI;').replace('manual per-frame foreground ROIs; no identity recognition','publisher pose-derived per-frame subject ROIs; conditional pose test').replace('manual fixed foreground ROI; no identity recognition','publisher reference ROI')
    runner.write_text(source)
    status('waiting_for_ared_gpu',status='queued',expected_frames=expected_frames)
    while True:
        p=json.loads((ROOT/'audit/new_video_intake_20261114/pipeline_status.json').read_text())
        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip().splitlines()[0])
        if p.get('phase')!='native_pose_inference' and used<2500:break
        time.sleep(15)
    status('sapiens_inference',status='running',expected_frames=expected_frames)
    subprocess.run([sys.executable,'-u',str(runner),'--queue',str(PREP/'queue.csv'),'--rois',str(PREP/'rois.json'),
        '--output',str(POSE),'--pilot','12','--checkpoint-every','12'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/audit_pose_batch.py',str(POSE),'--queue',str(PREP/'queue.csv'),'--pilot'],cwd=ROOT,check=True)
    names=json.load(open(POSE/'joint_schema.json'))['body_feet_23'];indices={n:int(i) for i,n in names.items()}
    comparisons=[];all_norm=[]
    for m,row in zip(chosen,rows):
        raw=np.load(POSE/row['clip_id']/'raw_308.npz',allow_pickle=False)
        with gzip.open(RAW/'annotations'/m['subset']/(m['ID']+'_keypoints.npy.gz'),'rb') as f:ref=np.load(f,allow_pickle=False)
        fi=raw['source_frame_index'];gt=ref[fi];pred=raw['keypoints'][:,[indices[n] for n in NAMES]]
        with av.open(row['local_path']) as con:w,h=con.streams.video[0].width,con.streams.video[0].height
        good=(gt[:,:,2]>=.3)&(gt[:,:,0]>=0)&(gt[:,:,0]<w)&(gt[:,:,1]>=0)&(gt[:,:,1]<h)
        delta=np.linalg.norm(pred-gt[:,:,:2],axis=2)
        diag=np.array([np.linalg.norm(a[valid,:2].max(axis=0)-a[valid,:2].min(axis=0)) for a,valid in zip(gt,good)])
        norm=delta/diag[:,None];values=norm[good];all_norm.extend(values.tolist())
        comparisons.append(dict(clip_id=row['clip_id'],sampled_frames=len(fi),reference_score_gated_points=int(good.sum()),
            median_pixel_distance=float(np.median(delta[good])),median_normalized_distance=float(np.median(values)),
            pck_005_reference_body_diagonal=float(np.mean(values<=.05))))
    result=dict(status='complete',sampled_frames=sum(c['sampled_frames'] for c in comparisons),clips=comparisons,
        overall_pck_005_reference_body_diagonal=float(np.mean(np.array(all_norm)<=.05)),
        interpretation='Cross-model agreement with publisher annotations under reference-derived crops; not an expert form score or fully manual accuracy benchmark.')
    write(AUDIT/'sapiens_comparison.json',result);status('complete',status='complete',sampled_frames=result['sampled_frames'])

if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',status='failed',error=type(error).__name__,message=str(error)[:500]);raise

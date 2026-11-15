"""Freeze a metadata-selected same-model ProGait subset and native GPU queue."""
import av
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/prepared/gait_curated_20261114'
OUT=ROOT/'results/gait_metrics_20261114'


def main():
    clips=json.loads((BASE/'manifest.json').read_text())['clips'];byid={c['clip_id']:c for c in clips}
    benchmark={r['clip_id'] for r in csv.DictReader((ROOT/'data/prepared/progait_comparison_20261114/queue.csv').open())}
    subset=[];coverage=[]
    for subject in ['progait_1','progait_2','progait_3','progait_4']:
        for view in ['front/rear','side']:
            candidates=[c for c in clips if c['source']=='progait' and c['environment']=='overground' and
                        c['subject']==subject and c['view']==view and c['clip_duration_s']>=2]
            chosen=[];source_ids=set()
            for c in sorted(candidates,key=lambda c:(c['id'] in benchmark,abs(c['clip_duration_s']-4),c['clip_id'])):
                if c['id'] in source_ids:continue
                chosen.append(c);source_ids.add(c['id'])
                if len(chosen)==2:break
            subset.extend(chosen);coverage.append(dict(subject=subject,view=view,selected=[c['clip_id'] for c in chosen],
                requested_distinct_sources=2,available_distinct_sources=len({c['id'] for c in candidates})))
    frozen=dict(created_at=datetime.now(timezone.utc).isoformat(),selected_before_group_comparison=True,
        rule='Overground walking; two distinct curated source videos per known ProGait participant and nominal front/rear or side view; duration >=2s, prefer non-benchmark sources, closest to 4s then clip ID. Keep all sparse cells explicit.',
        rationale='Same Sapiens model on a predefined subset supplies known-subject comparisons without conflating publisher and Sapiens poses. Additional-input identities remain unresolved.',
        coverage=coverage,clips=[c['clip_id'] for c in subset],frames=sum(c['clip_frames'] for c in subset),
        manifest_sha256=hashlib.sha256((BASE/'manifest.json').read_bytes()).hexdigest())
    (OUT/'harmonization_subset.json').write_text(json.dumps(frozen,indent=2))
    jobs=list(csv.DictReader((OUT/'sapiens_queue.csv').open()))
    rois=json.loads((OUT/'sapiens_rois.json').read_text())
    for c in subset:
        cid=c['clip_id'];folder=OUT/'inputs'/cid;folder.mkdir(exist_ok=True)
        records=json.loads((ROOT/'audit/gait_curation_20261114/screen'/f'{c["id"]}.json').read_text())['records']
        start,end=c['source_first_frame'],c['source_end_frame_exclusive']
        good=[r for r in records if r.get('pass_framing') and start-30<=r['frame']<end+30]
        good=sorted({r['frame']:r for r in good}.values(),key=lambda r:r['frame'])
        assert len(good)>=2,('Insufficient independently detected crop anchors',cid)
        xs=np.array([r['frame'] for r in good]);boxes=np.array([r['box'] for r in good]);ns=np.arange(start,end)
        track=np.stack([np.interp(ns,xs,boxes[:,j]) for j in range(4)],axis=1)
        pad=np.maximum((track[:,2:]-track[:,:2])*.12,12);track[:,:2]-=pad;track[:,2:]+=pad
        track=np.clip(track/[c['width'],c['height'],c['width'],c['height']],0,1)
        rois['clips'][cid]={'bbox_by_source_frame':{str(i):b.tolist() for i,b in enumerate(track)}}
        fm=[]
        with av.open(str(BASE/c['video'])) as con:
            for i,frame in enumerate(con.decode(video=0)):
                ts=float(frame.pts*frame.time_base)
                if i==0:t0=ts
                fm.append(dict(source_frame_index=i,source_pts=frame.pts,source_pts_s=ts,timestamp_s=ts-t0))
        assert len(fm)==c['clip_frames']
        path=folder/'frames.csv'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(fm[0]));w.writeheader();w.writerows(fm)
        jobs.append(dict(clip_id=cid,local_path=str(BASE/c['video']),sha256=c['sha256'],start_s=0,
            end_s=fm[-1]['timestamp_s']+1/c['fps'],frame_map=str(path),status='same_model_harmonization_subset',
            transformations='Native curated video; detector target matched using publisher IoU, interpolated crop'))
    jobs=sorted(jobs,key=lambda q:(byid[q['clip_id']]['clip_frames'],q['clip_id']))
    rois['provenance']='Reviewed Keypoint R-CNN target boxes; interpolated with 12% padding. ProGait target association used publisher-pose IoU. No per-frame manual visibility claim.'
    (OUT/'production_rois.json').write_text(json.dumps(rois,separators=(',',':')))
    with (OUT/'production_queue.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(jobs[0]));w.writeheader();w.writerows(jobs)
    summary=dict(total_jobs=len(jobs),new_video_jobs=206,harmonization_jobs=len(subset),
        total_frames=sum(byid[q['clip_id']]['clip_frames'] for q in jobs),model='Sapiens2-1B',
        queue_sha256=hashlib.sha256((OUT/'production_queue.csv').read_bytes()).hexdigest(),
        rois_sha256=hashlib.sha256((OUT/'production_rois.json').read_bytes()).hexdigest())
    (OUT/'production_inventory.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

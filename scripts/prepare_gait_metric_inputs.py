"""Prepare native-frame Sapiens jobs from the curated gait manifest, reusing reviewed detection crops."""
from pathlib import Path
import csv, json, hashlib
import av
import numpy as np
from run_sapiens2_pose import ROOT

OUT=ROOT/'results/gait_metrics_20261114'
BASE=ROOT/'data/prepared/gait_curated_20261114'

def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)

def main():
    clips=json.loads((BASE/'manifest.json').read_text())['clips']
    cache={}
    for path in (ROOT/'audit/fullbody_locomotion_20261114/framing_screen').glob('*.json'):
        d=json.loads(path.read_text());cache.setdefault(d['source'],[]).extend(d['frames'])
    jobs=[];rois={'provenance':'Reviewed detector target boxes; temporally interpolated with 12% padding. No per-frame manual identity or visibility claim.','clips':{}}
    model=None
    # Start with short examples from every source before the remainder of the batch.
    missing=[c for c in clips if c['source']!='progait']
    first=[]
    for source in ['lbmc','additional_input','ljmu360','movi','mpi3dhp']:
        subset=[c for c in missing if c['source']==source]
        first.append(min(subset,key=lambda c:c['clip_frames']))
    first_ids={c['clip_id'] for c in first}
    missing=first+sorted([c for c in missing if c['clip_id'] not in first_ids],key=lambda c:c['clip_frames'])
    for i,c in enumerate(missing):
        cid=c['clip_id'];folder=OUT/'inputs'/cid;folder.mkdir(parents=True,exist_ok=True)
        if c['source']=='additional_input':
            d=json.loads((ROOT/'audit/gait_curation_20261114/screen'/f'{c["id"]}.json').read_text())
            records=[r for r in d['records'] if r.get('pass_framing')]
        else:records=[r for r in cache.get(c['original'],[]) if r.get('proposal_pass')]
        start,end=c['source_first_frame'],c['source_end_frame_exclusive']
        records=[r for r in records if start-round(c['fps'])<=r['frame']<end+round(c['fps']) and r.get('box')]
        records=sorted({r['frame']:r for r in records}.values(),key=lambda r:r['frame'])
        fm=[];det=[]
        with av.open(str(BASE/c['video'])) as con:
            con.streams.video[0].thread_count=2
            for n,frame in enumerate(con.decode(video=0)):
                t=float(frame.pts*frame.time_base)
                if n==0:t0=t
                fm.append(dict(source_frame_index=n,source_pts=frame.pts,source_pts_s=t,timestamp_s=t-t0))
                if len(records)>=2 or (n%max(1,round(c['fps']/5)) and n!=c['clip_frames']-1):continue
                if model is None:
                    import torch
                    from torchvision.models.detection import keypointrcnn_resnet50_fpn,KeypointRCNN_ResNet50_FPN_Weights
                    torch.set_num_threads(4)
                    model=keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,min_size=640,max_size=960).eval().cuda()
                import torch,cv2
                im=frame.to_ndarray(format='rgb24');h,w=im.shape[:2];scale=min(1,960/max(h,w));small=cv2.resize(im,(round(w*scale),round(h*scale)))
                with torch.inference_mode():pred=model([torch.from_numpy(small).permute(2,0,1).float().cuda()/255])[0]
                use=pred['scores']>=.85
                boxes=pred['boxes'][use].cpu().numpy()/scale
                if len(boxes):
                    areas=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
                    # The intended MPI actor is in the central capture volume, away from the edge bystander.
                    if c['source']=='mpi3dhp':
                        cx=(boxes[:,0]+boxes[:,2])/(2*w);areas=np.where((cx>.15)&(cx<.9),areas,0)
                    if areas.max()>0:det.append(dict(frame=n+start,box=boxes[np.argmax(areas)].tolist()))
        assert len(fm)==c['clip_frames'],cid
        if len(records)<2:records=det
        assert len(records)>=2,('No track proposal',cid)
        xs=np.array([r['frame'] for r in records]);boxes=np.array([r['box'] for r in records]);ns=np.arange(start,end)
        track=np.stack([np.interp(ns,xs,boxes[:,j]) for j in range(4)],axis=1)
        pad=np.maximum((track[:,2:]-track[:,:2])*.12,12)
        track[:,:2]-=pad;track[:,2:]+=pad
        track=np.clip(track/np.array([c['width'],c['height'],c['width'],c['height']]),0,1)
        rois['clips'][cid]={'bbox_by_source_frame':{str(n):box.tolist() for n,box in enumerate(track)}}
        fmap=folder/'frames.csv'
        with fmap.open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(fm[0]));wr.writeheader();wr.writerows(fm)
        dump(folder/'track.json',dict(source=c['original'],source_first_frame=start,records=records,method='linear interpolation of target detector boxes; 12% padding',visibility_verified=False))
        jobs.append(dict(clip_id=cid,local_path=str(BASE/c['video']),sha256=c['sha256'],start_s=0,end_s=fm[-1]['timestamp_s']+1/c['fps'],frame_map=str(fmap),status='native_gait_metric_inference',transformations='Curated native video; source mapping retained in manifest'))
        print(f'Prepared {i+1}/{len(missing)} {cid}',flush=True)
    with (OUT/'sapiens_queue.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(jobs[0]));wr.writeheader();wr.writerows(jobs)
    dump(OUT/'sapiens_rois.json',rois)
    dump(OUT/'input_inventory.json',dict(clips=len(clips),publisher_pose_clips=390,sapiens_jobs=len(jobs),sapiens_frames=sum(c['clip_frames'] for c in missing),specification='Prosthetic Gait Analysis from a Single RGB Video, 2026-11-14',scope='All 596 curated clips; incompatible activities return not_applicable for walking-specific metrics.'))

if __name__=='__main__':main()

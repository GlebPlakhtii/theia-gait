"""Sparse person/landmark screen to propose cuts; visual QC is still required.

This is not clinical pose analysis. Inference visibility flags do not certify
that a joint is visible, and no clinical label is generated from these scores.
"""
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from scipy.io import loadmat
from torchvision.models.detection import keypointrcnn_resnet50_fpn,KeypointRCNN_ResNet50_FPN_Weights
from prepare_fullbody_references import ROOT,RAW,AUDIT,sha


def sources(mode):
    if mode=='ljmu':
        for p in sorted((RAW/'ljmu360').glob('*.mp4')):
            if p.stem.startswith('P1_'):continue
            subject=p.stem.split('_')[0]
            view='front' if 'Front' in p.stem else 'rear' if 'Rear' in p.stem else 'oblique (45° source label)' if '_45' in p.stem else 'side (90° source label)'
            yield dict(source='ljmu360',original=str(p.relative_to(ROOT)),subject=subject,
                trial_group='ljmu360_'+p.stem.rsplit('_',1)[0],activity='walk',view=view,
                role='reported_healthy_cohort',environment='overground',activity_label_basis='Publisher walking task / filename',
                notes='Published healthy-gait cohort; no individual clinical form verdict. Native view and frame rate retained. Camera files have independent trims; source angle is not calibrated anatomical yaw.')
    else:
        for subject in [1,3,4,5]:
            p=RAW/'movi'/f'F_v3d_Subject_{subject}.mat'
            move=loadmat(p,simplify_cells=True)[f'Subject_{subject}_F']['move']
            for label,flag in zip(move['motions_list'],move['flags30']):
                label=str(label).strip()
                if label not in ['walking','jogging','running_in_spot']:continue
                activity={'walking':'walk','jogging':'jog','running_in_spot':'run in place'}[label]
                for camera in [1,2]:
                    original=RAW/'movi'/f'F_PG{camera}_Subject_{subject}_L.avi'
                    yield dict(source='movi',original=str(original.relative_to(ROOT)),subject=f'Subject_{subject}',
                        trial_group=f'movi_F_Subject_{subject}_{activity}',activity=activity,
                        view=f'PG{camera} stationary camera; body turns',role='viewpoint_only',
                        environment='in-place' if activity=='run in place' else 'overground',
                        activity_label_basis='Publisher motions_list / flags30, 1-based inclusive to 0-based half-open; framing trim stays within original action',
                        publisher_start_frame=int(flag[0])-1,publisher_end_frame_exclusive=int(flag[1]),
                        metadata_sha256=sha(p),notes='Publisher-labeled locomotion interval. Indoor route includes turns and transitions. Health screening and correct form are unverified. Running in place remains separate from forward jogging.')


def run(mode):
    torch.set_num_threads(4)
    model=keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
        progress=False,min_size=640,max_size=1280).eval().cuda()
    cache=AUDIT/'framing_screen';cache.mkdir(exist_ok=True)
    plans=[];rejected=[];start=time.monotonic();inferred=0
    for i,item in enumerate(sources(mode)):
        p=ROOT/item['original'];cap=cv2.VideoCapture(str(p))
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        lo=item.get('publisher_start_frame',0);hi=item.get('publisher_end_frame_exclusive',count)
        assert 0<=lo<hi<=count,(p,lo,hi,count)
        step=max(1,round(fps/5));frames=sorted(set(range(lo,hi,step))|{hi-1})
        ident=f'{p.stem}_{lo}_{hi}'
        cachepath=cache/f'{ident}.json'
        if cachepath.exists():data=json.loads(cachepath.read_text())
        else:
            records=[]
            for offset in range(0,len(frames),4):
                batch=frames[offset:offset+4];tensors=[]
                for n in batch:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,img=cap.read();assert ok,(p,n)
                    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    tensors.append(torch.from_numpy(rgb).permute(2,0,1).to('cuda',dtype=torch.float32)/255)
                with torch.inference_mode():predictions=model(tensors)
                inferred+=len(batch)
                for n,pred in zip(batch,predictions):
                    valid=torch.where(pred['scores']>=0.8)[0]
                    if not len(valid):records.append(dict(frame=n,proposal_pass=False,reason='no confident target'));continue
                    boxes=pred['boxes'][valid];areas=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                    chosen=valid[torch.argmax(areas)]
                    box=pred['boxes'][chosen].cpu().tolist();k=pred['keypoints'][chosen].cpu().numpy()
                    score=float(pred['scores'][chosen]);bh=box[3]-box[1]
                    inside=(box[0]>=width*.02 and box[2]<=width*.98 and box[1]>=height*.01 and box[3]<=height-max(12,height*.02))
                    ankle_space=float(np.max(k[15:17,1]))<=height-max(20,bh*.08)
                    passed=inside and ankle_space and bh>=height*.15
                    records.append(dict(frame=n,proposal_pass=bool(passed),person_score=round(score,4),
                        box=[round(v,2) for v in box],keypoints=k.round(2).tolist(),
                        keypoint_scores=pred['keypoints_scores'][chosen].cpu().numpy().round(3).tolist(),
                        reason='interior person box and ankle margin' if passed else 'frame-edge / ankle-margin / small-scale gate'))
            data=dict(model='torchvision Keypoint R-CNN ResNet50 FPN COCO_V1',torchvision=torchvision.__version__,
                resize=dict(min_size=640,max_size=1280),sample_stride_frames=step,source=item['original'],fps=fps,
                width=width,height=height,frames=records,
                limitation='Sparse framing screen only. Predicted landmarks are not measured visibility or clinical motion labels.')
            cachepath.write_text(json.dumps(data,indent=2))
        cap.release()
        runs=[];current=[]
        for rec in data['frames']:
            if rec['proposal_pass']:current.append(rec['frame'])
            else:
                if current:runs.append(current)
                current=[]
        if current:runs.append(current)
        best=max(runs,key=lambda x:x[-1]-x[0],default=[])
        guard=max(1,round(fps*.1))
        a,b=(best[0]+guard,best[-1]-guard+1) if best else (0,0)
        if b-a>=round(2*fps):
            plans.append(dict(**item,start_frame=a,end_frame_exclusive=b,
                framing_screen=str(cachepath.relative_to(ROOT))))
            outcome=f'{(b-a)/fps:.2f}s proposed'
        else:
            rejected.append(dict(**item,reason='No continuous 2-second interior-framing interval in sparse model screen',framing_screen=str(cachepath.relative_to(ROOT))))
            outcome='withheld before cutting'
        (AUDIT/f'expansion_{mode}_plan.json').write_text(json.dumps(dict(clips=plans,rejected=rejected),indent=2))
        print(mode,i+1,p.name,item['activity'],outcome,flush=True)
    print('Screen complete',mode,len(plans),'candidates',len(rejected),'withheld;',inferred,'sampled frames;',round(time.monotonic()-start,1),'seconds',flush=True)


if __name__=='__main__':run(sys.argv[1])

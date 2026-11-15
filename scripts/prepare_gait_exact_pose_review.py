"""Create exact native-frame raw/pose pairs without altering videos or poses."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from gait_geometry import NAMES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gait_metrics_20261114'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--clip',required=True)
    parser.add_argument('--frames',type=int,nargs='+',required=True)
    parser.add_argument('--stem',required=True)
    parser.add_argument('--harmonized',action='store_true')
    args=parser.parse_args()
    assert Path(args.stem).name==args.stem
    base=OUT/'harmonized' if args.harmonized else OUT
    catalog=json.loads((base/'catalog.json').read_text())['clips']
    item=next(c for c in catalog if c['clip_id']==args.clip)
    folder=base/'clips'/args.clip
    metric=json.loads((folder/'metrics.json').read_text())
    roi=json.loads((OUT/'production_rois.json').read_text())['clips'][args.clip]
    frames=sorted(set(args.frames));assert all(0<=i<item['clip_frames'] for i in frames)
    cells=[];evidence=[]
    cap=cv2.VideoCapture(item['video_absolute'])
    with np.load(folder/'observations.npz',allow_pickle=False) as observations:
        for fi in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,raw=cap.read();assert ok
            h,w=raw.shape[:2];overlay=raw.copy()
            box=roi.get('bbox_by_source_frame',{}).get(str(fi),roi.get('bbox_normalized_xyxy'))
            assert box is not None,'No recorded target ROI for the requested frame'
            x0,y0,x1,y1=np.rint(np.array(box)*[w,h,w,h]).astype(int)
            xy=observations['xy_pixels'][fi];confidence=observations['confidence'][fi]
            inside=np.isfinite(xy).all(1)&(confidence>=.3)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            if inside.any():
                low=np.floor(xy[inside].min(0)).astype(int);high=np.ceil(xy[inside].max(0)).astype(int)
                x0=min(x0,low[0]);y0=min(y0,low[1]);x1=max(x1,high[0]);y1=max(y1,high[1])
            x0=max(0,x0-30);y0=max(0,y0-30);x1=min(w,x1+30);y1=min(h,y1+50)
            for side,color in [('left',(80,255,70)),('right',(0,170,255))]:
                for aa,bb in [('shoulder','elbow'),('elbow','wrist'),('shoulder','hip'),('hip','knee'),('knee','ankle'),('ankle','heel'),('heel','big_toe'),('big_toe','small_toe')]:
                    ia,ib=[NAMES.index(side+'_'+name) for name in [aa,bb]]
                    if inside[[ia,ib]].all():
                        cv2.line(overlay,tuple(np.rint(xy[ia]).astype(int)),tuple(np.rint(xy[ib]).astype(int)),color,2)
                for name in ['hip','knee','ankle','heel','big_toe','small_toe']:
                    j=NAMES.index(side+'_'+name)
                    if inside[j]:cv2.circle(overlay,tuple(np.rint(xy[j]).astype(int)),4,color,-1)
            source_frame=int(observations['source_frame'][fi]);pair=[]
            for label,image in [('raw',raw),('pose',overlay)]:
                crop=image[y0:y1,x0:x1];scale=min(400/crop.shape[1],460/crop.shape[0])
                crop=cv2.resize(crop,None,fx=scale,fy=scale);cell=np.zeros((495,410,3),np.uint8)
                cell[35:35+crop.shape[0],:crop.shape[1]]=crop
                cv2.putText(cell,args.clip,(2,12),cv2.FONT_HERSHEY_SIMPLEX,.38,(240,240,240),1)
                cv2.putText(cell,f'{label} | clip {fi} | source {source_frame}',(2,29),cv2.FONT_HERSHEY_SIMPLEX,.45,(190,220,250),1)
                pair.append(cell)
            cells.append(np.hstack(pair))
            evidence.append(dict(clip_frame=fi,source_frame=source_frame,
                time_s=float(observations['timestamp_s'][fi]),display_crop_xyxy=[int(v) for v in [x0,y0,x1,y1]],
                joints={name:dict(xy=xy[j].tolist(),confidence=float(confidence[j]),quality_valid=bool(observations['quality_mask'][fi,j])) for j,name in enumerate(NAMES)}))
    cap.release()
    output=OUT/'batch_pose_review'/args.stem
    assert cv2.imwrite(str(output.with_suffix('.jpg')),np.vstack(cells))
    output.with_suffix('.json').write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),
        clip_id=args.clip,branch='harmonized' if args.harmonized else 'main',pose=metric['pose'],frames=evidence,
        display_only='Recorded ROI expanded to include confident in-frame landmarks, then a margin. Sources, poses and inference ROIs are unchanged.'),indent=2)+'\n')
    print(json.dumps(dict(image=str(output.with_suffix('.jpg').relative_to(ROOT)),clip=args.clip,frames=frames)))


if __name__=='__main__':main()

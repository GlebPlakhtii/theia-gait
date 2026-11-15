"""Prepare reproducible sampled pose-QC boards and a non-diagnostic review queue."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import cv2,numpy as np
from gait_geometry import bend,NAMES
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gait_metrics_20261114';DEST=OUT/'batch_pose_review';DEST.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--max-clips',type=int,default=16);args=p.parse_args()
manifest=json.loads((OUT/'comparison_manifest_frozen.json').read_text());rows={r['clip_id']:r for r in manifest['rows']}
rois=json.loads((OUT/'production_rois.json').read_text())['clips']
previous=json.loads((OUT/'pose_visual_review.json').read_text())['clips'];candidates=[];screen=[]
for complete in sorted((OUT/'poses_sapiens2_1b').glob('*/complete.json')):
 cid=complete.parent.name;meta=rows[cid];branch='harmonized' if meta['same_model_subset'] else ''
 metrics_path=OUT/branch/'clips'/cid/'metrics.json'
 if not metrics_path.exists():continue
 m=json.loads(metrics_path.read_text())
 if m['status']!='observations_extracted':continue
 a=np.load(OUT/branch/'clips'/cid/'observations.npz');flags=[]
 if meta['view']=='side':
  scales=[s['fixed_L0_pixels'] for s in m['segments'] if s.get('fixed_L0_pixels')]
  if scales:
   scale=float(np.median(scales));points=a['xy_pixels'];valid=a['quality_mask'][:,[13,14,15,16]].all(1)
   collapsed=valid&(np.linalg.norm(points[:,13]-points[:,14],axis=1)<.03*scale)&(np.linalg.norm(points[:,15]-points[:,16],axis=1)<.03*scale)
   if collapsed.mean()>.2:flags.append(dict(reason='possible_left_right_keypoint_collapse_requires_visual_review',frames=np.flatnonzero(collapsed).tolist(),fraction=float(collapsed.mean()),engineering_screen_only=True))
  for side,h,k,ankle in [('left',11,13,15),('right',12,14,16)]:
   points=a['xy_pixels'];mask=a['quality_mask'][:,[h,k,ankle]].all(1)
   angles=bend(points[:,h],points[:,k],points[:,ankle]);ids=np.flatnonzero(mask&(angles>130))
   if len(ids):flags.append(dict(reason='large_projected_bend_requires_visual_check_not_clinical_abnormality',side=side,frames=ids.tolist()))
 if m['valid_observation_fraction']<.8:flags.append(dict(reason='low_overall_landmark_coverage'))
 if m['jump_masked_joint_observations']>m['processed_frames']*.5:flags.append(dict(reason='many_discontinuity_masks'))
 reviewed=cid in previous
 screen.append(dict(clip_id=cid,reviewed=reviewed,comparison_candidate=bool(meta['comparison_modes']),flags=flags))
 if not reviewed:candidates.append((not bool(flags),not bool(meta['comparison_modes']),cid))
candidates=sorted(candidates)[:args.max_clips]
boards=[]
for offset in range(0,len(candidates),4):
 cells=[];evidence=[]
 for _,_,cid in candidates[offset:offset+4]:
  paths=sorted((OUT/'poses_sapiens2_1b'/cid).glob('preview_*.jpg'));take=[paths[i] for i in sorted(set([0,len(paths)//2,len(paths)-1]))]
  if len(take)!=3:raise ValueError('Missing preview coverage: '+cid)
  branch='harmonized' if rows[cid]['same_model_subset'] else ''
  with np.load(OUT/branch/'clips'/cid/'observations.npz') as observations:
   review_xy=observations['xy_pixels'];review_conf=observations['confidence']
  row=[];frames=[]
  for path in take:
   im=cv2.imread(str(path));fi=int(path.stem.split('_')[1]);frames.append(fi);h,w=im.shape[:2]
   box=rois[cid].get('bbox_by_source_frame',{}).get(str(fi),rois[cid].get('bbox_normalized_xyxy'))
   x0,y0,x1,y1=np.round(np.array(box)*[w,h,w,h]).astype(int)
   # Include every confident body/foot point in the review display. A static
   # ROI edge must not make an in-frame foot look cropped in a contact sheet.
   # This affects only the review image, never the frozen inference crop.
   points=review_xy[fi];ok=np.isfinite(points).all(1)&(review_conf[fi]>=.3)
   ok&=(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
   if ok.any():
    low=np.floor(points[ok].min(0)).astype(int);high=np.ceil(points[ok].max(0)).astype(int)
    x0=min(x0,low[0]);y0=min(y0,low[1]);x1=max(x1,high[0]);y1=max(y1,high[1])
   # Display context beyond the reviewed ROI so border landmarks are not cut
   # off by the review board. This never changes the inference ROI or video.
   pad=max(20,round(.05*max(x1-x0,y1-y0)))
   x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(w,x1+pad);y1=min(h,y1+pad)
   crop=im[y0:y1,x0:x1];scale=min(360/crop.shape[1],350/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
   cell=np.zeros((390,370,3),np.uint8);cell[40:40+crop.shape[0],:crop.shape[1]]=crop
   cv2.putText(cell,cid,(3,14),cv2.FONT_HERSHEY_SIMPLEX,.37,(240,240,240),1)
   cv2.putText(cell,f'clip frame {fi} / target crop + margin',(3,31),cv2.FONT_HERSHEY_SIMPLEX,.4,(170,220,240),1)
   row.append(cell)
  cells.append(np.hstack(row));evidence.append(dict(clip_id=cid,frames=frames))
 stem='batch_'+datetime.now(timezone.utc).strftime('%H%M%S')+'_'+str(offset//4+1)
 cv2.imwrite(str(DEST/(stem+'.jpg')),np.vstack(cells));(DEST/(stem+'.json')).write_text(json.dumps(evidence,indent=2));boards.append(str((DEST/(stem+'.jpg')).relative_to(ROOT)))
(OUT/'pose_review_queue.json').write_text(json.dumps(dict(updated_at=datetime.now(timezone.utc).isoformat(),
 scope='Numerical flags prioritize visual review only. They do not label clinical abnormality and do not automatically exclude unusual gait.',clips=screen,boards=boards),indent=2))
print(json.dumps(dict(completed_extracted=len(screen),already_reviewed=sum(r['reviewed'] for r in screen),boards=boards),indent=2))

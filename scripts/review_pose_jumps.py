"""Render largest image-plane knee-angle steps for review, without correcting poses."""
import argparse,csv,json
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw
from plot_pose_review import angle


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--queue',default='data/manifests/pose_queue_ready.csv');a=p.parse_args()
 names={n:int(i) for i,n in json.loads((a.directory/'joint_schema.json').read_text())['all_308'].items()}
 reports=[]
 for q in csv.DictReader(open(a.queue)):
  d=a.directory/q['clip_id']
  if not (d/'complete.json').exists():continue
  raw=np.load(d/'raw_308.npz');pts=raw['keypoints'];sc=raw['scores'];ts=raw['timestamp_s'];frames=raw['source_frame_index']
  angles=np.stack([angle(*(pts[:,names[f'{s}_{j}']] for j in ('hip','knee','ankle'))) for s in ('left','right')],axis=1)
  jumps=np.nan_to_num(np.max(np.abs(np.diff(angles,axis=0)),axis=1),nan=0)
  chosen=[]
  for k in np.argsort(jumps)[::-1]:
   i=int(k+1)
   if all(abs(ts[i]-ts[j])>.5 for j in chosen):chosen.append(i)
   if len(chosen)==3:break
  wanted={int(frames[i]):i for j in chosen for i in (j-1,j)};ims={}
  with av.open(q['local_path']) as container:
   for fi,fr in enumerate(container.decode(video=0)):
    if fi>max(wanted):break
    if fi not in wanted:continue
    i=wanted[fi];im=fr.to_ndarray(format='bgr24')
    for side,color in [('left',(0,255,0)),('right',(0,128,255))]:
     inds=[names[f'{side}_{j}'] for j in ('shoulder','hip','knee','ankle','big_toe')]
     for i1,i2 in zip(inds[:-1],inds[1:]):
      cv2.line(im,tuple(np.rint(pts[i,i1]).astype(int)),tuple(np.rint(pts[i,i2]).astype(int)),color,2)
     for ji in inds:cv2.circle(im,tuple(np.rint(pts[i,ji]).astype(int)),3,color,-1)
    pil=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));pil.thumbnail((360,600));ims[fi]=pil
  cellh=max(im.height for im in ims.values())+45
  contact=Image.new('RGB',(380*3,cellh*2),'white');draw=ImageDraw.Draw(contact)
  selected=[]
  for col,j in enumerate(chosen):
   selected.append({'before_frame':int(frames[j-1]),'after_frame':int(frames[j]),'after_timestamp_s':float(ts[j]),'largest_knee_angle_step_degrees':float(jumps[j-1]),'left_right_before':angles[j-1].tolist(),'left_right_after':angles[j].tolist()})
   for row,i in enumerate((j-1,j)):
    fi=int(frames[i]);im=ims[fi];x=col*380;y=row*cellh
    contact.paste(im,(x+(380-im.width)//2,y+40));draw.text((x+8,y+5),f'frame {fi}; t={ts[i]:.3f}s',fill='black');draw.text((x+8,y+20),f'{"before" if row==0 else "after"}; pair step={jumps[j-1]:.1f} deg',fill='black')
  contact.save(d/'temporal_jump_review.jpg',quality=95)
  report={'clip_id':q['clip_id'],'interpretation':'Largest projected-angle changes for visual review; not automatic error, clinical asymmetry or impairment labels; no pose modification','pairs':selected}
  (d/'temporal_jump_review.json').write_text(json.dumps(report,indent=2));reports.append(report)
 (a.directory/'temporal_jump_review.json').write_text(json.dumps(reports,indent=2));print('Reviewed candidate pairs for',len(reports),'completed clips')

if __name__=='__main__':main()

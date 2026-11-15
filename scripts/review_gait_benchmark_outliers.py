"""Prepare evidence boards for large benchmark disagreements; no model rerun."""
import csv,gzip,json
from pathlib import Path
import av,cv2,numpy as np
from gait_geometry import bend,NAMES
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gait_metrics_20261114/model_benchmark'
BASE=ROOT/'data/poses/sapiens2_5b_progait_comparison'
rows=list(csv.DictReader((ROOT/'data/prepared/progait_comparison_20261114/queue.csv').open()))
schema=json.loads((BASE/'joint_schema.json').read_text())['body_feet_23'];indices=[next(int(k) for k,v in schema.items() if v==name) for name in NAMES]
cases=[];cache={};queue={r['clip_id']:r for r in rows}
for row in rows:
 cid=row['clip_id'];subset=cid.split('_')[1];ident=cid[len('PG_'+subset+'_'):]
 if '_s' not in ident:continue
 arrays={name:np.load(path) for name,path in [('1b',OUT/(cid+'_1b.npz')),('5b',BASE/cid/'raw_308.npz')]}
 fi=arrays['1b']['source_frame_index']
 with gzip.open(ROOT/'data/raw/ProGait/annotations'/subset/(ident+'_keypoints.npy.gz'),'rb') as f:ref=np.load(f)[fi]
 cache[cid]=(fi,ref,{k:a['keypoints'][:,indices] for k,a in arrays.items()})
 for si,frame in enumerate(fi):
  for side,h,k,a in [('left',11,13,15),('right',12,14,16)]:
   if min(ref[si,[h,k,a],2])<.3:continue
   expected=float(bend(ref[si:si+1,h,:2],ref[si:si+1,k,:2],ref[si:si+1,a,:2])[0])
   actual={m:float(bend(p['keypoints'][si:si+1,indices[h]],p['keypoints'][si:si+1,indices[k]],p['keypoints'][si:si+1,indices[a]])[0]) for m,p in arrays.items()}
   cases.append(dict(clip_id=cid,source_frame=int(frame),sample_index=si,side=side,reference_deg=expected,model_deg=actual,max_disagreement_deg=max(abs(v-expected) for v in actual.values())))
cases.sort(key=lambda x:x['max_disagreement_deg'],reverse=True);chosen=[];used=set();per_source={}
for c in cases:
 key=(c['clip_id'],c['source_frame'])
 if key in used or per_source.get(c['clip_id'],0)>=2:continue
 chosen.append(c);used.add(key);per_source[c['clip_id']]=per_source.get(c['clip_id'],0)+1
 if len(chosen)==6:break
boards=[]
for c in chosen:
 cid=c['clip_id'];im=None
 with av.open(queue[cid]['local_path']) as con:
  for i,frame in enumerate(con.decode(video=0)):
   if i==c['source_frame']:im=frame.to_ndarray(format='bgr24');break
 fi,ref,preds=cache[cid];s=c['sample_index'];valid=ref[s,:,2]>=.3;lo=ref[s,valid,:2].min(0);hi=ref[s,valid,:2].max(0);pad=(hi-lo)*.22
 x0,y0=np.maximum(0,lo-pad).astype(int);x1,y1=np.minimum([im.shape[1],im.shape[0]],hi+pad).astype(int)
 panels=[]
 for name,points,color in [('publisher',ref[s,:,:2],(70,220,90)),('1B',preds['1b'][s],(0,170,255)),('5B',preds['5b'][s],(255,170,90))]:
  z=im.copy()
  for side in ('left','right'):
   for a,b in [('shoulder','hip'),('hip','knee'),('knee','ankle'),('ankle','heel'),('heel','big_toe')]:
    p,q=[tuple(np.round(points[NAMES.index(side+'_'+n)]).astype(int)) for n in (a,b)]
    cv2.line(z,p,q,color,4);cv2.circle(z,q,6,color,-1)
  z=z[y0:y1,x0:x1];scale=min(440/z.shape[1],300/z.shape[0]);z=cv2.resize(z,None,fx=scale,fy=scale)
  cell=np.zeros((345,450,3),np.uint8);cell[42:42+z.shape[0],:z.shape[1]]=z
  val=c['reference_deg'] if name=='publisher' else c['model_deg'][name.lower()]
  cv2.putText(cell,f'{cid} / f{c["source_frame"]}',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.39,(230,230,230),1)
  cv2.putText(cell,f'{name}: {c["side"]} knee {val:.1f} deg',(4,32),cv2.FONT_HERSHEY_SIMPLEX,.47,color,1);panels.append(cell)
 boards.append(np.hstack(panels))
cv2.imwrite(str(OUT/'outlier_review.jpg'),np.vstack(boards));(OUT/'outlier_cases.json').write_text(json.dumps(chosen,indent=2))
print(json.dumps(chosen,indent=2))

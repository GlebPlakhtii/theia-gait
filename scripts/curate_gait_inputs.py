"""Auditable local quality selection for user footage and ProGait originals."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import subprocess
import sys
import gzip
import re
import time
from fractions import Fraction
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/gait_curation_20261114'
OUT = ROOT / 'data/prepared/gait_curated_20261114'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

# Seconds are coarse human-reviewed scene/activity limits. Automated framing
# and final visual review refine these; they are not clinical gait labels.
ADDITIONAL_SCENES = {
 '013_gait_side':[(0,28,'walking','overground')],
 '014_gait_front':[(0,30,'walking','overground')],
 '017_gait_oblique':[(.3,29,'walking','treadmill')],
 '018_gait_mixed':[(12,25,'running','treadmill')],
 '019_gait_mixed':[(5,29.5,'running','treadmill')],
 '020_gait_side':[(.2,11.8,'running','treadmill')],
 '021_gait_side':[(1.5,10,'jogging','overground')],
 '022_gait_front':[(.5,4.5,'jogging','overground'),(5.2,9.7,'jogging','overground')],
 '023_gait_side':[(3.8,11.8,'jogging','overground')],
 '024_gait_front':[(.3,4.2,'jogging','overground'),(4.8,9.5,'jogging','overground')],
 '025_gait_oblique':[(.5,3.8,'walking','overground'),(4.5,8.4,'running','overground'),(10.8,13.9,'running','overground')],
 '030_gait_front':[(.7,12.4,'walking','overground')],
 '031_gait_front':[(1.5,10.3,'walking','overground')],
 '033_gait_side':[(1.7,11.9,'walking','overground')],
 '034_gait_mixed':[(.4,6.8,'jogging','overground'),(7.8,15.8,'jogging','overground')],
 '053_misc_side':[],
 '054_misc_oblique':[(.2,3.4,'walking','ramp')],
 '12f6e2fd':[(.4,8.4,'jogging','overground'),(9.2,21.7,'jogging','overground')],
 '38e302e3':[(.2,4.6,'walking','overground')],
 '40b65e7c':[(3.5,19.3,'walking','parallel_bars'),(35.7,53.5,'walking','ramp'),(68.5,85,'walking','parallel_bars')],
 '5a94b3ba':[(13.5,17.6,'walking','parallel_bars'),(20.2,25,'walking','parallel_bars'),(31.6,35.1,'walking_with_walker','overground'),(39.2,45.3,'walking','overground')],
 '687f8866':[(.2,5.5,'jogging','treadmill')],
 '93726085':[],
 '9579b44e':[(5.5,14.7,'running','treadmill')],
 'a1680b11':[],
 'c447ff19':[(.3,5.6,'running','overground')],
 'f004cd67':[(.3,9,'running','overground'),(13.5,17.7,'running','overground')],
 'f2cdbd8e':[(.3,6.4,'running','overground')],
 'f31cb89a':[(.1,5.9,'jogging','overground')],
 'v2_-_uncut':[(11.6,20.6,'walking','overground'),(21.6,27,'walking','overground'),(33.5,44.7,'walking','treadmill'),(46,54.5,'stairs_up','stairs'),(56,65,'stairs_down','stairs')],
 'v5':[(.2,7.3,'walking','overground')],
 'v6_uncut':[(17.7,23,'walking','overground'),(27.4,32.5,'walking','overground'),(49.3,54.8,'walking','overground'),(56.5,61.3,'walking','ramp'),(67.3,71.2,'walking_with_weights','overground'),(72.1,75,'walking','treadmill'),(78.2,82.5,'curb_walking','outdoor_curb'),(84.1,87.4,'walking','overground'),(88.9,91,'walking','overground')],
 'v8':[(.2,21.1,'walking','overground')],
}
ADDITIONAL_REJECTIONS={
 '053_misc_side':'Head/upper body cropped; edited transitions and grayscale section.',
 '93726085':'Interview/event montage; no reliable sustained unobstructed head-to-foot locomotion of the intended participant.',
 'a1680b11':'Close-up legs followed by obstructed demonstration with ambiguous target/locomotion; no reliable full-body gait interval.'}

def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(d,indent=2)); tmp.replace(p)

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def probe(p):
    d=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,avg_frame_rate,nb_frames,duration:format=duration','-of','json',str(p)]))
    s=d['streams'][0];fps=float(Fraction(s['avg_frame_rate']));duration=float(s.get('duration',d['format'].get('duration',0)))
    return dict(width=s['width'],height=s['height'],fps=fps,frames=int(s.get('nb_frames') or round(duration*fps)),duration_s=duration)

def inventory():
    paths=sorted((ROOT/'additional_input').glob('*.mp4'))
    def one(p):
        return dict(id='additional_'+p.stem.replace(' ','_'),source='additional_input',original=str(p.relative_to(ROOT)),
            source_sha256=sha(p),cohort='with_amputation',cohort_basis='User supplied designation of additional_input',**probe(p))
    with cf.ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(one,paths))
    write(AUDIT/'additional_inventory.json',rows)
    boards=[];font=ImageFont.truetype(FONT,15)
    for offset in range(0,len(rows),3):
        group=rows[offset:offset+3];sheet=Image.new('RGB',(1920,280*len(group)),'#15222e');draw=ImageDraw.Draw(sheet)
        for row,item in enumerate(group):
            ns=np.linspace(0,item['frames']-1,8).round().astype(int).tolist()
            cap=cv2.VideoCapture(str(ROOT/item['original']))
            draw.text((6,row*280+3),f"{item['id']} | {item['duration_s']:.2f}s | {item['width']}x{item['height']} | {item['fps']:.3f} fps",font=font,fill='white')
            for col,n in enumerate(ns):
                cap.set(1,n);ok,bgr=cap.read()
                if not ok:continue
                im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((238,224))
                sheet.paste(im,(col*240+(240-im.width)//2,row*280+26))
                draw.text((col*240+5,row*280+252),f'{n} / {n/item["fps"]:.2f}s',font=font,fill='white')
            cap.release()
        p=AUDIT/'additional_overview'/f'{offset//3+1:02d}.jpg';p.parent.mkdir(parents=True,exist_ok=True);sheet.save(p,quality=90)
        boards.append(dict(path=str(p.relative_to(ROOT)),ids=[r['id'] for r in group]))
    write(AUDIT/'additional_overview/index.json',boards)
    print(json.dumps(dict(videos=len(rows),seconds=sum(r['duration_s'] for r in rows),boards=len(boards))))

def source_inventory():
    rows=json.loads((AUDIT/'additional_inventory.json').read_text())
    for line in (ROOT/'data/raw/ProGait/metadata.jsonl').read_text().splitlines():
        m=json.loads(line);cid='PG_'+m['subset']+'_'+m['ID']
        d=json.loads((ROOT/'data/prepared/progait_20261114'/cid/'complete.json').read_text())
        category=re.search(r'CCC\s*:\s*(\d+)',m['txt'])
        rows.append(dict(id=cid,source='progait',original='data/raw/ProGait/'+m['file_name'],
            source_sha256=d['source_sha256'],cohort='with_amputation',
            cohort_basis='Official ProGait release: four transfemoral prosthesis users',
            subject='progait_'+m['ID'].split('_')[0],trial_group='progait_'+'_'.join(m['ID'].split('_')[:3]),
            view='front/rear' if '_f' in m['ID'] else 'side',environment='parallel_bars' if m['subset']=='inside' and not m['ID'].startswith('4_0_1') else 'overground',
            subset=m['subset'],width=d['width'],height=d['height'],fps=float(Fraction(d['rate'])),frames=d['frames'],
            duration_s=d['frames']/float(Fraction(d['rate'])),activity='walking',
            annotation='data/raw/ProGait/annotations/'+m['subset']+'/'+m['ID']+'_keypoints.npy.gz',
            clinical_reference_status='available' if category else 'missing' if not m['txt'].strip() else 'publisher_not_for_analysis',
            clinical_reference_category=int(category[1]) if category else None))
    return rows

def load_reference(item):
    if not item.get('annotation'):return None
    with gzip.open(ROOT/item['annotation'],'rb') as f:a=np.load(f,allow_pickle=False)
    if a.shape!=(item['frames'],23,3) or not np.isfinite(a).all():return None
    return a

def screen():
    import torch
    from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
    torch.set_num_threads(4)
    model=keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=640,max_size=960).eval().cuda()
    rows=source_inventory();write(AUDIT/'source_inventory.json',rows)
    selected=set(sys.argv[2:]);rows=[r for r in rows if not selected or r['id'] in selected]
    started=time.monotonic();done=0;inferred=0
    for item in rows:
        cache=AUDIT/'screen'/f'{item["id"]}.json'
        if cache.exists():done+=1;continue
        step=max(1,round(item['fps']/(3 if item['source']=='additional_input' else 1)))
        ns=sorted(set(range(0,item['frames'],step))|{item['frames']-1})
        ref=load_reference(item);cap=cv2.VideoCapture(str(ROOT/item['original']));records=[]
        for offset in range(0,len(ns),6):
            batch=ns[offset:offset+6];tensors=[];actual=[]
            for n in batch:
                cap.set(1,n);ok,bgr=cap.read()
                if not ok:
                    records.append(dict(frame=n,pass_framing=False,reason='decode_failure'));continue
                # Resize only the inference input, never the exported video.
                h,w=bgr.shape[:2];scale=min(1.,960/max(h,w))
                im=cv2.resize(bgr,(round(w*scale),round(h*scale))) if scale<1 else bgr
                tensors.append(torch.from_numpy(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).permute(2,0,1).cuda().float()/255)
                actual.append((n,scale))
            with torch.inference_mode():preds=model(tensors) if tensors else []
            inferred+=len(tensors)
            for (n,scale),pred in zip(actual,preds):
                ids=torch.where(pred['scores']>=.8)[0]
                if not len(ids):records.append(dict(frame=n,pass_framing=False,reason='no_confident_person'));continue
                boxes=pred['boxes'][ids].cpu().numpy()/scale
                target_iou=None
                if ref is not None:
                    kp=ref[n];valid=kp[:,2]>=.3;xy=kp[valid,:2]
                    if len(xy)<8:records.append(dict(frame=n,pass_framing=False,reason='publisher_target_unreliable'));continue
                    lo=xy.min(0);hi=xy.max(0);pad=(hi-lo)*.08
                    rb=np.r_[lo-pad,hi+pad]
                    inter=np.prod(np.maximum(0,np.minimum(boxes[:,2:],rb[2:])-np.maximum(boxes[:,:2],rb[:2])),axis=1)
                    ar=np.prod(boxes[:,2:]-boxes[:,:2],axis=1);ra=np.prod(rb[2:]-rb[:2])
                    iou=inter/(ar+ra-inter+1e-9);chosen=int(np.argmax(iou));target_iou=float(iou[chosen])
                else:
                    candidate_weights=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
                    cx=(boxes[:,0]+boxes[:,2])/(2*item['width'])
                    if item['id']=='additional_v2_-_uncut':
                        candidate_weights=np.where((cx>.3)&(cx<.7),candidate_weights,0)
                    if item['id'] in ['additional_f004cd67','additional_f2cdbd8e']:
                        candidate_weights=np.where((cx>.35)&(cx<.88)&(boxes[:,3]>item['height']*.62),candidate_weights,0)
                    if candidate_weights.max()<=0:
                        records.append(dict(frame=n,pass_framing=False,reason='no_person_in_manually_reviewed_target_region'));continue
                    chosen=int(np.argmax(candidate_weights))
                ix=ids[chosen];box=boxes[chosen];kp=pred['keypoints'][ix].cpu().numpy();kp[:,:2]/=scale
                w,h=item['width'],item['height'];bh=box[3]-box[1]
                inside=box[0]>=max(4,w*.007) and box[2]<=w-max(4,w*.007) and box[1]>=max(4,h*.005) and box[3]<=h-max(8,h*.012)
                ankle_margin=max(kp[15,1],kp[16,1])<=h-max(12,bh*.035)
                size=bh>=max(140,h*.14)
                matched=target_iou is None or target_iou>=.35
                passed=bool(inside and ankle_margin and size and matched)
                records.append(dict(frame=n,pass_framing=passed,box=box.round(2).tolist(),
                    keypoints=kp.round(2).tolist(),keypoint_scores=pred['keypoints_scores'][ix].cpu().numpy().round(3).tolist(),
                    target_iou=target_iou,person_score=float(pred['scores'][ix]),
                    reason='interior_target' if passed else 'edge_or_foot_margin_or_small_target_or_unmatched',
                    detected_people=len(ids)))
        cap.release();records.sort(key=lambda r:r['frame'])
        write(cache,dict(source=item['original'],source_sha256=item['source_sha256'],sample_stride=step,
            target_selection='publisher-keypoint target IoU' if ref is not None else 'largest detection; manual subject review required',
            model='torchvision Keypoint R-CNN ResNet50 FPN COCO_V1',
            limitation='Sparse quality screen, not medical or every-frame visibility certification',records=records))
        done+=1
        status=dict(phase='framing_screen',complete=done,total=len(rows),inferred_frames=inferred,elapsed_s=round(time.monotonic()-started,1),last=item['id'])
        write(AUDIT/('rescreen_status.json' if selected else 'status.json'),status)
        print(json.dumps(status),flush=True)
    write(AUDIT/('rescreen_complete.json' if selected else 'screen_complete.json'),dict(sources=len(rows),inferred_frames=inferred,elapsed_s=time.monotonic()-started))

def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.where(edges==1)[0].tolist(),np.where(edges==-1)[0].tolist()))

def reference_quality(item,a):
    """Every-frame annotation geometry screen; confidence is not visibility."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, median_filter
    xy=a[:,:,:2];scores=a[:,:,2];w,h=item['width'],item['height']
    lo=xy.min(1);hi=xy.max(1);bh=hi[:,1]-lo[:,1]
    # Includes predicted toe/heel positions and room above the face landmarks.
    geometry=(lo[:,0]>w*.009)&(hi[:,0]<w*.991)&(lo[:,1]>np.maximum(8,bh*.065))&(hi[:,1]<h-np.maximum(10,bh*.025))&(bh>max(140,h*.14))
    important=[5,6,11,12,13,14,15,16]
    confidence=(scores[:,important].min(1)>=.25)&(scores[:,:5].max(1)>=.25)&(scores[:,17:20].max(1)>=.25)&(scores[:,20:23].max(1)>=.25)
    # Fill only <=3-frame confidence dips; never bridge out-of-frame geometry.
    for s,e in runs(~confidence):
        if s>0 and e<len(confidence) and e-s<=3:confidence[s:e]=True
    hips=xy[:,11:13].mean(1);feet=xy[:,15:17]
    relative=((feet-hips[:,None,:])/np.maximum(bh[:,None,None],1)).reshape(len(a),4)
    relative=median_filter(relative,size=(5,1))
    span=max(3,round(item['fps']*.8))
    movement=np.max(maximum_filter1d(relative,size=span,axis=0)-minimum_filter1d(relative,size=span,axis=0),axis=1)
    moving=movement>=.025
    return geometry & confidence & moving,dict(geometry_pass_fraction=float(geometry.mean()),
        confidence_pass_fraction=float(confidence.mean()),locomotion_proxy_fraction=float(moving.mean()),
        limitation='Pose changes are only a motion proxy. Publisher walking labels and visual review establish activity; no gait normality threshold is used.')

def plan():
    rows=source_inventory();plans=[];rejected=[];pending=[]
    write(AUDIT/'additional_scene_review.json',dict(scenes=ADDITIONAL_SCENES,rejections=ADDITIONAL_REJECTIONS,
        evidence='All 33 eight-frame source overview boards and nine denser 2-second scene sheets reviewed; final interval review follows.'))
    for item in rows:
        path=AUDIT/'screen'/f'{item["id"]}.json'
        if not path.exists():pending.append(item['id']);continue
        data=json.loads(path.read_text());ns=np.array([r['frame'] for r in data['records']]);good=np.array([r['pass_framing'] for r in data['records']])
        # A span is supported only between two passing screen anchors.
        mask=np.zeros(item['frames'],dtype=bool)
        for j in range(len(ns)-1):
            if good[j] and good[j+1]:mask[ns[j]:ns[j+1]+1]=True
        details={};fps=item['fps'];reason=None
        if item['source']=='progait':
            if item['clinical_reference_status']=='publisher_not_for_analysis':reason='Publisher explicitly marks source not intended for analysis.'
            a=load_reference(item)
            if a is None:reason='Publisher annotations do not align to native frames; target identity/framing needs separate review.'
            if reason:
                rejected.append(dict(id=item['id'],original=item['original'],reason=reason));continue
            refmask,details=reference_quality(item,a);mask &= refmask
            scenes=[(0,item['duration_s'],'walking',item['environment'])]
        else:
            name=item['id'].removeprefix('additional_');scenes=ADDITIONAL_SCENES[name]
            if not scenes:reason=ADDITIONAL_REJECTIONS.get(name,'No accepted locomotion scene')
        before=len(plans)
        for number,(start,end,activity,environment) in enumerate(scenes):
            scene_mask=mask.copy();a=max(0,round(start*fps));b=min(len(mask),round(end*fps))
            scene_mask[:a]=False;scene_mask[b:]=False
            for s,e in runs(scene_mask):
                # Native-frame guards leave room around detected failures/cuts.
                guard=max(2,round(fps*.1));s+=guard;e-=guard
                if e-s<round(2*fps):continue
                # Keep natural continuous bouts intact up to 15 seconds.
                count=max(1,int(np.ceil((e-s)/(15*fps))))
                boundaries=np.linspace(s,e,count+1).round().astype(int)
                for aa,bb in zip(boundaries[:-1],boundaries[1:]):
                    ident=item['id']+f'_{aa}_{bb}'
                    plans.append(dict(item,clip_id=ident,activity=activity,environment=environment,
                        source_first_frame=int(aa),source_end_frame_exclusive=int(bb),
                        clip_frames=int(bb-aa),clip_duration_s=(bb-aa)/fps,source_start_s=aa/fps,source_end_s=bb/fps,
                        scene_index=number,quality_details=details,review_status='pending_visual_review',
                        selection_basis='Publisher-target geometry at native cadence + 1 Hz independent model framing + visual review' if item['source']=='progait' else '3 Hz model framing + manual scene/activity boundaries + visual review',
                        all_frames_visibility_certified=False))
        if len(plans)==before:rejected.append(dict(id=item['id'],original=item['original'],reason=reason or 'No continuous >=2-second locomotion interval passed conservative framing/target/motion gates.'))
    adjustment_path=AUDIT/'manual_candidate_adjustments.json'
    adjustments=json.loads(adjustment_path.read_text()) if adjustment_path.exists() else {}
    adjusted=[];rejected_candidates=[]
    for c in plans:
        edit=adjustments.get(c['clip_id'],{})
        if edit.get('action')=='reject':
            rejected_candidates.append(dict(clip_id=c['clip_id'],source=c['source'],original=c['original'],reason=edit['reason']));continue
        if edit.get('action')=='trim':
            c=dict(c,parent_candidate_id=c['clip_id'],source_first_frame=edit['source_first_frame'],source_end_frame_exclusive=edit['source_end_frame_exclusive'])
            c['clip_id']=c['id']+f'_{c["source_first_frame"]}_{c["source_end_frame_exclusive"]}'
            c['clip_frames']=c['source_end_frame_exclusive']-c['source_first_frame']
            c['clip_duration_s']=c['clip_frames']/c['fps'];c['source_start_s']=c['source_first_frame']/c['fps'];c['source_end_s']=c['source_end_frame_exclusive']/c['fps']
            c['manual_trim_reason']=edit['reason']
        if edit.get('activity'):c['activity']=edit['activity']
        adjusted.append(c)
    plans=adjusted
    write(AUDIT/'selection_plan.json',dict(clips=plans,rejected_sources=rejected,rejected_candidates=rejected_candidates,pending_sources=pending))
    print(json.dumps(dict(clips=len(plans),source_clips={s:sum(p['source']==s for p in plans) for s in ['additional_input','progait']},rejected_sources=len(rejected),pending_sources=len(pending))))

def boards():
    plans=json.loads((AUDIT/'selection_plan.json').read_text())['clips'];font=ImageFont.truetype(FONT,13)
    pending_only='pending' in sys.argv[2:]
    if pending_only:
        reviewed=json.loads((AUDIT/'visual_review.json').read_text())
        plans=[c for c in plans if reviewed.get(c['clip_id'],{}).get('status') not in ['accept','reject']]
    prefix=('final_' if 'final' in sys.argv[2:] else 'pending_') if pending_only else ''
    groups={}
    for c in plans:groups.setdefault(c['source'],[]).append(c)
    index=[]
    for group,clips in groups.items():
        for offset in range(0,len(clips),6):
            selected=clips[offset:offset+6];p=AUDIT/'candidate_review'/f'{prefix}{group}_{offset//6+1:03d}.jpg';p.parent.mkdir(exist_ok=True)
            stamps=[dict(clip_id=c['clip_id'],source_frames=np.linspace(c['source_first_frame'],c['source_end_frame_exclusive']-1,5).round().astype(int).tolist()) for c in selected]
            signature=hashlib.sha256(json.dumps(stamps).encode()).hexdigest();meta=p.with_suffix('.json')
            if meta.exists() and json.loads(meta.read_text()).get('signature')==signature and p.exists():
                index.append(dict(path=str(p.relative_to(ROOT)),clips=stamps));continue
            sheet=Image.new('RGB',(1500,276*len(selected)),'#15222e');draw=ImageDraw.Draw(sheet)
            for row,(c,record) in enumerate(zip(selected,stamps)):
                cap=cv2.VideoCapture(str(ROOT/c['original']))
                reference=load_reference(c)
                screen_data=json.loads((AUDIT/'screen'/f'{c["id"]}.json').read_text())
                dets=[r for r in screen_data['records'] if r.get('box')]
                draw.text((4,row*276+3),f"{c['clip_id']} | {c['activity']} {c['environment']} | {c['clip_duration_s']:.2f}s | QC context only; exported frames uncropped",font=font,fill='white')
                for col,n in enumerate(record['source_frames']):
                    cap.set(1,n);ok,bgr=cap.read();assert ok,(c['id'],n)
                    h,w=bgr.shape[:2];box=min(dets,key=lambda r:abs(r['frame']-n))['box'];bw,bh=box[2]-box[0],box[3]-box[1]
                    if reference is not None:
                        lo=reference[n,:,:2].min(0);hi=reference[n,:,:2].max(0)
                        box=[min(box[0],lo[0]),min(box[1],lo[1]),max(box[2],hi[0]),max(box[3],hi[1])]
                        bw,bh=box[2]-box[0],box[3]-box[1]
                    rect=[max(0,int(box[0]-.6*bw)),max(0,int(box[1]-.12*bh)),min(w,int(box[2]+.6*bw)),min(h,int(box[3]+.12*bh))]
                    x1,y1,x2,y2=rect;im=Image.fromarray(cv2.cvtColor(bgr[y1:y2,x1:x2],cv2.COLOR_BGR2RGB));im.thumbnail((296,226))
                    x=col*300+(300-im.width)//2;y=row*276+24;sheet.paste(im,(x,y))
                    if x1==0:draw.line((x,y,x,y+im.height-1),fill='#ffc344',width=3)
                    if y1==0:draw.line((x,y,x+im.width-1,y),fill='#ffc344',width=3)
                    if x2==w:draw.line((x+im.width-1,y,x+im.width-1,y+im.height-1),fill='#ffc344',width=3)
                    if y2==h:draw.line((x,y+im.height-1,x+im.width-1,y+im.height-1),fill='#ffc344',width=3)
                    draw.text((col*300+4,row*276+254),f'{n/fps:.2f}s' if (fps:=c['fps']) else str(n),font=font,fill='white')
                cap.release()
            sheet.save(p,quality=90);write(meta,dict(signature=signature,clips=stamps));index.append(dict(path=str(p.relative_to(ROOT)),clips=stamps))
            print(p.name,flush=True)
    write(AUDIT/f'candidate_review/{prefix}index.json',index)
    print('Boards',len(index),'clips',len(plans))

def encode_candidates():
    plans=json.loads((AUDIT/'selection_plan.json').read_text())['clips']
    reviewed=json.loads((AUDIT/'visual_review.json').read_text())
    plans=[c for c in plans if reviewed.get(c['clip_id'],{}).get('status')!='reject']
    staging=OUT/'staging';staging.mkdir(parents=True,exist_ok=True)
    def one(c):
        p=staging/(c['clip_id']+'.mp4');meta=p.with_suffix('.json')
        if p.exists() and meta.exists():return json.loads(meta.read_text())
        tmp=p.with_suffix('.part.mp4')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(ROOT/c['original']),'-map','0:v:0','-an',
            '-vf',f'trim=start_frame={c["source_first_frame"]}:end_frame={c["source_end_frame_exclusive"]},setpts=PTS-STARTPTS',
            '-c:v','h264_nvenc','-preset','p5','-rc','vbr','-cq','19','-b:v','0','-bf','0','-pix_fmt','yuv420p',
            '-fps_mode','passthrough','-movflags','+faststart','-y',str(tmp)],check=True,capture_output=True)
        d=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,width,height,avg_frame_rate','-of','json',str(tmp)]))['streams'][0]
        assert int(d['nb_read_frames'])==c['clip_frames'],(c['clip_id'],d)
        assert (d['width'],d['height'])==(c['width'],c['height'])
        tmp.replace(p);r=dict(c,encoded_sha256=sha(p),encoded_path=str(p.relative_to(ROOT)),
            decoded_frames=int(d['nb_read_frames']),transformations='Temporal trim only; native dimensions and cadence retained; H.264 re-encoded; audio removed; original privacy masking/watermarks retained.')
        write(meta,r);return r
    done=[]
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(one,c):c for c in plans}
        for f in cf.as_completed(futures):
            done.append(f.result())
            if len(done)%20==0:
                print('Encoded',len(done),'/',len(plans),flush=True)
                write(AUDIT/'encoding_status.json',dict(complete=len(done),planned=len(plans)))
    write(AUDIT/'encoding_status.json',dict(complete=len(done),planned=len(plans),status='complete_for_plan'))

if __name__=='__main__':
    if sys.argv[1]=='inventory':inventory()
    elif sys.argv[1]=='screen':screen()
    elif sys.argv[1]=='plan':plan()
    elif sys.argv[1]=='boards':boards()
    elif sys.argv[1]=='encode':encode_candidates()

"""Frame-preserving cuts for the expanded full-body source review."""
import hashlib
import json
import subprocess
import concurrent.futures
from pathlib import Path
from fractions import Fraction
import cv2
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/prepared/fullbody_locomotion_20261114'
RAW=ROOT/'data/raw/fullbody_locomotion_20261114'
AUDIT=ROOT/'audit/fullbody_locomotion_20261114'

def probe(p,count_frames=True):
    return json.loads(subprocess.check_output(['ffprobe','-v','error',*(['-count_frames'] if count_frames else []),'-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames','-of','json',str(p)]))['streams'][0]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()

def expansion_plans():
    overrides_path=AUDIT/'manual_interval_overrides.json'
    overrides=json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    for expansion in ['ljmu','movi']:
        plan=AUDIT/f'expansion_{expansion}_plan.json'
        if plan.exists():
            for c in json.loads(plan.read_text())['clips']:
                c.update(overrides.get(c['original']+':'+c['activity'],{}))
                yield c

def build():
    for d in ['clips','posters','qc']: (OUT/d).mkdir(parents=True,exist_ok=True)
    plans=[]
    def add(source,path,start,end,activity,view,subject,trial,role,note,environment='overground',label_basis='Visual source review'):
        plans.append(dict(source=source,original=str((RAW/path).relative_to(ROOT)),start_frame=start,end_frame_exclusive=end,
            activity=activity,view=view,subject=subject,trial_group=trial,role=role,notes=note,environment=environment,activity_label_basis=label_basis))
    lbmc_views={'26578':'side A','26579':'front','26580':'rear oblique A','26582':'rear oblique B',
                '26583':'rear','26584':'side B','26585':'front oblique A','26586':'front oblique B','26587':'front oblique C'}
    for cam,view in lbmc_views.items():
        add('lbmc',f'lbmc/{cam}.avi',180,900,'walk',view,'participant_02','lbmc_participant_02_gait','viewpoint_only',
            'Whole body in sampled frames. Publisher face blur and physical markers preserved. Medical screening not established. Public anonymized preview; confirm terms before training.',
            environment='treadmill',label_basis='Publisher gait trial description')
    for p in sorted((RAW/'ljmu360').glob('P1_*.mp4')):
        n=p.stem
        if 'E3T1_Front' in n:start,end=15,90
        elif 'E3T2_Front' in n:start,end=15,75
        elif 'E3T1_Rear' in n:start,end=160,240
        elif 'E3T2_Rear' in n:start,end=135,205
        elif n=='P1_E2T1_90':start,end=50,194
        else:
            count=int(probe(p)['nb_read_frames']);start,end=35,count-35
        view='front' if 'Front' in n else 'rear' if 'Rear' in n else 'oblique (45° source label)' if '_45' in n else 'side (90° source label)'
        add('ljmu360',f'ljmu360/{p.name}',start,end,'walk',view,'P1','ljmu360_'+n.rsplit('_',1)[0],
            'reported_healthy_cohort','Whole body selected away from entry/exit and close camera approach. Some backlighting and small person scale. Nominal source angle retained; opposite travel direction changes body-facing view. Camera files are independently trimmed: temporal alignment is not assumed.',
            label_basis='Publisher walking task / filename')
    for camera in [0,4]:
        add('mpi3dhp',f'mpi3dhp/S1_Seq1/video_{camera}.avi',425,525,'walk transition',f'camera {camera}; changing body orientation','S1','mpi3dhp_S1_Seq1_425_525',
            'viewpoint_only','Walking/standing block with gestures and turns; pose/view robustness only. Green floor and occasional bystander at frame edge. Not a clinical or ordinary-stride reference.')
    movi_plan=AUDIT/'movi_action_selection.json'
    if movi_plan.exists():
        for x in json.loads(movi_plan.read_text())['clips']:
            add('movi',f'movi/{x["file"]}',x['start_frame'],x['end_frame_exclusive'],x['activity'],x['view'],
                'Subject_2','movi_F_Subject_2_'+x['activity'],'viewpoint_only',x['notes'],
                environment=x.get('environment','overground'),label_basis=x['label_basis'])
    plans.extend(expansion_plans())
    (OUT/'selection_plan.json').write_text(json.dumps(plans,indent=2))
    previous={}
    if (OUT/'manifest.json').exists():
        previous={c['id']:c for c in json.loads((OUT/'manifest.json').read_text())['clips']}
    def prepare(x):
        original=ROOT/x['original']
        ident=f'{x["source"]}_{original.stem}_{x["start_frame"]}_{x["end_frame_exclusive"]}'
        dest=OUT/'clips'/f'{ident}.mp4'
        if ident in previous and dest.exists() and (OUT/previous[ident]['qc_sheet']).exists():
            old=previous[ident]
            if all(old.get(k)==x.get(k) for k in ['source','original','start_frame','end_frame_exclusive','subject','trial_group','activity']):
                return dict(old,**x)
        meta=probe(original,count_frames=False);fps=float(Fraction(meta['r_frame_rate']))
        if not dest.exists():
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(original),'-an','-vf',
                f'trim=start_frame={x["start_frame"]}:end_frame={x["end_frame_exclusive"]},setpts=PTS-STARTPTS',
                '-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-fps_mode','passthrough',
                '-movflags','+faststart',str(dest)],check=True)
        check=probe(dest);frames=int(check['nb_read_frames'])
        assert frames==x['end_frame_exclusive']-x['start_frame']
        assert (check['width'],check['height'])==(meta['width'],meta['height'])
        c=cv2.VideoCapture(str(dest));positions=sorted(set([0,frames-1]+list(range(0,frames,max(1,int(fps/2))))));cells=[]
        for frame in positions:
            c.set(1,frame);ok,img=c.read();assert ok
            im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im.thumbnail((320,230))
            tile=Image.new('RGB',(320,254),'#112431');tile.paste(im,((320-im.width)//2,0));ImageDraw.Draw(tile).text((4,234),f'{frame} / source {frame+x["start_frame"]}',fill='white');cells.append(tile)
        c.set(1,frames//2);ok,img=c.read();assert ok
        poster=OUT/'posters'/f'{ident}.jpg';cv2.imwrite(str(poster),img,[cv2.IMWRITE_JPEG_QUALITY,85]);c.release()
        sheet=Image.new('RGB',(320*5,254*((len(cells)+4)//5)),'#112431')
        for i,im in enumerate(cells):sheet.paste(im,((i%5)*320,(i//5)*254))
        sheet.save(OUT/'qc'/f'{ident}.jpg')
        result=dict(**x,id=ident,path=str(dest.relative_to(OUT)),poster=str(poster.relative_to(OUT)),
            qc_sheet=f'qc/{ident}.jpg',frames=frames,fps=fps,width=check['width'],height=check['height'],duration_s=frames/fps,
            source_start_s=x['start_frame']/fps,source_end_s=x['end_frame_exclusive']/fps,
            original_sha256=sha(original),sha256=sha(dest),clinical_form_label=None,pose_status='not_run',
            framing_status='pending_final_visual_review',all_frames_inside_frame_certified=False,
            automated_per_joint_visibility_checked=False)
        print('Prepared',ident,frames,flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        clips=list(pool.map(prepare,plans))
    manifest=dict(version=1,review_date='2026-11-14',source_catalog='source_catalog.json',
        note='Full-body framing candidates; natural self-occlusion remains possible. Pose and clinical labels not inferred. Same participant and all views must remain in one split.',clips=clips)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Prepared',len(clips),'clips;',sum(x['frames'] for x in clips),'frames')

if __name__=='__main__':build()

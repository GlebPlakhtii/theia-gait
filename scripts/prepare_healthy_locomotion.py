"""Prepare a small RGB reference collection without inventing good-form labels."""
import json
import subprocess
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'audit/healthy_locomotion_20261114'
OUT = ROOT/'data/prepared/healthy_locomotion_20261114'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024*1024): h.update(block)
    return h.hexdigest()


def prepare():
    (OUT/'clips').mkdir(parents=True, exist_ok=True)
    (OUT/'posters').mkdir(exist_ok=True)
    records = []
    for name in ['riglet', 'twente']:
        p = AUDIT/f'{name}_acquisition.json'
        if p.exists(): records.extend(json.loads(p.read_text())['files'])
    clips, sources = [], []
    for source in records:
        path = ROOT/source['local_path']
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(3)), int(cap.get(4))
        assert fps > 0 and n > 0
        intervals = []
        if source['source'] == 'riglet2024':
            # Black gaps are part of the published videoshow AVI. Do not join
            # separated passes or describe intervening black frames as motion.
            visible, frame = [], 0
            while True:
                ok, im = cap.read()
                if not ok: break
                visible.append(float(im[::8, ::8].mean()) > 2)
                frame += 1
            assert frame == n
            edge = np.diff(np.r_[False, visible, False].astype(int))
            starts, ends = np.where(edge == 1)[0], np.where(edge == -1)[0]
            intervals = [(int(a), int(b)) for a,b in zip(starts,ends) if b-a >= fps*.8]
            source['visible_intervals_frames'] = intervals
            source['nonblack_frames'] = int(sum(visible))
            intervals = sorted(sorted(intervals, key=lambda ab: ab[1]-ab[0], reverse=True)[:3])
        else:
            intervals = [(int(fps*20), min(n, int(fps*32)))]
        cap.release()
        source.update(fps=fps, native_frames=n, width=width, height=height, duration_s=n/fps)
        sources.append(source)
        stem = path.stem.replace(' ', '_')
        for seq,(start,end) in enumerate(intervals, 1):
            cid = f"{source['source']}_{source['subject']}_{stem}_{seq:02d}"
            dest = OUT/'clips'/f'{cid}.mp4'
            if not dest.exists():
                subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
                    '-i',str(path),'-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
                    '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                    '-fps_mode','passthrough','-movflags','+faststart',str(dest)],check=True)
            check = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
                '-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,avg_frame_rate',
                '-of','json',str(dest)]))['streams'][0]
            assert int(check['nb_read_frames']) == end-start
            cap = cv2.VideoCapture(str(dest))
            cap.set(cv2.CAP_PROP_POS_FRAMES,(end-start)//2)
            ok,im=cap.read();cap.release();assert ok
            poster = OUT/'posters'/f'{cid}.jpg'
            cv2.imwrite(str(poster), im, [cv2.IMWRITE_JPEG_QUALITY,85])
            clips.append({'id':cid,'source':source['source'],'subject':source['subject'],
                'trial_group':source.get('trial_group', f"riglet2024_{source['subject']}_Session1_{source['activity']}_comfortable"),
                'activity':source['activity'],'environment':source['environment'],
                'view':source.get('view','side'), 'speed_label':source.get('speed_label'),
                'speed_kmh':source.get('speed_kmh'), 'path':str(dest.relative_to(OUT)),
                'poster':str(poster.relative_to(OUT)), 'original_path':str(path.relative_to(ROOT)),
                'source_start_frame':start,'source_end_frame_exclusive':end,'fps':fps,
                'source_start_s':start/fps,'source_end_s':end/fps,'frames':end-start,
                'duration_s':(end-start)/fps,'width':width,'height':height,
                'sha256':digest(dest), 'frame_count_verified':True,
                'clinical_form_label':None,'pose_status':'not_run',
                'visual_qc':'pending', 'reference_role':'healthy_cohort_observed_motion'})
            print('Prepared',cid,end-start,'frames',flush=True)
    manifest={'created':'2026-11-14','sources':sources,'clips':clips,
        'clinical_labels_complete':False,'pose_inference_run':False,
        'scope':'Healthy-cohort candidate references; treadmill and overground kept separate.',
        'split_policy':'Group every participant, session and simultaneous camera into one split; never split neighboring clips randomly.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for group in ['riglet2024','twente2023']:
        items=[x for x in clips if x['source']==group]
        if not items:continue
        sheet=Image.new('RGB',(1200,230*((len(items)+3)//4)), '#15202b');draw=ImageDraw.Draw(sheet)
        for i,x in enumerate(items):
            im=Image.open(OUT/x['poster']);im.thumbnail((300,195))
            px,py=(i%4)*300,(i//4)*230
            sheet.paste(im,(px,py+35));draw.text((px+4,py+4),x['id'].replace(group+'_','')[:43],fill='white')
            draw.text((px+4,py+19),f"{x['view']} | {x['duration_s']:.2f}s",fill='#7edccc')
        sheet.save(AUDIT/f'{group}_prepared_contact.jpg')
    print('Total',len(sources),'recordings',len(clips),'clips')


if __name__=='__main__': prepare()

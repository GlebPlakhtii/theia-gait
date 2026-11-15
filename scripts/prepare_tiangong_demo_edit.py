"""Make an explicitly labeled flag-compositing derivative for the local demo.

Original Tiangong-2 source pixels, source review, and research data are retained.
The edit changes wall flags only; it is not input to inference or evaluation.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/space_form_search_20261115'
OUT = ROOT / 'results/gait_selected_paper_20261115/case_studies/tiangong2'
SOURCE = AUDIT / 'tiangong2_treadmill_candidate_1129_1143.mp4'


def flag_texture():
    # Vector construction: thirteen stripes and fifty five-point stars.
    w, h = 760, 400
    flag = np.empty((h, w, 3), np.uint8)
    for row in range(13):
        flag[round(row*h/13):round((row+1)*h/13)] = (46, 44, 157) if row % 2 == 0 else (205, 209, 215)
    cw, ch = round(w*.4), round(h*7/13)
    flag[:ch, :cw] = (105, 53, 34)
    for row in range(9):
        for col in range(6 if row % 2 == 0 else 5):
            x = cw*(col*2+(1 if row % 2 == 0 else 2))/12
            y = ch*(row+1)/10
            pts = []
            for k in range(10):
                a = -math.pi/2+k*math.pi/5
                r = h*.0308*(1 if k % 2 == 0 else .382)
                pts.append((round(x+r*math.cos(a)), round(y+r*math.sin(a))))
            cv2.fillPoly(flag, [np.array(pts, np.int32)], (205, 209, 215), cv2.LINE_AA)
    return flag


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SOURCE))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert len(frames) == 350 and frames[0].shape == (360, 480, 3)
    # Cuts were verified at frames 120 and 227; the middle shot has no wall flags.
    shots = [
        dict(start=0, end=120, quads=[
            [[30,116],[69,108],[76,182],[45,209]],
            [[364,80],[411,80],[411,137],[364,127]],
        ]),
        dict(start=227, end=350, quads=[
            [[7,47],[30,37],[39,118],[7,150]],
            [[398,6],[457,6],[457,69],[398,60]],
        ]),
    ]
    texture = flag_texture()
    th, tw = texture.shape[:2]
    texture_quad = np.float32([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]])
    tracks = []
    for shot in shots:
        ref = cv2.cvtColor(frames[shot['start']], cv2.COLOR_BGR2GRAY)
        for quad in shot['quads']:
            quad = np.float32(quad)
            x0, y0 = np.maximum(np.floor(quad.min(0)).astype(int)-3, 0)
            x1, y1 = np.minimum(np.ceil(quad.max(0)).astype(int)+4, [480,360])
            template = ref[y0:y1, x0:x1]
            offsets = []
            for i in range(shot['start'], shot['end']):
                crop = cv2.cvtColor(frames[i][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                warp = np.eye(2,3,dtype=np.float32)
                correlation, warp = cv2.findTransformECC(template, crop, warp, cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.0001),None,3)
                assert correlation > .88 and np.max(np.abs(warp[:,2])) < 5, (i,correlation,warp)
                offsets.append(warp[:,2].tolist())
            tracks.append(dict(start=shot['start'], end=shot['end'], quad=quad.tolist(), offsets=offsets))
    raw = OUT / 'edit_silent.mp4'
    pipe = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pix_fmt','bgr24','-s','480x396','-r','25','-i','pipe:0','-an','-c:v','libx264',
        '-crf','18','-preset','fast','-pix_fmt','yuv420p',str(raw)],stdin=subprocess.PIPE)
    for i, frame in enumerate(frames):
        edited = frame.copy()
        for track in tracks:
            if not track['start'] <= i < track['end']:
                continue
            quad = np.float32(track['quad']) + np.float32(track['offsets'][i-track['start']])
            transform = cv2.getPerspectiveTransform(texture_quad, quad)
            warped = cv2.warpPerspective(texture, transform, (480,360), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full((th,tw),255,np.uint8),transform,(480,360),flags=cv2.INTER_LINEAR)
            # Retain broadcast watermark strokes where the second shot's flag overlaps them.
            if track['start'] == 227 and quad[:,0].min() > 300:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                yy,xx = np.indices(mask.shape)
                watermark = (xx>=397)&(xx<=456)&(yy>=31)&(yy<=62)&(hsv[:,:,1]<85)&(hsv[:,:,2]>95)
                mask[watermark] = 0
            # Match local lighting, without importing the original flag's stars.
            illumination = cv2.GaussianBlur(frame[:,:,2],(25,25),0).astype(float)
            brightness = np.clip(illumination/145,.65,1.1)[...,None]
            warped = cv2.GaussianBlur(warped,(3,3),.45).astype(float)*brightness
            alpha = (mask.astype(float)/255)[...,None]
            edited = np.clip(edited*(1-alpha)+warped*alpha,0,255).astype(np.uint8)
        # A permanent footer keeps the manipulation and actual mission explicit in exports.
        footer = np.full((36,480,3),(33,22,9),np.uint8)
        cv2.putText(footer,'EDITED VISUALS - U.S. FLAGS',(10,14),cv2.FONT_HERSHEY_SIMPLEX,.37,(225,232,240),1,cv2.LINE_AA)
        cv2.putText(footer,'Tiangong-2, China | Original footage: CCTV',(10,29),cv2.FONT_HERSHEY_SIMPLEX,.33,(181,194,204),1,cv2.LINE_AA)
        output = np.vstack([edited,footer])
        pipe.stdin.write(output.tobytes())
        if i in [0,100,125,230,250,325,349]:
            cv2.imwrite(str(AUDIT/f'edited_review_{i:03d}.jpg'),output)
        if i == 250:
            cv2.imwrite(str(OUT/'edited_poster.jpg'),output)
    pipe.stdin.close()
    assert pipe.wait() == 0
    dest = OUT/'edited_us_flags.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(raw),'-i',str(SOURCE),
        '-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',str(dest)],check=True)
    raw.unlink()
    import shutil
    shutil.copy2(SOURCE,OUT/'original.mp4')
    shutil.copy2(AUDIT/'tiangong2_treadmill_candidate_poster.jpg',OUT/'original_poster.jpg')
    review=json.loads((AUDIT/'source_review.json').read_text())
    review['demo_edit']={
        'requested_change':'Replace visible wall flags with U.S. flags for the demo.',
        'edited_file':dest.name,'original_file':'original.mp4',
        'label':'EDITED VISUALS - U.S. FLAGS; Tiangong-2, China | Original footage: CCTV',
        'method':'Vector flag compositing, planar perspective warp and per-frame translation registration; original audio retained.',
        'frames':350,'fps':25,'duration_s':14,'source_width':480,'source_height':360,'edited_height_with_label':396,
        'inference_or_evaluation_input':False,'clinical_label':None,
        'original_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'edited_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'tracks':tracks,
    }
    (OUT/'source_review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in review['demo_edit'].items() if k!='tracks'},indent=2))


if __name__=='__main__':
    main()

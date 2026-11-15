"""Five-frame context crops for manual framing review, grouped by participant.

Only review boards are cropped. Prepared RGB videos keep original dimensions.
"""
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from prepare_fullbody_references import ROOT,AUDIT,expansion_plans


def build():
    directory=AUDIT/'expansion_review';directory.mkdir(exist_ok=True)
    groups=defaultdict(list)
    for c in expansion_plans():groups[(c['source'],c['subject'])].append(c)
    previous_path=directory/'index.json'
    previous={b['path']:b for b in json.loads(previous_path.read_text())} if previous_path.exists() else {}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    boards=[]
    for (source,subject),clips in sorted(groups.items()):
        for chunk in range(0,len(clips),6):
            selected=clips[chunk:chunk+6]
            path=directory/f'{source}_{subject}_{chunk//6+1}.jpg'
            expected=[f'{c["source"]}_{Path(c["original"]).stem}_{c["start_frame"]}_{c["end_frame_exclusive"]}' for c in selected]
            prior=previous.get(str(path.relative_to(ROOT)))
            if prior and [r['id'] for r in prior['clips']]==expected and path.exists():
                boards.append(prior);continue
            sheet=Image.new('RGB',(1280,246*len(selected)),'#122330');draw=ImageDraw.Draw(sheet)
            records=[]
            for row,c in enumerate(selected):
                ident=f'{c["source"]}_{Path(c["original"]).stem}_{c["start_frame"]}_{c["end_frame_exclusive"]}'
                screen=json.loads((ROOT/c['framing_screen']).read_text())
                detections=[r for r in screen['frames'] if 'box' in r]
                frames=np.linspace(c['start_frame'],c['end_frame_exclusive']-1,5).round().astype(int).tolist()
                cap=cv2.VideoCapture(str(ROOT/c['original']))
                draw.text((5,row*246+3),ident+' | '+c['activity']+' | expanded person context; original videos are NOT cropped',font=font,fill='white')
                for col,n in enumerate(frames):
                    cap.set(1,n);ok,im=cap.read();assert ok
                    h,w=im.shape[:2];b=min(detections,key=lambda r:abs(r['frame']-n))['box']
                    bw,bh=b[2]-b[0],b[3]-b[1]
                    rect=[max(0,int(b[0]-.6*bw)),max(0,int(b[1]-.15*bh)),min(w,int(b[2]+.6*bw)),min(h,int(b[3]+.15*bh))]
                    a,t,z,v=rect
                    picture=Image.fromarray(cv2.cvtColor(im[t:v,a:z],cv2.COLOR_BGR2RGB));picture.thumbnail((250,195))
                    x=col*256+(256-picture.width)//2;y=row*246+24
                    sheet.paste(picture,(x,y))
                    # Yellow marks when the QC window meets an original image edge.
                    if a==0:draw.line((x,y,x,y+picture.height-1),fill='#ffcb60',width=3)
                    if z==w:draw.line((x+picture.width-1,y,x+picture.width-1,y+picture.height-1),fill='#ffcb60',width=3)
                    if t==0:draw.line((x,y,x+picture.width-1,y),fill='#ffcb60',width=3)
                    if v==h:draw.line((x,y+picture.height-1,x+picture.width-1,y+picture.height-1),fill='#ffcb60',width=3)
                    draw.text((col*256+5,row*246+222),f'source {n}; crop {a},{t}–{z},{v}',font=font,fill='#c8dce9')
                cap.release();records.append(dict(id=ident,source_frames=frames))
            sheet.save(path,quality=91)
            boards.append(dict(path=str(path.relative_to(ROOT)),clips=records))
    (directory/'index.json').write_text(json.dumps(boards,indent=2))
    print('Review boards:',len(boards),'clips:',sum(len(b['clips']) for b in boards))

if __name__=='__main__':build()

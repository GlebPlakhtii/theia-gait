"""Create local pose-review figures and an HTML index; no decision/clinical labels."""
import argparse,csv,html,json,os
from pathlib import Path
from urllib.parse import quote
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps,ImageDraw


def angle(a,b,c):
    u,v=a-b,c-b
    den=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
    cos=np.divide((u*v).sum(-1),den,out=np.full(den.shape,np.nan),where=den>1e-8)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--queue',default='data/manifests/pose_queue_ready.csv');a=p.parse_args()
    names={n:int(i) for i,n in json.loads((a.directory/'joint_schema.json').read_text())['all_308'].items()}
    parts=[]
    for q in csv.DictReader(open(a.queue)):
        cid=q['clip_id'];d=a.directory/cid
        if not (d/'complete.json').exists():continue
        raw=np.load(d/'raw_308.npz');pts=raw['keypoints'];scores=raw['scores'];ts=raw['timestamp_s']
        fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
        for side,col in [('left','#16834b'),('right','#d66a16')]:
            idx=[names[f'{side}_{j}'] for j in ('hip','knee','ankle')]
            angles=angle(*(pts[:,i] for i in idx)); conf=scores[:,idx].min(axis=1)
            angles[conf<.3]=np.nan
            axes[0].plot(ts,angles,label=side,color=col,lw=1)
            axes[1].plot(ts,conf,label=f'{side} hip/knee/ankle minimum',color=col,lw=1)
        axes[0].set(ylabel='Projected knee angle (degrees)',ylim=(0,185))
        axes[0].legend(loc='lower left');axes[1].axhline(.3,color='gray',ls='--',lw=.7)
        axes[1].set(ylabel='Raw heatmap score',xlabel='Native source timestamp (s)')
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(cid+'\nPredicted 2D geometry; visibility and anatomical accuracy unvalidated',fontsize=10)
        fig.savefig(d/'signals.png',dpi=160);plt.close(fig)
        previews=sorted(d.glob('preview_*.jpg'))
        if previews:
            thumbs=[]
            for pth in previews:
                im=Image.open(pth).convert('RGB');im.thumbnail((360,420))
                canvas=Image.new('RGB',(380,450),'white');canvas.paste(im,((380-im.width)//2,23))
                ImageDraw.Draw(canvas).text((10,5),pth.stem,fill='black');thumbs.append(canvas)
            contact=Image.new('RGB',(380*3,450*((len(thumbs)+2)//3)), '#eeeeee')
            for i,im in enumerate(thumbs):contact.paste(im,((i%3)*380,(i//3)*450))
            contact.save(d/'contact.jpg',quality=94)
        src=quote(os.path.relpath(q['local_path'],a.directory))
        src+=f"#t={q['start_s']},{q['end_s']}"
        parts.append(f'''<section id="{html.escape(cid)}"><h2>{html.escape(cid)}</h2>
<p><a href="{src}">Original RGB video</a> · <a href="{cid}/poses.csv">Body/feet CSV</a> · <a href="{cid}/raw_308.npz">Raw predictions</a></p>
<details><summary>Play original source (queued interval)</summary><video controls preload="none" src="{src}"></video></details>
<p>Derived preview overlays:</p><a href="{cid}/contact.jpg"><img src="{cid}/contact.jpg" loading="lazy"></a>
<a href="{cid}/signals.png"><img src="{cid}/signals.png" loading="lazy"></a></section>''')
    page='''<!doctype html><meta charset="utf-8"><title>Theia pose review</title>
<style>body{font:16px system-ui;max-width:1120px;margin:36px auto;padding:0 20px;color:#16252b;background:#f7f9fa}section{background:white;padding:24px;margin:24px 0;border:1px solid #dce4e7;border-radius:8px}h2{font-size:18px;overflow-wrap:anywhere}img{max-width:100%;height:auto}video{max-width:100%;max-height:560px}a{color:#076273}p{line-height:1.5}</style>
<h1>Theia — pose review</h1><p>Local Sapiens2-5B extraction. Original RGB is unchanged; overlays are derived review images. The source cohort is general fitness with undocumented health status. ARED is development footage.</p>
<p><strong>No clinical transfer or decision benefit is established.</strong> Scores are not visibility probabilities. A 0.3 score cutoff only hides low-score curve segments; it does not validate the others. Projected knee angles depend on viewpoint and are not anatomical flexion, load, pain or fatigue. Occluded joints may be confidently predicted. No interpolation or temporal smoothing is applied.</p>
<p><a href="poses_all.csv">Combined body/feet CSV</a> · <a href="integrity_report.json">Integrity and coverage checks</a> · <a href="run.json">Execution provenance</a></p>'''+''.join(parts)
    (a.directory/'index.html').write_text(page)
    print(a.directory/'index.html')

if __name__=='__main__':main()

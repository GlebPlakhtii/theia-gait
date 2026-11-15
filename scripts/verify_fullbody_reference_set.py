"""Verify clip integrity, identity, color modality, frame mapping and HTTP playback."""
import concurrent.futures
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.request import Request,urlopen
import cv2
import numpy as np
from prepare_fullbody_references import ROOT,OUT,AUDIT,probe


def verify():
    manifest=json.loads((OUT/'manifest.json').read_text())
    queue=json.loads((OUT/'pose_input_candidates.json').read_text())
    clips=queue['clips']
    assert len(clips)>=100
    assert len({c['id'] for c in clips})==len(clips)
    assert len({c['sha256'] for c in clips})==len(clips),'Duplicate video bytes'
    assert all(c['framing_status']=='sampled_full_body_pass' for c in clips)
    assert all(c.get('color_modality')!='grayscale' for c in clips)
    assert len(list((OUT/'pose_inputs').glob('*.mp4')))==len(clips)
    for c in clips:
        assert (OUT/'pose_inputs'/Path(c['path']).name).resolve()==(OUT/c['path']).resolve()
    def check(c):
        path=OUT/c['path'];p=probe(path)
        assert int(p['nb_read_frames'])==c['frames']==c['end_frame_exclusive']-c['start_frame']
        assert (p['width'],p['height'])==(c['width'],c['height'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256']
        original=cv2.VideoCapture(str(ROOT/c['original']));derived=cv2.VideoCapture(str(path));samples=[]
        for n in sorted(set([0,c['frames']//2,c['frames']-1])):
            original.set(1,c['start_frame']+n);ok,a=original.read();assert ok
            derived.set(1,n);ok,b=derived.read();assert ok and a.shape==b.shape
            delta=a.astype(np.float32)-b.astype(np.float32);mse=float(np.mean(delta*delta))
            psnr=10*math.log10(255**2/max(mse,1e-10));assert psnr>28,(c['id'],n,psnr)
            channel_range=float((np.max(b,axis=2).astype(np.int16)-np.min(b,axis=2)).mean())
            samples.append(dict(clip_frame=n,source_frame=c['start_frame']+n,psnr_db=round(psnr,3),mean_channel_range=round(channel_range,3)))
        original.release();derived.release()
        assert max(s['mean_channel_range'] for s in samples)>1,c['id']
        with urlopen(Request('http://localhost:8006/fullbody-reference/'+c['path'],headers={'Range':'bytes=0-1023'}),timeout=10) as r:
            assert r.status==206 and len(r.read())==1024
        return dict(id=c['id'],decoded_frame_count=c['frames'],sha256_matched=True,http_range_status=206,samples=samples)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(check,clips):
            records.append(result)
            if len(records)%20==0:print('Verified',len(records),'/',len(clips),flush=True)
    page=(OUT/'index.html').read_text()
    js=re.search(r'<script>(.*?)</script>',page,re.S).group(1)
    (AUDIT/'gallery_script.js').write_text(js)
    subprocess.run(['node','--check',str(AUDIT/'gallery_script.js')],check=True)
    for name in ['', 'manifest.json','source_catalog.json','pose_input_candidates.json']:
        with urlopen('http://localhost:8006/fullbody-reference/'+name,timeout=10) as r:assert r.status==200
    report=dict(all_passed=True,accepted_clips=len(clips),prepared_candidates=len(manifest['clips']),
        withheld_prepared=len(manifest['clips'])-len(clips),
        participants=len({(c['source'],c['subject']) for c in clips}),
        trial_groups=len({c['trial_group'] for c in clips}),
        total_frames=sum(c['frames'] for c in clips),video_seconds=sum(c['duration_s'] for c in clips),
        sources=dict(Counter(c['source'] for c in clips)),activities=dict(Counter(c['activity'] for c in clips)),
        environments=dict(Counter(c['environment'] for c in clips)),
        published_healthy_cohort_clips=sum(c['role']=='reported_healthy_cohort' for c in clips),
        unique_sha256_count=len({c['sha256'] for c in clips}),
        limitation='Framing reviewed in sampled source frames. Pixel/frame mapping checks do not verify every-frame joint visibility or correct form.',
        full_rate_pose_analysis=False,clinical_form_labels=False,records=records)
    (AUDIT/'expansion_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':verify()

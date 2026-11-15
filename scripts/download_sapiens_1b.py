"""Pinned public checkpoint download with verified HTTP ranges and SHA256."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import argparse

DEST = Path(__file__).resolve().parents[1]/'models/sapiens2-1b'
REV = 'f5fed8b97b99698d5eea1d14ff0855d0b4c3f000'
SHA = '2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f'
SIZE = 6079194752
NAME = 'sapiens2_1b_pose.safetensors'

def main():
    global DEST, REV, SHA, SIZE, NAME
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['1b','0.4b'],default='1b');a=parser.parse_args()
    if a.variant=='0.4b':
        DEST=DEST.parent/'sapiens2-0.4b';REV='73572e121602e8a6eb40eb841732e88f386d250b'
        SHA='f1b7b0e5423d667634a355dd86a14c5b03d243f2ac77943301c7db9f93329106';SIZE=1702754120
        NAME='sapiens2_0.4b_pose.safetensors'
    DEST.mkdir(exist_ok=True,parents=True)
    path=DEST/(NAME+'.ranges.part');fd=os.open(path,os.O_CREAT|os.O_RDWR,0o644)
    os.ftruncate(fd,SIZE);chunk=32*1024*1024
    progress=DEST/'download_ranges.json'
    done=set(json.loads(progress.read_text())) if progress.exists() else set()
    def fetch(start):
        end=min(start+chunk,SIZE)-1
        url=f'https://huggingface.co/facebook/sapiens2-pose-{a.variant}/resolve/{REV}/{NAME}?range_start={start}'
        for attempt in range(5):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=90) as r:
                    assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{SIZE}'
                    data=r.read()
                assert len(data)==end-start+1
                written=os.pwrite(fd,data,start);assert written==len(data)
                return start
            except Exception:
                if attempt==4:raise
                time.sleep(min(2**attempt,8))
    starttime=time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(fetch,s) for s in range(0,SIZE,chunk) if s not in done]
            for f in concurrent.futures.as_completed(futures):
                done.add(f.result());progress.write_text(json.dumps(sorted(done)))
                if len(done)%8==0:print(f'{len(done)*chunk/1e9:.2f}/{SIZE/1e9:.2f} GB, {time.monotonic()-starttime:.0f}s',flush=True)
        os.fsync(fd)
    finally:os.close(fd)
    assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==SHA
    path.replace(DEST/NAME)
    (DEST/'provenance.json').write_text(json.dumps(dict(repository=f'facebook/sapiens2-pose-{a.variant}',revision=REV,sha256=SHA,bytes=SIZE),indent=2))
    print('Verified checkpoint complete',flush=True)

if __name__=='__main__':main()

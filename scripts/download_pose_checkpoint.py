"""Resume the pinned public checkpoint with bounded, checked HTTP range parts."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

URL='https://huggingface.co/facebook/sapiens2-pose-5b/resolve/ada1f29aa1fd454ca28665c700923a0101b6b24f/sapiens2_5b_pose.safetensors'
TOTAL=20480899148
SHA='b4848da8691c72e14d3ff71319f077363107129bf4128019eb39d072129b2a52'
DIR=Path('models/sapiens2-5b')
FINAL=DIR/'sapiens2_5b_pose.safetensors'
PART=FINAL.with_suffix('.safetensors.part')
PLAN=DIR/'range_plan.json'


def get_part(item):
    start,end=item
    p=DIR/f'tail_{start}_{end}.part'
    expected=end-start+1
    for attempt in range(4):
        size=p.stat().st_size if p.exists() else 0
        assert size<=expected
        if size==expected:return p
        lo=start+size
        try:
            request=urllib.request.Request(URL,headers={'Range':f'bytes={lo}-{end}','Accept-Encoding':'identity'})
            with urllib.request.urlopen(request,timeout=120) as r:
                assert r.status==206,r.status
                assert r.headers['Content-Range']==f'bytes {lo}-{end}/{TOTAL}',r.headers
                with p.open('ab') as f:
                    while data:=r.read(1024*1024):
                        if f.tell()+len(data)>expected:raise ValueError('Range response exceeded requested size')
                        f.write(data)
            assert p.stat().st_size==expected
            print('Verified range length',start,end,flush=True)
            return p
        except Exception as e:
            print('Range retry',start,type(e).__name__,str(e)[:120],flush=True)
            if attempt==3:raise
            time.sleep(2)


def main():
    if FINAL.exists():
        assert hashlib.file_digest(FINAL.open('rb'),'sha256').hexdigest()==SHA
        return
    DIR.mkdir(parents=True,exist_ok=True)
    if not PLAN.exists():
        start=PART.stat().st_size if PART.exists() else 0
        chunks=[[lo,min(lo+1500000000,TOTAL)-1] for lo in range(start,TOTAL,1500000000)]
        PLAN.write_text(json.dumps({'initial_prefix_bytes':start,'total':TOTAL,'chunks':chunks},indent=2))
    plan=json.loads(PLAN.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts=list(pool.map(get_part,plan['chunks']))
    # Resume an interrupted assembly at the exact existing contiguous byte offset.
    with PART.open('ab') as out:
        for (start,end),p in zip(plan['chunks'],parts):
            pos=out.tell()
            if pos>end:continue
            assert start<=pos<=end
            with p.open('rb') as src:
                src.seek(pos-start)
                while block:=src.read(8*1024*1024):out.write(block)
    assert PART.stat().st_size==TOTAL
    digest=hashlib.file_digest(PART.open('rb'),'sha256').hexdigest()
    assert digest==SHA,digest
    PART.rename(FINAL)
    print('Complete checkpoint SHA-256 verified',digest,flush=True)
    # Remove only this task's now-redundant range pieces after full verification.
    for p in parts:p.unlink()

if __name__=='__main__':main()

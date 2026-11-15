"""Sparse keyframe perceptual-hash screening; matches are only review candidates."""
import concurrent.futures
import json
from pathlib import Path
import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit/new_video_intake_20261114'


def hashes(record):
    rows=[]
    with av.open(record['local_path']) as container:
        stream=container.streams.video[0];stream.codec_context.skip_frame='NONKEY'
        for frame in container.decode(stream):
            gray=frame.to_ndarray(format='gray')
            if float(gray.std())<18:continue
            for k in range(4):
                small=cv2.resize(np.rot90(gray,k),(32,32)).astype(np.float32)
                coeff=cv2.dct(small)[:8,:8].ravel()[1:]
                code=sum(int(v>np.median(coeff))<<i for i,v in enumerate(coeff))
                rows.append(dict(id=record['id'],pts_s=float(frame.pts*frame.time_base),rotation_quarters=k,hash=code))
    return rows


if __name__=='__main__':
    sources=json.load(open(ROOT/'data/manifests/exercise_intake_20261114.json'))['sources']
    sources += [dict(id='existing_adenot_2026',local_path=str(ROOT/'data/raw/esa_adenot_20260809_ared.mp4')),
                dict(id='existing_gerst_2014',local_path=str(ROOT/'data/raw/reserved_candidates/gerst_2014_ared_web.mp4'))]
    with concurrent.futures.ThreadPoolExecutor(3) as pool:groups=list(pool.map(hashes,sources))
    matches=[];counts={}
    for i,aa in enumerate(groups):
        counts[sources[i]['id']]=len(aa)//4
        for j in range(i):
            bb=groups[j]
            if not aa or not bb:continue
            codes=np.array([b['hash'] for b in bb],dtype=np.uint64)
            for a in aa:
                if a['rotation_quarters']!=0:continue
                distances=np.bitwise_count(codes ^ np.uint64(a['hash']))
                for index in np.where(distances<=5)[0]:
                    b=bb[int(index)];matches.append(dict(a={k:v for k,v in a.items() if k!='hash'},b={k:v for k,v in b.items() if k!='hash'},hamming_distance=int(distances[index])))
    result=dict(method='Sparse decoded keyframes; 63-bit DCT perceptual hash; four rigid orientations; Hamming distance <=5.',
        limitations='Candidates require visual confirmation. No matches cannot establish independence; crops, edits and sampling can hide reuse. Repeated crew identity must be grouped regardless.',
        keyframes_per_source=counts,candidates=matches)
    (OUT/'visual_duplicate_candidates.json').write_text(json.dumps(result,indent=2))
    print('Keyframes',sum(counts.values()),'candidate pairs',len(matches),flush=True)

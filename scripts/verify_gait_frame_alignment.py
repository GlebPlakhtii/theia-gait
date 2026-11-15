"""Check original-to-cut frame correspondence in every curated video.

Samples beginning/middle/end; uses the existing curation PSNR gate (28 dB).
This is a temporal/crop provenance check, not a clinical or pose-quality test.
"""
from collections import Counter
from datetime import datetime,timezone
import fcntl,hashlib,json,time
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/prepared/gait_curated_20261114'
OUT=ROOT/'results/gait_metrics_20261114'
TARGET=OUT/'source_frame_alignment_audit.json'


def stamp(path):
    stat=path.stat()
    return dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def write(rows,manifest_hash,expected):
    counts=dict(Counter(r['status'] for r in rows.values()))
    completed=len(rows)==expected
    values=[score for r in rows.values() for score in r.get('psnr_db',[])]
    data=dict(updated_at=datetime.now(timezone.utc).isoformat(),curated_manifest_sha256=manifest_hash,
        expected_clips=expected,checked_clips=len(rows),sampled_frame_pairs=len(values),counts=counts,
        status='complete' if completed and not counts.get('failed') else 'failed' if counts.get('failed') else 'running',
        minimum_psnr_db=min(values) if values else None,required_psnr_db=28,
        gate_source='Existing scripts/verify_gait_curation.py temporal alignment criterion; not fitted to gait outcome differences.',
        evidence='Beginning, middle and last image of every clip compared with its mapped original frame. Dimensions must match; PSNR must exceed 28 dB.',
        limitation='Three sampled frames per clip, not certification of every intermediate frame. Global-image PSNR does not establish pose accuracy or physical playback timing.',
        clips=rows)
    temporary=TARGET.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2,allow_nan=False));temporary.replace(TARGET)


def main():
    cv2.setNumThreads(1)
    lock=(OUT/'.source_frame_alignment.lock').open('w')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=BASE/'manifest.json';manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    clips=json.loads(manifest.read_text())['clips']
    old=json.loads(TARGET.read_text()) if TARGET.exists() else {}
    rows=old.get('clips',{}) if old.get('curated_manifest_sha256')==manifest_hash else {}
    start=time.monotonic()
    for number,clip in enumerate(clips,1):
        cid=clip['clip_id'];source_path=ROOT/clip['original'];cut_path=BASE/clip['video']
        source_stamp=stamp(source_path);cut_stamp=stamp(cut_path)
        prior=rows.get(cid,{})
        if prior.get('status')=='passed' and prior.get('source_stat')==source_stamp and prior.get('cut_stat')==cut_stamp:continue
        result=dict(source=clip['source'],source_stat=source_stamp,cut_stat=cut_stamp,
                    original=str(source_path),cut=str(cut_path),source_first_frame=clip['source_first_frame'])
        source=cv2.VideoCapture(str(source_path));cut=cv2.VideoCapture(str(cut_path))
        try:
            assert source.isOpened() and cut.isOpened(),'Video could not be opened'
            frames=[0,clip['clip_frames']//2,clip['clip_frames']-1];scores=[]
            for frame in frames:
                source.set(cv2.CAP_PROP_POS_FRAMES,clip['source_first_frame']+frame)
                cut.set(cv2.CAP_PROP_POS_FRAMES,frame)
                a_ok,a=source.read();b_ok,b=cut.read()
                assert a_ok and b_ok,f'Could not decode mapped frame {frame}'
                assert a.shape==b.shape==(clip['height'],clip['width'],3),'Source and cut geometry differ'
                scores.append(float(cv2.PSNR(a,b)))
            result.update(clip_frames=frames,original_frames=[clip['source_first_frame']+f for f in frames],psnr_db=scores)
            assert min(scores)>28,'Mapped source/cut pixel similarity is below the existing curation gate'
            assert source_stamp==stamp(source_path) and cut_stamp==stamp(cut_path),'A video changed during verification'
            result['status']='passed'
        except Exception as error:
            result.update(status='failed',error=type(error).__name__,reason=str(error))
        finally:
            source.release();cut.release()
        rows[cid]=result
        if number%20==0:
            write(rows,manifest_hash,len(clips))
            print(f'Source alignment {len(rows)}/{len(clips)}; failures {sum(r["status"]=="failed" for r in rows.values())}; elapsed {time.monotonic()-start:.1f}s',flush=True)
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==manifest_hash,'Manifest changed during verification'
    write(rows,manifest_hash,len(clips))
    if any(r['status']!='passed' for r in rows.values()):raise SystemExit(1)
    print(f'All {len(clips)} original-to-cut mappings passed sampled pixel alignment',flush=True)


if __name__=='__main__':main()

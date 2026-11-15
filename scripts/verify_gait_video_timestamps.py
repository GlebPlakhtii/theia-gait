"""Independently decode curated videos and compare native PTS to pose inputs."""
from collections import Counter
from datetime import datetime,timezone
import csv,fcntl,hashlib,json,time
from pathlib import Path
import av,numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'data/prepared/gait_curated_20261114';OUT=ROOT/'results/gait_metrics_20261114'


def write(rows):
    counts=dict(Counter(r['status'] for r in rows.values()))
    result=dict(updated_at=datetime.now(timezone.utc).isoformat(),expected_clips=596,checked_clips=len(rows),counts=counts,
        status='complete' if len(rows)==596 and not counts.get('failed') else 'failed' if counts.get('failed') else 'running',
        evidence='Fresh full decode of each curated clip; image count/dimensions, monotonically increasing presentation timestamps, current video SHA256, and exact queue or publisher-pose timing.',
        limitation='Source-to-cut alignment remains the prior independent sampled pixel audit. Playback speed is not a physical clock calibration.',clips=rows)
    p=OUT/'video_timestamp_audit.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(p)


def main():
    lock=(OUT/'.video_timestamp_audit.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    clips=json.loads((BASE/'manifest.json').read_text())['clips'];queue={r['clip_id']:r for r in csv.DictReader((OUT/'production_queue.csv').open())}
    target=OUT/'video_timestamp_audit.json';rows=json.loads(target.read_text())['clips'] if target.exists() else {}
    start=time.monotonic()
    for i,c in enumerate(clips):
        cid=c['clip_id'];path=Path(c['video_absolute']);stamp=dict(size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        previous=rows.get(cid,{})
        if previous.get('status')=='passed' and previous.get('video_stat')==stamp and previous.get('sha256')==c['sha256']:continue
        try:
            with path.open('rb') as f:actual_hash=hashlib.file_digest(f,'sha256').hexdigest()
            assert actual_hash==c['sha256'],'video hash changed'
            times=[];pts=[]
            with av.open(path) as con:
                stream=con.streams.video[0];stream.thread_count=1
                for frame in con.decode(video=0):
                    assert frame.width==c['width'] and frame.height==c['height'],'image dimensions changed'
                    assert frame.pts is not None,'missing presentation timestamp'
                    times.append(float(frame.pts*frame.time_base));pts.append(int(frame.pts))
            times=np.asarray(times);assert len(times)==c['clip_frames'],'decoded image count'
            assert np.all(np.diff(times)>0),'nonmonotonic presentation time'
            assert abs(times[0])<1e-9,'nonzero clip start'
            relative=times-times[0]
            errors={}
            if cid in queue:
                frame_map=list(csv.DictReader(open(queue[cid]['frame_map'])))
                np.testing.assert_array_equal([int(r['source_frame_index']) for r in frame_map],np.arange(len(times)))
                np.testing.assert_array_equal([int(r['source_pts']) for r in frame_map],pts)
                expected=np.array([float(r['timestamp_s']) for r in frame_map])
                np.testing.assert_allclose(relative,expected,atol=1e-10,rtol=0)
                errors['queue_max_time_difference_s']=float(np.max(abs(relative-expected)))
            if c.get('publisher_pose'):
                with np.load(BASE/c['publisher_pose'],allow_pickle=False) as a:
                    np.testing.assert_array_equal(a['source_frame'],np.arange(c['source_first_frame'],c['source_end_frame_exclusive']))
                    expected=a['source_time_s']-a['source_time_s'][0]
                    np.testing.assert_allclose(relative,expected,atol=1e-6,rtol=0)
                    errors['publisher_pose_max_time_difference_s']=float(np.max(abs(relative-expected)))
            rows[cid]=dict(status='passed',source=c['source'],frames=len(times),width=c['width'],height=c['height'],
                sha256=actual_hash,video_stat=stamp,first_pts=pts[0],last_pts=pts[-1],last_time_s=float(times[-1]),
                median_delta_time_s=float(np.median(np.diff(times))),**errors)
        except Exception as e:
            rows[cid]=dict(status='failed',source=c['source'],video_stat=stamp,error=type(e).__name__,reason=str(e)[:1000])
        if (i+1)%20==0:
            write(rows);print(f'Timestamp audit {len(rows)}/596; failures {sum(r["status"]=="failed" for r in rows.values())}; elapsed {time.monotonic()-start:.1f}s',flush=True)
    write(rows)
    if any(r['status']!='passed' for r in rows.values()):raise SystemExit(1)
    print('All 596 native videos fully decoded and timestamps verified',flush=True)


if __name__=='__main__':main()

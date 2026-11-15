"""Run the frozen native-frame batch and persist live progress/failure status."""
import csv
import fcntl
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gait_metrics_20261114'


def write(data):
    data['updated_at']=datetime.now(timezone.utc).isoformat()
    p=OUT/'production_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)


def main():
    controller_lock=(OUT/'.production_controller.lock').open('w')
    try:fcntl.flock(controller_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Production controller is already running; existing job was not changed.')
    # Also detect a runner launched before this controller lock was introduced.
    existing_lock=OUT/'poses_sapiens2_1b/.inference.lock'
    if existing_lock.exists():
        with existing_lock.open('a') as handle:
            try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise SystemExit('Native inference is already running; existing job and status were not changed.')
    selected=json.loads((OUT/'model_benchmark/model_selection.json').read_text())
    assert selected['selected']=='1b' and all(selected['gates'].values())
    assert json.loads((OUT/'pilot_review.json').read_text())['status']=='sampled_overlay_check_passed'
    inv=json.loads((OUT/'production_inventory.json').read_text());dest=OUT/'poses_sapiens2_1b'
    queue=list(csv.DictReader((OUT/'production_queue.csv').open()))
    start=time.monotonic();state=dict(phase='native_pose_inference',status='running',**inv)
    command=[sys.executable,'-u','scripts/run_sapiens2_pose.py','--model-variant','1b','--queue',str(OUT/'production_queue.csv'),
        '--rois',str(OUT/'production_rois.json'),'--output',str(dest),'--checkpoint-every','50']
    counts={};initial=0;initial_complete=0
    for q in queue:
        p=dest/q['clip_id']/'complete.json'
        if p.exists():
            counts[q['clip_id']]=json.loads(p.read_text())['frames'];initial_complete+=1
        else:
            checkpoint=p.parent/'checkpoint_308.npz'
            if checkpoint.exists():
                with np.load(checkpoint,allow_pickle=False) as a:counts[q['clip_id']]=len(a['source_frame_index'])
    initial=sum(counts.values())
    write(dict(state,completed_frames=initial,completed_clips=initial_complete,runner_command=command))
    with (ROOT/'audit/gait_production_pose.log').open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        write(dict(state,runner_pid=process.pid,completed_frames=initial,completed_clips=initial_complete))
        for line in process.stdout:
            log.write(line);log.flush()
            match=re.match(r'^(\S+) (\d+)/(\d+) last=',line)
            if match:
                cid,n,total=match.groups();counts[cid]=int(n)
                complete={p.parent.name:json.loads(p.read_text())['frames'] for p in dest.glob('*/complete.json')}
                counts.update(complete);done=sum(counts.values());elapsed=time.monotonic()-start
                speed=(done-initial)/elapsed if done>initial else None
                eta=(inv['total_frames']-done)/speed if speed else None
                write(dict(state,runner_pid=process.pid,completed_frames=done,completed_clips=len(complete),
                    current_clip=cid,current_clip_frames=int(n),current_clip_total=int(total),elapsed_s=elapsed,
                    observed_frames_per_second=speed,eta_seconds=eta,
                    estimated_finish_utc=(datetime.now(timezone.utc)+timedelta(seconds=eta)).isoformat() if eta else None))
        code=process.wait()
    if code:raise RuntimeError(f'Pose runner exited {code}; resumable checkpoints retained; see audit/gait_production_pose.log')
    complete=list(dest.glob('*/complete.json'));assert len(complete)==inv['total_jobs']
    actual=sum(json.loads(p.read_text())['frames'] for p in complete);assert actual==inv['total_frames']
    write(dict(state,phase='pose_complete',status='complete',completed_frames=actual,completed_clips=len(complete),elapsed_s=time.monotonic()-start))
    print('All native pose jobs complete',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        write(dict(phase='failed',status='failed',error=type(e).__name__,message=str(e)[:500]));raise

"""Wait read-only for existing inference, then execute the bounded training plan."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from motionbert_gait_common import ROOT,SOURCE,OUT,now,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'.pipeline.lock').open('a')
    try:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('An existing MotionBERT controller owns the lock; leaving it unchanged.',flush=True)
        raise SystemExit(1)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    required=['prepare_motionbert_gait_dataset.py','train_motionbert_gait.py','evaluate_motionbert_gait.py','report_motionbert_gait.py']
    for name in required:assert (ROOT/'scripts'/name).exists()
    source_state=json.loads((SOURCE/'production_status.json').read_text())
    if args.check_only:
        print(json.dumps(dict(status='preflight_only',source_phase=source_state['phase'],
            source_jobs=source_state.get('completed_clips'),would_wait_for_current_run=True,
            source_mutation_or_stop_commands=False)));return
    start_limit=datetime.fromisoformat('2026-11-15T05:10:00+00:00').timestamp()
    previous_jobs=None
    while True:
        state=json.loads((SOURCE/'production_status.json').read_text())
        post=json.loads((SOURCE/'postprocessing_status.json').read_text())
        if time.time()>start_limit:
            write(OUT/'pipeline_status.json',dict(status='deadline_window_insufficient',updated_at=now(),
                note='Existing pose run was not interrupted. No completed fine-tuning result is claimed.'))
            return
        if state.get('phase')=='pose_complete' and post.get('status')=='derived_outputs_complete':
            break
        write(OUT/'pipeline_status.json',dict(status='waiting_for_existing_pose_run',updated_at=now(),
            pid=os.getpid(),
            source_jobs=state.get('completed_clips'),source_total=state.get('total_jobs'),
            source_estimated_finish_utc=state.get('estimated_finish_utc'),
            source_status=state.get('status'),source_postprocessing=post.get('status'),
            note='Read-only waiting. Existing run is never paused, killed or restarted.'))
        if state.get('completed_clips')!=previous_jobs:
            print(json.dumps(dict(updated_at=now(),status='waiting',source_jobs=state.get('completed_clips'),
                estimated_finish_utc=state.get('estimated_finish_utc'))),flush=True)
            previous_jobs=state.get('completed_clips')
        time.sleep(30)
    audit=json.loads((SOURCE/'output_audit.json').read_text())
    assert audit['native_completed']==219 and not audit['failures']
    commands=[
        ['scripts/prepare_motionbert_gait_dataset.py'],
        ['scripts/train_motionbert_gait.py'],
        ['scripts/evaluate_motionbert_gait.py'],
        ['scripts/report_motionbert_gait.py']]
    for command in commands:
        write(OUT/'pipeline_status.json',dict(status='running',step=command[0],updated_at=now(),pid=os.getpid()))
        print(json.dumps(dict(step=command[0],status='starting',updated_at=now())),flush=True)
        subprocess.run([sys.executable,'-u',*command],cwd=ROOT,env=env,check=True)
        if command[0] in ('scripts/prepare_motionbert_gait_dataset.py','scripts/train_motionbert_gait.py'):
            subprocess.run([sys.executable,'scripts/report_motionbert_gait.py'],cwd=ROOT,env=env,check=True)
    write(OUT/'pipeline_status.json',dict(status='complete',updated_at=now(),
        report='http://127.0.0.1:8006/motionbert_gait/report.html',manual_report_review_required=True))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if (OUT/'dataset_manifest.json').exists():
            subprocess.run([sys.executable,'scripts/report_motionbert_gait.py'],cwd=ROOT,check=False)
        write(OUT/'pipeline_status.json',dict(status='failed',updated_at=now(),error=type(e).__name__,message=str(e)))
        raise

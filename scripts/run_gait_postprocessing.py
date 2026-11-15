"""Refresh derived outputs while the existing inference job runs; finalize once."""
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gait_metrics_20261114'


def write(state):
    state['updated_at']=datetime.now(timezone.utc).isoformat()
    p=OUT/'postprocessing_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)


def refresh():
    commands=[['scripts/extract_gait_metrics.py'],['scripts/extract_gait_metrics.py','--harmonized'],
              ['scripts/compare_gait_groups.py'],['scripts/audit_gait_outputs.py'],['scripts/plot_gait_comparisons.py']]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    for command in commands:
        write(dict(status='running',step=command[0],pid=os.getpid()))
        subprocess.run([sys.executable,'-u',*command],cwd=ROOT,env=env,check=True)


def main():
    lock=(OUT/'.postprocessing.lock').open('w')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Postprocessing is already running; existing worker and status were not changed.')
    previous=-1;last_refresh=0
    while True:
        state=json.loads((OUT/'production_status.json').read_text());done=state.get('completed_clips',0)
        finished=state.get('phase')=='pose_complete'
        if done!=previous and (time.monotonic()-last_refresh>180 or finished):
            refresh();previous=done;last_refresh=time.monotonic()
        if finished:
            report=json.loads((OUT/'comparison_results.json').read_text());audit=json.loads((OUT/'output_audit.json').read_text())
            assert report['status']=='final' and not audit['failures']
            write(dict(status='derived_outputs_complete',native_jobs=done,manual_final_review_required=True,
                       note='Goal completion still requires final visual QC and report verification.'))
            return
        if state.get('status')=='failed':raise RuntimeError('Inference failed; checkpoints retained. '+state.get('message',''))
        pid=state.get('runner_pid')
        if pid and not Path(f'/proc/{pid}').exists():raise RuntimeError('Inference process is no longer running; inspect production log.')
        write(dict(status='waiting_for_more_completed_poses',pid=os.getpid(),last_refreshed_native_jobs=previous,native_jobs_current=done))
        time.sleep(30)


if __name__=='__main__':
    try:main()
    except Exception as e:
        write(dict(status='failed',error=type(e).__name__,message=str(e)[:500]));raise

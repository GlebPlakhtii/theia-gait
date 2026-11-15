"""Run and audit the authorized full batch, recording terminal status for monitoring."""
import datetime,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/poses/sapiens2_5b_full'
STATUS=ROOT/'audit/pose_model_selection/full_job_status.json'

def save(data):
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(STATUS)

def main():
    state={'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':[]}
    save(state);t=time.monotonic()
    commands=[
        [sys.executable,'scripts/run_sapiens2_pose.py','--output',str(OUT)],
        [sys.executable,'scripts/audit_pose_batch.py',str(OUT)],
        [sys.executable,'scripts/plot_pose_review.py',str(OUT)],
    ]
    try:
        for command in commands:
            result=subprocess.run(command,cwd=ROOT)
            state['steps'].append({'command':command,'exit_code':result.returncode});save(state)
            if result.returncode:raise RuntimeError(f'Step failed with {result.returncode}: {command[1]}')
        state['status']='complete'
    except BaseException as e:
        state.update(status='failed',error=repr(e));raise
    finally:
        state.update(finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-t);save(state)

if __name__=='__main__':main()

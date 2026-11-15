"""Serialize local GPU benchmarks as the verified public checkpoints arrive."""
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gait_metrics_20261114/model_benchmark'


def status(phase,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'status.json';tmp=p.with_suffix('.tmp')
    data=dict(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**kw)
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(p);print(json.dumps(data),flush=True)


def main():
    remaining=['1b','0.4b'];deadline=time.monotonic()+3600
    status('waiting_for_checkpoint_download',remaining=remaining)
    while remaining:
        ready=[v for v in remaining if (ROOT/f'models/sapiens2-{v}/sapiens2_{v}_pose.safetensors').exists()]
        if not ready:
            if time.monotonic()>deadline:raise TimeoutError('Checkpoint download not completed within one hour')
            time.sleep(5);continue
        v=ready[0];status('benchmark_running',model=v)
        with (ROOT/f'audit/gait_benchmark_{v}.log').open('w') as log:
            subprocess.run([sys.executable,'-u','scripts/benchmark_sapiens_sizes.py','--variant',v],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        remaining.remove(v);status('benchmark_finished',model=v,remaining=remaining)
    status('complete')


if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=type(e).__name__,message=str(e)[:300]);raise

"""Download public source assets and preserve native clocks for manual shot review."""
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path
from prepare_videos import inspect

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'audit/new_video_intake_20261114'
RAW = ROOT / 'data/raw/new_exercise_20261114'
META = ROOT / 'data/metadata/new_exercise_20261114'


def process(row):
    target = RAW / (row['id'] + '.mp4')
    partial = target.with_suffix('.part')
    old_partial = RAW / (row['key'] + '.part')
    if old_partial.exists() and not partial.exists():
        old_partial.rename(partial)
    if not target.exists():
        subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                        '--connect-timeout', '25', '--max-time', '900',
                        '--continue-at', '-', '--output', str(partial), row['url']],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if row.get('bytes') and partial.stat().st_size != row['bytes']:
            raise ValueError('Download size mismatch: ' + row['id'])
        partial.replace(target)
    out = META / row['id']
    if (out / 'metadata.json').exists():
        metadata = json.loads((out / 'metadata.json').read_text())
    else:
        metadata = inspect(target, out)
    result = dict(row, local_path=str(target), metadata=str(out / 'metadata.json'),
                  sha256=metadata['sha256'], frames=metadata['decoded_frames'],
                  duration_s=metadata['last_timestamp_s']+metadata['median_frame_interval_s'])
    (BASE / (row['id'] + '_acquisition.json')).write_text(json.dumps(result, indent=2))
    sheet = BASE / (row['id'] + '_overview.jpg')
    if not sheet.exists():
        subprocess.run([str(ROOT / '.venv-pose/bin/python'),
                        'scripts/render_native_contact_sheet.py', str(target),
                        metadata['frame_map'], str(sheet), '--start', '0',
                        '--end', str(metadata['last_timestamp_s']),
                        '--step', str(max(2, round(result['duration_s']/24)))], cwd=ROOT, check=True)
    print(json.dumps({k:result[k] for k in ['id','frames','duration_s','sha256']}), flush=True)
    return result


if __name__ == '__main__':
    RAW.mkdir(parents=True, exist_ok=True)
    rows = json.loads((BASE / 'download_plan.json').read_text())
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {pool.submit(process, row):row for row in rows}
        for future in concurrent.futures.as_completed(tasks):
            try:
                outcomes.append(dict(status='acquired_and_clock_mapped', **future.result()))
            except Exception as error:
                outcomes.append(dict(id=tasks[future]['id'], status='failed', error=repr(error)))
                print(outcomes[-1], flush=True)
            (BASE / 'acquisition_status.json').write_text(json.dumps(outcomes, indent=2))
    sys.exit(any(r['status']=='failed' for r in outcomes))

"""Propose shot boundaries for human screening; never treat proposals as labels."""
import concurrent.futures
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'audit/new_video_intake_20261114'


def detect(path):
    record = json.loads(path.read_text())
    out = BASE / (record['id'] + '_scene_candidates.json')
    if out.exists():
        return
    logfile = BASE / (record['id'] + '_scene_scores.txt')
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                    '-threads', '2', '-i', record['local_path'], '-map', '0:v:0',
                    '-filter_threads', '1', '-vf',
                    "scale=320:-2,select='gt(scene,0.22)',metadata=print:file="+str(logfile),
                    '-an', '-f', 'null', '-'], check=True)
    text = logfile.read_text()
    candidates = [dict(source_pts_s=float(t), scene_score=float(s)) for t,s in re.findall(
        r'pts_time:([0-9.]+)\s+lavfi.scene_score=([0-9.]+)', text)]
    meta = json.loads(Path(record['metadata']).read_text())
    for candidate in candidates:
        candidate['timestamp_s'] = candidate['source_pts_s'] - meta['first_pts_s']
    out.write_text(json.dumps(dict(source_sha256=record['sha256'],
        method='ffmpeg scene score > 0.22 after 320px-width scale, every native frame',
        status='automatic_candidates_require_visual_check', candidates=candidates), indent=2))
    print(record['id'], len(candidates), flush=True)


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        list(pool.map(detect, BASE.glob('*_acquisition.json')))

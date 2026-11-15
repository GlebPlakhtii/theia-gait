"""Inspect native videos, preserve every decoded presentation timestamp, queue intervals.

No resampling, cropping, rotation or pose estimation is performed. Queue timestamps
are seconds relative to the first decoded presentation timestamp of the source.
"""
import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path
from common import sha256, write_csv, write_json


def inspect(video, output_dir):
    video = Path(video).resolve()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)]))
    stream = next(s for s in meta['streams'] if s['codec_type'] == 'video')
    raw = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'frame=best_effort_timestamp,best_effort_timestamp_time',
        '-of', 'json', str(video)]))['frames']
    times = [float(f['best_effort_timestamp_time']) for f in raw]
    if not times or any(not math.isfinite(t) for t in times):
        raise ValueError('Missing or nonfinite decoded timestamps')
    gaps = [b-a for a,b in zip(times, times[1:])]
    if any(g <= 0 for g in gaps):
        raise ValueError('Non-increasing PTS; manual clock review required')
    median = statistics.median(gaps) if gaps else None
    variable = any(abs(g-median) > max(1e-5, median*.001) for g in gaps) if gaps else None
    mapping = [{'source_frame_index': i, 'source_pts': f['best_effort_timestamp'],
                'source_pts_s': f['best_effort_timestamp_time'], 'timestamp_s': f'{times[i]-times[0]:.9f}'}
               for i, f in enumerate(raw)]
    write_csv(out/'frames.csv', list(mapping[0]), mapping)
    result = {'local_path': str(video), 'sha256': sha256(video), 'decoded_frames': len(raw),
              'first_pts_s': times[0], 'last_timestamp_s': times[-1]-times[0],
              'variable_fps': variable, 'median_frame_interval_s': median,
              'frame_map': str((out/'frames.csv').resolve()), 'ffprobe': meta}
    write_json(out/'metadata.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--clip-id', required=True)
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--end', type=float)
    a = p.parse_args()
    meta = inspect(a.video, a.output_dir)
    end = a.end if a.end is not None else meta['last_timestamp_s'] + (meta['median_frame_interval_s'] or 0)
    if not 0 <= a.start < end <= meta['last_timestamp_s'] + (meta['median_frame_interval_s'] or 0) + 1e-5:
        p.error('Interval outside native timeline')
    queue = {'clip_id': a.clip_id, 'local_path': meta['local_path'], 'sha256': meta['sha256'],
             'start_s': a.start, 'end_s': end, 'frame_map': meta['frame_map'],
             'status': 'awaiting_hlib', 'transformations': 'none; decode native frames; honor display rotation metadata'}
    write_csv(Path(a.output_dir)/'pose_queue.csv', list(queue), [queue])
    print(json.dumps({k:v for k,v in meta.items() if k != 'ffprobe'}, indent=2))


if __name__ == '__main__':
    main()

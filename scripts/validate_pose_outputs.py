"""Validate Hlib's long-form CSV against native frame maps and the pose queue."""
import argparse
import math
from common import read_csv

FIELDS = ('clip_id source_frame_index timestamp_s person_track_id joint_schema joint_id '
          'x y optional_z confidence coordinate_system model_name model_version preprocessing').split()
SYSTEMS = {'pixel', 'normalized_image', 'camera_metric_3d', 'world_metric_3d', 'monocular_relative_3d'}


def validate(rows, queue):
    errors = []
    if not rows:
        return ['No pose rows supplied']
    if set(FIELDS) - set(rows[0]):
        return ['Missing columns: ' + ', '.join(sorted(set(FIELDS)-set(rows[0])))]
    maps = {q['clip_id']: {int(f['source_frame_index']): float(f['timestamp_s'])
                         for f in read_csv(q['frame_map'])} for q in queue}
    qs = {q['clip_id']: q for q in queue}
    seen, last, metadata, frame_times = set(), {}, {}, {}
    for i,r in enumerate(rows, 2):
        try:
            if r['clip_id'] not in qs:
                raise ValueError('Unknown clip_id')
            for key in ('person_track_id','joint_schema','joint_id','model_name','model_version','preprocessing'):
                if not r[key].strip():
                    raise ValueError(f'Empty {key}')
            frame, t, conf = int(r['source_frame_index']), float(r['timestamp_s']), float(r['confidence'])
            if not math.isfinite(t) or not math.isfinite(conf) or not 0 <= conf <= 1:
                raise ValueError('Invalid timestamp/confidence')
            q, fmap = qs[r['clip_id']], maps[r['clip_id']]
            if frame not in fmap or abs(fmap[frame]-t) > 1e-5:
                raise ValueError('Frame/timestamp differs from native frame map')
            if not float(q['start_s']) <= t < float(q['end_s']):
                raise ValueError('Frame outside half-open queued interval')
            key = (r['clip_id'], r['person_track_id'], frame, r['joint_id'])
            if key in seen:
                raise ValueError('Duplicate joint row')
            seen.add(key)
            track = (r['clip_id'], r['person_track_id'])
            if frame < last.get(track, -1):
                raise ValueError('Rows not ordered by source frame within track')
            last[track] = frame
            m = tuple(r[k] for k in ('joint_schema','coordinate_system','model_name','model_version','preprocessing'))
            if track in metadata and metadata[track] != m:
                raise ValueError('Coordinate/model metadata changed within track')
            metadata[track] = m
            system = r['coordinate_system']
            if system not in SYSTEMS:
                raise ValueError('Unknown coordinate system')
            missing = r['x'] == '' and r['y'] == '' and r['optional_z'] == ''
            if missing:
                if conf != 0:
                    raise ValueError('Missing joint must have confidence 0')
                continue
            xy = [float(r[k]) for k in ('x','y')]
            if not all(math.isfinite(v) for v in xy):
                raise ValueError('Use empty fields for missing joints, not NaN/inf')
            if system == 'normalized_image' and not all(0 <= v <= 1 for v in xy):
                raise ValueError('Normalized coordinates outside [0,1]; use missing for out-of-frame joints')
            if r['optional_z'] and not math.isfinite(float(r['optional_z'])):
                raise ValueError('Nonfinite depth')
            if system.endswith('_3d') and not r['optional_z']:
                raise ValueError('3D coordinate system needs z')
        except (ValueError, KeyError) as e:
            errors.append(f'row {i}: {e}')
    return errors


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('poses')
    p.add_argument('--queue', default='data/manifests/pose_queue.csv')
    a = p.parse_args()
    issues = validate(read_csv(a.poses), read_csv(a.queue))
    for issue in issues[:50]:
        print(issue)
    print(f'{len(issues)} errors; schema validation does not establish pose accuracy or calibrated 3D')
    raise SystemExit(bool(issues))

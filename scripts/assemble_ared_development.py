"""Join already audited ARED predictions without changing parent clip identities."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'data/poses/ared_continuous_development'
    parents = [
        ('sapiens2_5b_extension_full', 'adenot_ared_dev_670_680'),
        ('sapiens2_5b_full', 'adenot_ared_dev_680_695'),
        ('sapiens2_5b_extension_full', 'adenot_ared_dev_695_779'),
    ]
    queues = {
        'sapiens2_5b_full': root / 'data/manifests/pose_queue_sapiens2_5b_20261113.csv',
        'sapiens2_5b_extension_full': root / 'audit/overnight_20261113/pose_extension_queue.csv',
    }
    if output.exists():
        raise ValueError('Refuse to overwrite assembled evidence')
    all_rows, records, source_hashes, models, schemas = [], [], set(), set(), set()
    queue_rows = []
    for batch, clip in parents:
        directory = root / 'data/poses' / batch
        selection = json.loads((directory / 'analysis_selection.json').read_text())
        if digest(queues[batch]) != selection['frozen_queue_sha256']:
            raise ValueError('Changed parent inference queue')
        with queues[batch].open(newline='') as stream:
            queue_rows.append(next(row for row in csv.DictReader(stream) if row['clip_id'] == clip))
        eligible = next(c for c in selection['clips'] if c['clip_id'] == clip)
        if not eligible['included_in_provisional_export']:
            raise ValueError(f'Held parent {clip}')
        if json.loads((directory / 'integrity_report.json').read_text())['errors']:
            raise ValueError(f'Parent batch failed integrity {batch}')
        run = json.loads((directory / 'run.json').read_text())
        models.add(json.dumps({k: run[k] for k in ('model', 'vendor_revision', 'weight_sha256', 'precision', 'resolution_hw', 'flip_test', 'coordinate_system')}, sort_keys=True))
        done = json.loads((directory / clip / 'complete.json').read_text())
        source_hashes.add(done['source_sha256'])
        path = directory / clip / 'poses.csv'
        if digest(path) != done['pose_csv_sha256']:
            raise ValueError(f'Parent checksum mismatch {clip}')
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            schemas.add(tuple(reader.fieldnames))
            rows = list(reader)
        if any(r['clip_id'] != clip for r in rows):
            raise ValueError('Parent clip ID mismatch')
        all_rows.extend(rows)
        records.append(dict(batch=batch, clip_id=clip, csv_sha256=digest(path),
                            completion_sha256=digest(directory / clip / 'complete.json'),
                            analysis_selection_sha256=digest(directory / 'analysis_selection.json'),
                            first_frame=done['source_frame_first'], last_frame=done['source_frame_last'],
                            frames=done['frames']))
    if len(models) != 1 or len(source_hashes) != 1 or len(schemas) != 1:
        raise ValueError('Incompatible parents')
    by_frame, keys = {}, set()
    for r in all_rows:
        frame = int(r['source_frame_index'])
        key = (frame, r['person_track_id'], r['joint_id'])
        if key in keys:
            raise ValueError('Duplicate native frame/joint')
        keys.add(key)
        by_frame.setdefault(frame, set()).add(r['joint_id'])
        if abs(float(r['timestamp_s']) - frame / 25) > 1e-8:
            raise ValueError('Unexpected source clock')
    if list(by_frame) != list(range(16750, 19475)):
        raise ValueError('Missing, extra or reordered native frames')
    if any(len(joints) != 23 for joints in by_frame.values()) or len({tuple(sorted(v)) for v in by_frame.values()}) != 1:
        raise ValueError('Incomplete or inconsistent joint schema')
    output.mkdir(parents=True)
    destination = output / 'poses.csv'
    with destination.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=next(iter(schemas)))
        writer.writeheader()
        writer.writerows(all_rows)
    queue_destination = output / 'pose_queue.csv'
    with queue_destination.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(queue_rows[0]))
        writer.writeheader()
        writer.writerows(queue_rows)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  role='Contiguous provisional development export; no new inference or independent participants',
                  frames=len(by_frame), rows=len(all_rows), start_s=670, end_s_exclusive=779,
                  duration_s=109, source_sha256=next(iter(source_hashes)), parents=records,
                  model=json.loads(next(iter(models))), csv_sha256=digest(destination),
                  combined_queue_sha256=digest(queue_destination),
                  assembly_script_sha256=digest(Path(__file__)),
                  clip_ids_preserved=True, smoothing=False, interpolation=False,
                  independent_decision_reference=None, split='development',
                  limitations=['Same astronaut and recording across all parents',
                               'Targeted visibility screening only; no anatomical accuracy reference',
                               'Original apparatus/self-occlusions retained; raw heatmap scores are not visibility',
                               'Fixed manually selected crops support offline feasibility'])
    (output / 'continuity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(output, len(by_frame), 'frames;', len(all_rows), 'rows')


if __name__ == '__main__':
    main()

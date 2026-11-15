"""Export a provisional pose subset, keeping held outputs in the audit batch.

Run only after full extraction and integrity checks. This does not certify
visibility, anatomical accuracy, clinical status or decision suitability.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--queue', type=Path, default=Path('data/manifests/pose_queue_sapiens2_5b_20261113.csv'))
    parser.add_argument('--eligibility', type=Path, default=Path('data/manifests/pose_analysis_eligibility.json'))
    args = parser.parse_args()
    audit = json.loads((args.directory / 'integrity_report.json').read_text())
    if audit['errors']:
        raise ValueError('Full-batch integrity errors must be resolved before export')
    with args.queue.open(newline='') as f:
        queue = list(csv.DictReader(f))
    expected = {q['clip_id'] for q in queue}
    if {c['clip_id'] for c in audit['clips']} != expected:
        raise ValueError('Integrity report does not cover the entire frozen queue')
    eligibility = json.loads(args.eligibility.read_text())
    if digest(args.queue) != eligibility['inference_queue_sha256']:
        raise ValueError('Eligibility was prepared against a different input queue')
    held = set(eligibility['clips'])
    if not held <= expected:
        raise ValueError('Unknown clip in hold list')
    accepted = expected - held
    destination = args.directory / 'poses_provisional.csv'
    temporary = destination.with_suffix('.csv.part')
    counts = dict.fromkeys(expected, 0)
    seen_frames = {cid: set() for cid in expected}
    with (args.directory / 'poses_all.csv').open(newline='') as src, temporary.open('w', newline='') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            cid = row['clip_id']
            counts[cid] += 1
            seen_frames[cid].add(int(row['source_frame_index']))
            if cid in accepted:
                writer.writerow(row)
    for clip in audit['clips']:
        cid = clip['clip_id']
        if counts[cid] != clip['csv_rows'] or len(seen_frames[cid]) != clip['frames']:
            raise ValueError(f'Combined export disagrees with integrity report: {cid}')
    for q in queue:
        with open(q['frame_map'], newline='') as f:
            native_frames = {int(r['source_frame_index']) for r in csv.DictReader(f)
                             if float(q['start_s']) <= float(r['timestamp_s']) < float(q['end_s'])}
        if seen_frames[q['clip_id']] != native_frames:
            raise ValueError(f'Export requires complete native coverage: {q["clip_id"]}')
    temporary.replace(destination)
    records = []
    for q in queue:
        cid = q['clip_id']
        records.append({
            'clip_id': cid,
            'analysis_status': eligibility['clips'].get(cid, {}).get('analysis_status', 'provisional_requires_visibility_review'),
            'included_in_provisional_export': cid in accepted,
            'frames': len(seen_frames[cid]),
            'csv_rows': counts[cid],
            'clinical_or_verified_healthy_arm': False,
            'details': eligibility['clips'].get(cid, {}).get('evidence', 'Targeted visual screening only; no full per-joint visibility or anatomical accuracy reference'),
        })
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Provisional review subset excluding known whole-clip holds; NOT a validated clean or clinical dataset',
        'all_clips': len(expected), 'all_frames': sum(map(len, seen_frames.values())),
        'held_clips': len(held), 'provisional_clips': len(accepted),
        'provisional_frames': sum(len(seen_frames[c]) for c in accepted),
        'provisional_csv_rows': sum(counts[c] for c in accepted),
        'frozen_queue_sha256': digest(args.queue),
        'eligibility_sha256': digest(args.eligibility),
        'all_csv_sha256': digest(args.directory / 'poses_all.csv'),
        'provisional_csv_sha256': digest(destination),
        'clips': records,
    }
    (args.directory / 'analysis_selection.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'clips'}, indent=2))


if __name__ == '__main__':
    main()

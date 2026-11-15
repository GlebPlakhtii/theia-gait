"""Register the reviewed Mendeley pilot without altering other source rows.

Run from the repository root after acquisition, native-frame preparation and
visual review. No pose inference or train/test assignment occurs here.
"""
from pathlib import Path
import csv
import json
from common import read_csv, sha256, write_csv


def main():
    root = Path.cwd()
    audit = root / 'audit/source_search_20261113'
    reviews = json.loads((audit / 'fitness_visual_review.json').read_text())
    validations = {r['path']: r for r in json.loads((audit / 'media_validation.json').read_text())}
    acquired = {}
    for name in ('fitness_acquisition_004.json', 'fitness_acquisition_007_009.json'):
        record = json.loads((audit / name).read_text())
        for f in record['selected_files']:
            acquired[f['local_path']] = (f, record['url'])
    manifest = root / 'data/manifests/videos.csv'
    queue_path = root / 'data/manifests/pose_queue.csv'
    with manifest.open(newline='') as f:
        fields = next(csv.reader(f))
    with queue_path.open(newline='') as f:
        qfields = next(csv.reader(f))
    rows = [r for r in read_csv(manifest) if r['dataset_id'] != 'mendeley_fitness_multiview']
    queue = [r for r in read_csv(queue_path) if not r['clip_id'].startswith('mendeley_subject_')]
    ready = []
    for review in reviews:
        path = Path(review['local_path'])
        meta = json.loads((root / 'data/metadata/fitness_multiview_squat' / path.stem / 'metadata.json').read_text())
        source, url = acquired[str(path)]
        digest = sha256(path)
        check = validations[str(path)]
        if not digest == meta['sha256'] == source['sha256'] == check['sha256']:
            raise ValueError(f'Hash mismatch: {path}')
        if check['decode_exit'] or check['decode_errors'] or not source['crc32_verified']:
            raise ValueError(f'Integrity failure: {path}')
        stream = next(s for s in meta['ffprobe']['streams'] if s['codec_type'] == 'video')
        frames = read_csv(meta['frame_map'])
        if len(frames) != meta['decoded_frames']:
            raise ValueError(f'Frame map length mismatch: {path}')
        duration = meta['ffprobe']['format']['duration']
        held = review['status'] != 'awaiting_hlib'
        row = dict.fromkeys(fields, 'unknown')
        row.update(dataset_id='mendeley_fitness_multiview', dataset_version='kgbb3yn47p.3',
            source_recording_id=path.stem, clip_id=review['clip_id'],
            participant_id='mendeley_subject_' + review['publisher_participant_id'],
            cohort='general_fitness_health_status_unknown', diagnosis_evidence='not_reported',
            domain='general_fitness', exercise=review['exercise_review'], equipment='bodyweight' if review['exercise_review'] != 'wrong_exercise_dumbbell_row' else 'dumbbells',
            view=review['publisher_view'], source_page='https://data.mendeley.com/datasets/kgbb3yn47p/3',
            download_url=url, access_checked_at='2026-11-13', access_status='byte_probe_verified',
            license_url='https://data.mendeley.com/datasets/kgbb3yn47p/3',
            permitted_use_notes='Release CC BY 4.0; cite DOI 10.17632/kgbb3yn47p.3 and contributors; preserve publisher anonymization',
            local_path=str(path), sha256=digest, width=stream['width'], height=stream['height'],
            duration_s=duration, fps_nominal=stream['r_frame_rate'], time_base=stream['time_base'],
            variable_fps=str(meta['variable_fps']).lower(), original_start_s=0, original_end_s=duration,
            transformation_history='Unmodified extracted publisher MP4; publisher face/body blur; no local crop/resampling/overlay',
            annotation_path=str(audit / 'fitness_visual_review.json'),
            annotation_type='filename good/bad labels unvalidated; sampled AI feasibility review; no event labels',
            annotator_provenance='Publisher filename labels; Codex sampled visual QC',
            prior_exposure='visual_development_2026-11-13', split='unassigned',
            exclusion_reason=review['notes'] if held else '',
            duplicate_group='mendeley_subject_' + review['publisher_participant_id'],
            duplicate_review_status='confirmed_cross_exercise_duplicate_excluded' if review['status']=='hold_wrong_exercise' else 'exact_checked_near_duplicate_pending',
            frame_map=meta['frame_map'])
        rows.append(row)
        q = dict(clip_id=review['clip_id'],local_path=str(path),sha256=digest,start_s=0,end_s=duration,
            frame_map=meta['frame_map'],status=review['status'],
            transformations=row['transformation_history'],
            purpose='Provisional lower-body squat pose feasibility; general-fitness only; no matched clinical arm' if not held else review['notes'])
        queue.append(q)
        if not held:
            ready.append(q)
    # All inputs are checked before any manifest is replaced.
    write_csv(manifest, fields, rows)
    write_csv(queue_path, qfields, queue)
    write_csv(root / 'data/manifests/pose_queue_squat_sources.csv', qfields, ready)
    write_csv(root / 'data/manifests/pose_queue_ready.csv', qfields, [q for q in queue if q['status']=='awaiting_hlib'])
    write_csv(audit / 'fitness_visual_review.csv', list(reviews[0]), reviews)
    print(f'{len(rows)} canonical videos; {len(ready)} ready source clips; {len(queue)} total queue rows including holds')


if __name__ == '__main__':
    main()

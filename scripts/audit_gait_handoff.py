"""Final data gate before training; does not certify the UI or finish the goal.

Run only after native inference and derived-output processing have finished.
This script never starts training, edits poses, or marks a goal complete.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gait_metrics_20261114'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def stable_sha(path):
    before = path.stat()
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, k) == getattr(after, k) for k in fields),
            f'File changed during hashing: {path}')
    return digest


def check_records(records, expected_ids, reviews, extraction_sha, native_ids):
    """Reject pending exports and stale QC, including non-failure review notes."""
    ids = [r['clip_id'] for r in records]
    require(len(ids) == len(set(ids)) and set(ids) == set(expected_ids),
            'Missing, extra or duplicate exported clips')
    for record in records:
        cid = record['clip_id']
        require(record['status'] == 'observations_extracted', f'Pending observations: {cid}')
        require(record.get('extraction_code_sha256') == extraction_sha,
                f'Stale extraction code: {cid}')
        review = reviews.get(cid)
        require(record.get('pose_visual_review') == review, f'Stale QC snapshot: {cid}')
        if cid in native_ids:
            require(review is not None, f'Missing sampled review: {cid}')
            require(review.get('status') in {'sampled_pose_track_passed', 'sampled_pose_track_with_limitations',
                                             'regional_pose_failure'}, f'Unfinished sampled review: {cid}')
            require(review.get('weight_sha256') == record['pose'].get('weight_sha256'),
                    f'Review belongs to another pose model: {cid}')
            frames = review.get('inspected_clip_frames', [])
            require(bool(frames) and all(type(f) is int and 0 <= f < record['clip_frames'] for f in frames),
                    f'Missing or invalid sampled review frames: {cid}')


def run_final_gate():
    production = read(OUT / 'production_status.json')
    post = read(OUT / 'postprocessing_status.json')
    require(production.get('phase') == 'pose_complete' and production.get('completed_clips') == 219,
            'Native pose processing is not complete')
    require(post.get('status') == 'derived_outputs_complete', 'Derived outputs are not complete')
    manifest_path = ROOT / 'data/prepared/gait_curated_20261114/manifest.json'
    manifest_sha = stable_sha(manifest_path)
    clips = read(manifest_path)['clips']
    require(len(clips) == 596 and len({c['clip_id'] for c in clips}) == 596, 'Corpus coverage changed')
    queue = list(csv.DictReader((OUT / 'production_queue.csv').open()))
    native_ids = {q['clip_id'] for q in queue}
    subset = set(read(OUT / 'harmonization_subset.json')['clips'])
    require(len(queue) == len(native_ids) == 219 and len(subset) == 13, 'Native queue changed')
    require(native_ids == {c['clip_id'] for c in clips if c['source'] != 'progait'} | subset,
            'Native queue does not cover the intended source/method branches')
    for filename, field in [('production_queue.csv', 'queue_sha256'), ('production_rois.json', 'rois_sha256')]:
        require(stable_sha(OUT / filename) == production[field], f'Production input changed: {filename}')
    for filename in ['comparison_manifest_frozen', 'extended_outcomes_frozen']:
        require(stable_sha(OUT / (filename + '.json')) == (OUT / (filename + '.sha256')).read_text().strip(),
                f'Frozen comparison definition changed: {filename}')
    protocol = read(OUT / 'processing_protocol.json')
    for filename, expected in protocol['files_sha256'].items():
        require(stable_sha(ROOT / filename) == expected, f'Unrecorded implementation change: {filename}')

    inputs = [manifest_path, OUT / 'processing_protocol.json', OUT / 'production_queue.csv',
              OUT / 'production_rois.json', OUT / 'comparison_manifest_frozen.json',
              OUT / 'extended_outcomes_frozen.json', OUT / 'harmonization_subset.json',
              OUT / 'additional_context_review.json', OUT / 'progait_subset_context_review.json',
              OUT / 'pose_visual_review.json'] + [ROOT / filename for filename in protocol['files_sha256']]
    input_hashes = {str(path.relative_to(ROOT)): stable_sha(path) for path in inputs}

    review_path = OUT / 'pose_visual_review.json'
    review_sha = stable_sha(review_path)
    reviews = read(review_path)['clips']
    require(set(reviews) == native_ids, 'Sampled review coverage differs from the native queue')
    extraction_sha = hashlib.sha256((ROOT / 'scripts/extract_gait_metrics.py').read_bytes()
                                   + (ROOT / 'scripts/gait_geometry.py').read_bytes()).hexdigest()
    contexts = {}
    for filename in ['additional_context_review.json', 'progait_subset_context_review.json']:
        contexts.update(read(OUT / filename)['clips'])
    import extract_gait_metrics as extractor
    records = []
    artifact_hashes = {}
    for base, selected, forced in [(OUT, clips, False),
                                   (OUT / 'harmonized', [c for c in clips if c['clip_id'] in subset], True)]:
        branch_records = [read(base / 'clips' / c['clip_id'] / 'metrics.json') for c in selected]
        branch_native = subset if forced else native_ids - subset
        check_records(branch_records, {c['clip_id'] for c in selected}, reviews, extraction_sha, branch_native)
        extractor.FORCE_SAPIENS = forced
        for clip, record in zip(selected, branch_records):
            c = dict(clip)
            cid = c['clip_id']
            if cid in reviews:
                c['pose_visual_review'] = reviews[cid]
            if cid in contexts:
                c['context_review'] = contexts[cid]
                c['contains_turns_or_transitions'] = contexts[cid]['contains_turns_or_transitions'] or contexts[cid]['scene_edit_visible']
                c['view'] = {'front_rear': 'front/rear'}.get(contexts[cid]['nominal_view'], contexts[cid]['nominal_view'])
            require(record['input_signature'] == extractor.input_signature(c, extraction_sha),
                    f'Stale extraction inputs: {cid}')
            path = base / 'clips' / cid / 'metrics.json'
            artifact_hashes[str(path.relative_to(ROOT))] = stable_sha(path)
        records.extend(branch_records)

    # Reuse the independent frame/mapping/17-family/region-exclusion audit.
    import audit_gait_outputs
    audit_gait_outputs.main()
    output = read(OUT / 'output_audit.json')
    require(output['native_completed'] == 219 and output['native_pending'] == 0 and not output['failures'],
            'Native or metric output audit is incomplete')
    # Fresh original-byte hashes; source alignment and full-decode evidence remain
    # valid only if the corresponding original AND curated bytes are unchanged.
    import verify_gait_source_integrity
    verify_gait_source_integrity.main()
    originals = read(OUT / 'source_integrity_audit.json')
    alignment = read(OUT / 'source_frame_alignment_audit.json')
    timestamps = read(OUT / 'video_timestamp_audit.json')
    for audit in (originals, alignment):
        require(audit['status'] == 'complete' and audit['curated_manifest_sha256'] == manifest_sha,
                'Source integrity/alignment evidence does not match the corpus')
    for audit in (alignment, timestamps):
        require(audit['status'] == 'complete' and set(audit['clips']) == {c['clip_id'] for c in clips}
                and all(x['status'] == 'passed' for x in audit['clips'].values()), 'Incomplete video audit')
    checked_files = {}
    for clip in clips:
        path = Path(clip['video_absolute'])
        actual = stable_sha(path)
        require(actual == clip['sha256'] == timestamps['clips'][clip['clip_id']]['sha256'],
                f'Curated video bytes changed: {clip["clip_id"]}')
        checked_files[str(path.relative_to(ROOT))] = actual
    for record in records:
        path = Path(record['pose']['path'])
        key = str(path.relative_to(ROOT))
        if key not in checked_files:
            checked_files[key] = stable_sha(path)
        require(checked_files[key] == record['pose']['sha256'], f'Pose bytes changed: {record["clip_id"]}')
    comparison = read(OUT / 'comparison_results.json')
    figures = read(OUT / 'figures/review_figures.json')
    require(comparison['status'] == figures['status'] == 'final', 'Comparison/figures are still interim')
    require(figures['comparison_updated_at'] == comparison['updated_at'], 'Stale comparison figures')
    require(figures['generator_sha256'] == stable_sha(ROOT / 'scripts/gait_review_figures.py'), 'Stale figure generator')
    require(stable_sha(manifest_path) == manifest_sha and stable_sha(review_path) == review_sha,
            'Corpus or QC changed during final gate')
    require(all(stable_sha(ROOT / filename) == digest for filename, digest in input_hashes.items()),
            'An input or implementation changed during final gate')
    return dict(status='passed_data_gate', updated_at=datetime.now(timezone.utc).isoformat(),
                curated_clips=596, native_jobs=219, harmonized_clips=13, sampled_pose_reviews=len(reviews),
                curated_manifest_sha256=manifest_sha, pose_review_sha256=review_sha,
                extraction_code_sha256=extraction_sha, output_audit_at=output['updated_at'],
                original_integrity_audit_at=originals['updated_at'], comparison_updated_at=comparison['updated_at'],
                input_sha256=input_hashes, metrics_sha256=artifact_hashes, video_and_pose_sha256=checked_files,
                final_ui_review_required=True, training_started=False, goal_complete=False,
                limitation='Data consistency and sampled-review coverage only. Final report/UI review, training input freeze, independent clinical validation and later goal phases are separate requirements.')


def main():
    # Do not race detached workers or a manual extraction, even while they idle.
    with ExitStack() as stack:
        for path in [OUT / '.production_controller.lock', OUT / 'poses_sapiens2_1b/.inference.lock',
                     OUT / '.postprocessing.lock', OUT / '.extract.lock', OUT / 'harmonized/.extract.lock']:
            handle = stack.enter_context(path.open('a'))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f'Not ready: an active worker holds {path.name}; existing work was not changed.')
        report = run_final_gate()
        path = OUT / 'data_handoff_audit.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(path)
        print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}))


if __name__ == '__main__':
    main()

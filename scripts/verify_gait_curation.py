"""Verify final curation inventory, hashes, annotation slices and sampled temporal alignment."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import subprocess
import cv2
import numpy as np
from curate_gait_inputs import ROOT, OUT, AUDIT, sha, load_reference, write


def main():
    cv2.setNumThreads(1)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    rows = manifest['clips']
    queue = json.loads((OUT / 'pose_input_queue.json').read_text())['clips']
    assert {r['clip_id'] for r in queue} == {r['clip_id'] for r in rows}
    expected = {OUT / r['video'] for r in rows}
    assert expected == set(OUT.glob('with_amputation/**/*.mp4')) | set(OUT.glob('without_amputation/**/*.mp4'))
    assert len(expected) == len(rows) == manifest['summary']['clips']
    assert len(json.loads((OUT / 'source_decisions.json').read_text())) == 445
    pg = [r for r in rows if r['source'] == 'progait']
    # All varied additional sources, plus a systematic sample spanning the sorted ProGait set.
    alignment_ids = {r['clip_id'] for r in rows if r['source'] in ['additional_input', 'lbmc', 'movi', 'mpi3dhp']}
    alignment_ids.update(pg[i]['clip_id'] for i in np.linspace(0, len(pg)-1, 32, dtype=int))

    def verify(row):
        path = OUT / row['video']
        assert sha(path) == row['sha256'], path
        assert row['clinical_form_label'] is None
        assert row['clip_frames'] == row['source_end_frame_exclusive'] - row['source_first_frame']
        info = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=width,height,avg_frame_rate,nb_frames', '-of', 'json', str(path)]))['streams'][0]
        assert (info['width'], info['height']) == (row['width'], row['height']), path
        assert int(info['nb_frames']) == row['clip_frames'], path
        assert abs(float(Fraction(info['avg_frame_rate'])) - row['fps']) < .01, path
        if row['source'] != 'ljmu360':
            assert row['decoded_frames'] == row['clip_frames']
            assert row['visual_review']['status'] == 'accept'
        if row['source'] == 'progait':
            pose = OUT / row['publisher_pose']
            assert sha(pose) == row['publisher_pose_sha256']
            with np.load(pose, allow_pickle=False) as z:
                start, end = row['source_first_frame'], row['source_end_frame_exclusive']
                assert np.array_equal(z['keypoints'], load_reference(row)[start:end])
                assert np.array_equal(z['source_frame'], np.arange(start, end))
                assert np.array_equal(z['clip_frame'], np.arange(end-start))
                assert z['joint_names'].shape == (23,)
        result = dict(clip_id=row['clip_id'], hash=True, dimensions=True, frame_count=True, cadence=True,
                      publisher_pose_exact_slice=row['source'] == 'progait')
        if row['clip_id'] in alignment_ids:
            source = cv2.VideoCapture(str(ROOT / row['original']))
            cut = cv2.VideoCapture(str(path))
            scores = []
            for frame in [0, row['clip_frames']//2, row['clip_frames']-1]:
                source.set(cv2.CAP_PROP_POS_FRAMES, row['source_first_frame'] + frame)
                cut.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok1, a = source.read(); ok2, b = cut.read()
                assert ok1 and ok2, (path, frame)
                scores.append(float(cv2.PSNR(a, b)))
            source.release(); cut.release()
            assert min(scores) > 28, (path, scores)
            result['source_alignment_psnr_db'] = scores
        return result

    checks = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(verify, row) for row in rows]
        for future in as_completed(futures):
            checks.append(future.result())
            if len(checks) % 50 == 0:
                print(f'Verified {len(checks)}/{len(rows)}', flush=True)
    result = dict(status='passed', clips=len(rows), hashes_verified=len(rows),
                  metadata_frame_counts_and_dimensions_verified=len(rows),
                  prior_full_decode_frame_count_checks=sum('decoded_frames' in r for r in rows),
                  publisher_pose_slices_verified=len(pg),
                  source_alignment_clips=len(alignment_ids),
                  source_alignment_frames=3*len(alignment_ids),
                  minimum_alignment_psnr_db=min(min(c['source_alignment_psnr_db']) for c in checks if 'source_alignment_psnr_db' in c),
                  note='Temporal alignment sampled independently at beginning/middle/end. Visual usability remains sampled, not certified every frame.',
                  checks=checks)
    write(AUDIT / 'verification.json', result)
    write(OUT / 'verification_summary.json', {k: v for k, v in result.items() if k != 'checks'})
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2), flush=True)


if __name__ == '__main__':
    main()

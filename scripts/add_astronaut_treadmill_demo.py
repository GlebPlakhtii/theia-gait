"""Add existing ISS running excerpts and new native-frame poses to the gait demo.

These are measurement-only examples. No terrestrial gait score is transferred.
"""
from pathlib import Path
import hashlib
import json
import shutil
import argparse
import subprocess
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'results/gait_selected_paper_20261115'
PREP = ROOT / 'data/prepared/exercise_intake_20261114/in_mission_treadmill'
POSE = ROOT / 'data/poses/sapiens2_1b_astronaut_treadmill_20261115'
AUDIT = ROOT / 'audit/astronaut_treadmill_demo_20261115'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['1b', '5b'], default='5b')
    parser.add_argument('--trim-stable', action='store_true', help='Retain the longest continuous interval with all displayed body joints supported; at least 2 seconds.')
    args = parser.parse_args()
    model_name = 'Sapiens2-' + args.model.upper()
    page = json.loads((WEB / 'page_data.json').read_text())
    if not (AUDIT / 'page_data_before.json').exists():
        shutil.copy2(WEB / 'page_data.json', AUDIT / 'page_data_before.json')
    page['clips'] = [c for c in page['clips'] if c['kind'] != 'astronaut']
    schema = json.loads((POSE / 'joint_schema.json').read_text())['body_feet_23']
    names = list(schema.values()); selected = list(map(int, schema))
    report = []
    for cid, label, agency, view in [
        ('GERST18_TREADMILL_01', 'Horizons mission · ISS treadmill running', 'ESA / NASA', 'side'),
        ('RUBIO_TREADMILL_01', 'Rubio recording · ISS treadmill running', 'NASA', 'oblique'),
    ]:
        origin = json.loads((PREP / cid / 'provenance.json').read_text())
        if cid == 'RUBIO_TREADMILL_01':
            # NASA's verified catalog ID includes a space in Frank Rubio.
            origin['source_page'] = 'https://images.nasa.gov/details/jsc2023m000211_Frank%20Rubio_Treadmill_Exercise_in_Space_Spanish_HighQuality_1080p_MP4'
        pose_dir = POSE
        refined = ROOT / 'data/poses/sapiens2_1b_astronaut_treadmill_refined_20261115'
        if (refined / cid / 'complete.json').exists():
            pose_dir = refined
        if args.model == '5b':
            pose_dir = ROOT / 'data/poses/sapiens2_5b_astronaut_treadmill_20261115'
        assert (pose_dir / cid / 'complete.json').exists(), f'Inference still incomplete: {cid}'
        raw = np.load(pose_dir / cid / 'raw_308.npz')
        xy = raw['keypoints'][:, selected].astype(float)
        scores = np.clip(raw['scores'][:, selected], 0, 1)
        video = PREP / (cid + '.mp4')
        cap = cv2.VideoCapture(str(video))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS); frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert len(xy) == frames == origin['frames']
        assert raw['source_frame_index'].tolist() == list(range(frames))
        q = (scores >= .5) & np.isfinite(xy).all(2) & (xy[:, :, 0] >= 0) & (xy[:, :, 0] < width) & (xy[:, :, 1] >= 0) & (xy[:, :, 1] < height)
        required = [names.index('nose')] + [names.index(s + '_' + j) for s in ['left', 'right'] for j in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']]
        supported = q[:, required].all(1)
        full_body_coverage = float(supported.mean())
        missing_before_trim = np.flatnonzero(~supported).tolist()
        start, end = 0, frames
        if args.trim_stable and not supported.all():
            ii = np.flatnonzero(supported)
            runs = np.split(ii, np.flatnonzero(np.diff(ii) > 1) + 1)
            longest = max(runs, key=len)
            assert len(longest) / fps >= 2, f'{cid}: no continuous two-second interval; inspect tracking before delivery'
            start, end = int(longest[0]), int(longest[-1] + 1)
        xy, scores, q = xy[start:end], scores[start:end], q[start:end]
        timestamps = raw['timestamp_s'][start:end] - raw['timestamp_s'][start]
        source_first = origin['original_first_frame'] + start
        frames = end - start
        curves = {}
        for side in ['left', 'right']:
            a, b, c = [names.index(side + '_' + j) for j in ['hip', 'knee', 'ankle']]
            u, v = xy[:, a] - xy[:, b], xy[:, c] - xy[:, b]
            lengths = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
            angle = 180 - np.degrees(np.arccos(np.clip(np.sum(u * v, axis=1) / np.maximum(lengths, 1e-9), -1, 1)))
            good = q[:, [a, b, c]].all(1) & (lengths > 4)
            curves[side + '_knee_bend'] = [round(float(x), 3) if valid else None for x, valid in zip(angle, good)]
        dest = WEB / 'astronauts' / cid; dest.mkdir(parents=True, exist_ok=True)
        if start or end != origin['frames']:
            # Prepared inputs are video-only; retain exact source frames and native frame rate.
            subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(video),
                '-vf', f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
                '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-fps_mode', 'passthrough',
                '-movflags', '+faststart', str(dest / 'video.mp4')], check=True)
            cap.release(); cap = cv2.VideoCapture(str(dest / 'video.mp4'))
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == frames
        else:
            shutil.copy2(video, dest / 'video.mp4')
        viewer = dict(joint_names=names, timestamp_s=timestamps.tolist(),
            clip_frame=list(range(frames)), native_timestamp_s=timestamps.tolist(),
            original_source_frame=list(range(source_first, source_first + frames)),
            xy_pixels=xy.round(3).tolist(), confidence=scores.round(4).tolist(), quality_mask=q.tolist(),
            observations=curves, model=model_name, display_only_sampling='All retained native frames; no temporal interpolation',
            visibility_note='Confidence and image-bounds gates only; occluded landmarks can still be wrong.')
        (dest / 'viewer.json').write_text(json.dumps(viewer, separators=(',', ':'), allow_nan=False))
        provenance = dict(origin, demo_trim_start_frame=start, demo_trim_end_frame_exclusive=end,
            demo_source_first_frame=source_first, demo_source_end_frame_exclusive=source_first + frames,
            demo_model=model_name, demo_body_joint_coverage=float(q[:, required].all(1).mean()),
            trim_basis='Longest continuous supported-body interval; no form outcome selection' if args.trim_stable else 'No trim')
        (dest / 'source_provenance.json').write_text(json.dumps(provenance, indent=2))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 2); ok, frame = cap.read(); cap.release(); assert ok
        cv2.imwrite(str(WEB / 'posters' / (cid + '.jpg')), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        entry = dict(id='astronaut:' + cid, clip_id=cid, kind='astronaut', curated=False,
            title=label, subject=agency, source=origin['source'], view=view,
            activity='running', environment='in_mission_treadmill', source_page=origin['source_page'],
            source_start_s=source_first / fps,
            source_end_s=(source_first + frames) / fps,
            transforms=origin['transforms'], pose_model=model_name,
            body_joint_coverage=float(q[:, required].all(1).mean()), trimmed=bool(start or end != origin['frames']),
            video_url=f'/gait_selected/astronauts/{cid}/video.mp4?v={sha(dest / "video.mp4")[:12]}',
            pose_url=f'/gait_selected/astronauts/{cid}/viewer.json', poster=f'posters/{cid}.jpg',
            video=dict(width=width, height=height, fps=fps, source_first_frame=source_first, frames=frames),
            windows=[], clinical_label=None, assessment='not_evaluated_domain_shift')
        page['clips'].append(entry)
        report.append(dict(clip_id=cid, frames=frames, duration_s=frames/fps,
            video_sha256=sha(dest / 'video.mp4'), pose_sha256=sha(dest / 'viewer.json'),
            joint_confidence_coverage=dict(zip(names, q.mean(0).round(4).tolist())),
            model=model_name, inference_run=str(pose_dir / 'run.json'),
            untrimmed_frames=origin['frames'], untrimmed_body_coverage=full_body_coverage,
            untrimmed_missing_body_frames=missing_before_trim,
            trim_start=start, trim_end_exclusive=end, retained_body_coverage=float(q[:, required].all(1).mean()),
            source_page=origin['source_page'], no_gait_classification=True))
    (WEB / 'page_data.json').write_text(json.dumps(page, separators=(',', ':'), allow_nan=False))
    (WEB / 'astronaut_treadmill_provenance.json').write_text(json.dumps(report, indent=2))
    (AUDIT / 'processing_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

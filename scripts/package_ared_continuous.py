"""Package audited ARED development poses into one native-rate QC replay."""
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / 'data/poses/ared_continuous_development'
    continuity = json.loads((directory / 'continuity.json').read_text())
    if digest(directory / 'poses.csv') != continuity['csv_sha256']:
        raise ValueError('Changed assembled poses')
    destination = directory / 'ared_670_779_pose_review.mp4'
    temporary = destination.with_suffix('.part.mp4')
    if destination.exists() or temporary.exists():
        raise ValueError('Refuse to overwrite replay evidence')
    records = []
    raw_min_scores = {'left': [], 'right': []}
    for parent in continuity['parents']:
        group = 'initial_provisional' if parent['batch'] == 'sapiens2_5b_full' else 'ared_extension'
        metadata_path = root / 'data/pose_replays' / group / (parent['clip_id'] + '_pose_review.json')
        metadata = json.loads(metadata_path.read_text())
        if digest(metadata['path']) != metadata['sha256']:
            raise ValueError('Changed parent replay')
        if (metadata['frames'], metadata['source_frame_first'], metadata['source_frame_last']) != (parent['frames'], parent['first_frame'], parent['last_frame']):
            raise ValueError('Replay and pose coverage differ')
        if metadata['source_sha256'] != continuity['source_sha256'] or Fraction(metadata['fps']) != 25:
            raise ValueError('Different source or frame cadence')
        if metadata['encoded']['width'] != 1536 or metadata['encoded']['height'] != 522:
            raise ValueError('Incompatible replay dimensions')
        batch = root / 'data/poses' / parent['batch']
        raw_path = batch / parent['clip_id'] / 'raw_308.npz'
        if digest(raw_path) != metadata['raw_pose_sha256']:
            raise ValueError('Changed parent raw predictions')
        schema = json.loads((batch / 'joint_schema.json').read_text())
        names = {name: int(index) for index, name in schema['body_feet_23'].items()}
        with np.load(raw_path) as raw:
            if list(raw['source_frame_index']) != list(range(parent['first_frame'], parent['last_frame'] + 1)):
                raise ValueError('Raw score frame coverage differs')
            for side in raw_min_scores:
                indices = [names[side + '_' + joint] for joint in ('hip', 'knee', 'ankle')]
                raw_min_scores[side].append(raw['scores'][:, indices].min(axis=1))
        records.append(metadata)
    # Paths are generated locally; quote for the FFmpeg concat-demuxer syntax.
    listing = directory / 'replay_parents.ffconcat'
    listing.write_text('ffconcat version 1.0\n' + ''.join("file '" + r['path'].replace("'", "'\\''") + "'\n" for r in records))
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
               '-i', str(listing), '-map', '0:v:0', '-an', '-c:v', 'copy', '-movflags', '+faststart', str(temporary)]
    subprocess.run(command, check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(temporary)]))
    times = np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']])
    expected = np.arange(continuity['frames']) / 25
    if len(times) != len(expected) or not np.allclose(times, expected, atol=1e-6, rtol=0):
        raise ValueError('Concatenation changed native frame count or presentation clock')
    decode = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
        '-i', str(temporary), '-map', '0:v:0', '-f', 'null', '-'], capture_output=True, text=True)
    if decode.returncode or decode.stderr.strip():
        raise ValueError('Strict decode failed: ' + decode.stderr)
    temporary.rename(destination)

    with (directory / 'poses.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    joints = {side + '_' + joint for side in ('left', 'right') for joint in ('hip', 'knee', 'ankle')}
    data = {joint: [] for joint in joints}
    for row in rows:
        if row['joint_id'] in data:
            data[row['joint_id']].append([float(row['timestamp_s']), float(row['x']) if row['x'] else np.nan,
                float(row['y']) if row['y'] else np.nan, float(row['confidence'])])
    data = {joint: np.asarray(values) for joint, values in data.items()}
    native_times = 670 + expected
    if any(len(v) != len(expected) or not np.allclose(v[:, 0], native_times, atol=1e-8, rtol=0) for v in data.values()):
        raise ValueError('Signal clock mismatch')
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True, layout='constrained')
    for side, color in [('left', '#16834b'), ('right', '#d66a16')]:
        hip, knee, ankle = (data[side + '_' + joint] for joint in ('hip', 'knee', 'ankle'))
        u, v = hip[:, 1:3] - knee[:, 1:3], ankle[:, 1:3] - knee[:, 1:3]
        denominator = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        cosine = np.divide((u * v).sum(1), denominator, out=np.full(len(u), np.nan), where=denominator > 1e-8)
        angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        scores = np.minimum.reduce([hip[:, 3], knee[:, 3], ankle[:, 3]])
        angles[scores < .3] = np.nan
        hip_y = hip[:, 2].copy()
        hip_y[hip[:, 3] < .3] = np.nan
        axes[0].plot(native_times, angles, color=color, lw=.8, label=side)
        axes[1].plot(native_times, hip_y, color=color, lw=.8)
        axes[2].plot(native_times, np.concatenate(raw_min_scores[side]), color=color, lw=.8)
    axes[0].set(ylabel='Projected knee angle (deg)', ylim=(0, 185))
    axes[0].legend(loc='lower left', ncol=2)
    axes[1].set(ylabel='Hip image y (pixels)')
    axes[1].invert_yaxis()
    axes[2].set(ylabel='Minimum raw joint score', xlabel='Native source time (seconds)')
    axes[2].axhline(.3, color='gray', linestyle=':', lw=.8)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.set_xlim(670, 779)
        for boundary in (680, 695):
            axis.axvline(boundary, color='#526879', linestyle='--', lw=.8)
    fig.suptitle('ARED development recording — 2,725 native frames, no smoothing\nDashed lines mark extraction boundaries; geometry and scores are not clinical measurements', fontsize=11)
    fig.savefig(directory / 'signals.png', dpi=160)
    plt.close(fig)
    original = root / 'data/raw/esa_adenot_20260809_ared.mp4'
    original_link = quote(os.path.relpath(original, directory)) + '#t=670.044,779.044'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Theia — continuous ARED pose review</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#f5f8fa;color:#152b35}p{line-height:1.5}video,img{width:100%;height:auto;border-radius:6px}section{background:white;padding:22px;border:1px solid #d8e2e7;border-radius:10px;margin:20px 0}a{color:#076273}button{font:inherit;padding:9px 14px;margin:4px;border:1px solid #afc4cc;border-radius:6px;background:#eff7f9;cursor:pointer}output{font-variant-numeric:tabular-nums}small{color:#4d6470}</style>
<h1>ARED — continuous pose review</h1>
<p>109 seconds · 2,725 frames at native 25 fps · one astronaut and recording · development only</p>
<p>Includes squat sequences, transitions and a near-stationary posture around source time 720–740 s. Task boundaries and prescribed exercise phases have not been independently annotated.</p>
<p>Original RGB appears on the left, Sapiens2-5B pose on the right. This silent derived replay preserves every selected frame. Apparatus and self-occlusion remain; targeted review does not establish anatomical accuracy. No clinical transfer comparison or technique/load recommendation has been evaluated.</p>
<section><video id="replay" controls preload="metadata" src="ared_670_779_pose_review.mp4"></video>
<p><output id="clock">Native source time: 670.000 s</output></p>
<button data-native="670">Start · 670 s</button><button data-native="680">Boundary · 680 s</button><button data-native="695">Boundary · 695 s</button><button data-native="730">730 s</button><button data-native="770">770 s</button>
<p><small>Replay time 0 corresponds to native frame 16,750 / source time 670.000 s. Browser seeking is approximate; exported frame IDs and timestamps are authoritative.</small></p></section>
<section><h2>Continuous measurements</h2><img src="signals.png" alt="Left and right projected knee angles, hip image height, and raw joint scores across the selected ARED interval">
<p>These are camera-dependent 2D predictions. Scores are not visibility probabilities. The 0.3 display cutoff hides low-score geometry; it does not validate retained points. No interpolation or temporal smoothing was applied.</p></section>
<section><h2>Files and provenance</h2><p><a href="poses.csv">Body/feet CSV</a> · <a href="pose_queue.csv">Matching input queue</a> · <a href="continuity.json">Parent pose lineage</a> · <a href="replay.json">Replay checks</a> · <a href="ORIGINAL_LINK">Unmodified source recording</a></p>
<p>Three extraction intervals have been joined with their original clip IDs preserved. They do not represent three participants. Full 308-landmark arrays remain in the parent batch folders.</p></section>
<script>const video=document.getElementById('replay');const clock=document.getElementById('clock');document.querySelectorAll('[data-native]').forEach(button=>button.addEventListener('click',()=>{video.currentTime=Number(button.dataset.native)-670;}));video.addEventListener('timeupdate',()=>{clock.textContent='Native source time: '+(670+video.currentTime).toFixed(3)+' s (browser estimate)';});</script></html>'''.replace('ORIGINAL_LINK', html.escape(original_link, quote=True))
    (directory / 'index.html').write_text(page)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), role='Derived development QC only',
        path=str(destination), sha256=digest(destination), frames=len(times), duration_s=109,
        native_source_start_s=670, native_source_end_s_exclusive=779, frame_rate='25/1',
        source_sha256=continuity['source_sha256'], parent_replays=records,
        command=command, strict_decode_exit_code=decode.returncode,
        max_clock_residual_s=float(np.max(np.abs(times - expected))),
        continuity_sha256=digest(directory / 'continuity.json'), script_sha256=digest(__file__),
        plot_sha256=digest(directory / 'signals.png'), page_sha256=digest(directory / 'index.html'),
        reencoding=False, source_audio_included=False, pose_modified=False,
        signal_geometry='Exported pixel coordinates; missing or low-confidence geometry hidden',
        signal_scores='Unclipped raw hip/knee/ankle heatmap-score minimum from parent308 arrays')
    (directory / 'replay.json').write_text(json.dumps(report, indent=2) + '\n')
    print(directory / 'index.html', len(times), 'frames; strict decode and presentation clock passed')


if __name__ == '__main__':
    main()

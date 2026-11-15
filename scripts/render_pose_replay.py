"""Render silent side-by-side RGB/pose review, preserving native-frame lineage.

This is a derived QC video. Original RGB, poses and native clocks are unchanged.
Only contiguous constant-rate selections are accepted; no temporal interpolation.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--queue', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--clip', action='append')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    schema = json.loads((args.directory / 'joint_schema.json').read_text())
    names = {name: int(index) for index, name in schema['body_feet_23'].items()}
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 15)
    records = []
    for q in csv.DictReader(args.queue.open()):
        cid = q['clip_id']
        if args.clip and cid not in args.clip:
            continue
        clip = args.directory / cid
        complete = json.loads((clip / 'complete.json').read_text())
        if complete['source_sha256'] != q['sha256'] or digest(q['local_path']) != q['sha256']:
            raise ValueError('Changed source')
        raw_path = clip / 'raw_308.npz'
        raw = np.load(raw_path)
        frames, times = raw['source_frame_index'], raw['timestamp_s']
        points, scores = raw['keypoints'], raw['scores']
        with Path(q['frame_map']).open() as f:
            fmap = {int(r['source_frame_index']): r for r in csv.DictReader(f)
                    if float(q['start_s']) <= float(r['timestamp_s']) < float(q['end_s'])}
        if list(frames) != list(fmap) or not np.all(np.diff(frames) == 1):
            raise ValueError('Replay requires full contiguous native-frame coverage')
        steps = np.diff(times)
        step = float(np.median(steps))
        if step <= 0 or np.max(np.abs(steps - step)) > 1e-5:
            raise ValueError('Variable-rate clip needs a timestamp-preserving encoder')
        ticks = [int(row['source_pts']) for row in fmap.values()]
        tick_steps = np.diff(ticks)
        if len(tick_steps) == 0 or not np.all(tick_steps == tick_steps[0]) or tick_steps[0] <= 0:
            raise ValueError('Variable native PTS cadence needs a timestamp-preserving encoder')
        with av.open(q['local_path']) as clock_container:
            time_base = Fraction(clock_container.streams.video[0].time_base)
        fps = Fraction(1, 1) / (int(tick_steps[0]) * time_base)
        if not np.allclose(times, times[0] + np.arange(len(times)) / float(fps), atol=1e-5, rtol=0):
            raise ValueError('Derived frame clock would drift')
        dest = args.output / f'{cid}_pose_review.mp4'
        part = dest.with_name(dest.stem + '.part.mp4')
        if dest.exists() or part.exists():
            raise ValueError(f'Refuse overwrite: {dest}')
        process = None
        count = 0
        width = height = None
        try:
            with av.open(q['local_path']) as container:
                for fi, frame in enumerate(container.decode(video=0)):
                    if fi > int(frames[-1]): break
                    if fi not in fmap: continue
                    i = count
                    row = fmap[fi]
                    if fi != int(frames[i]) or abs(float(frame.pts * frame.time_base) - float(row['source_pts_s'])) > 1e-5:
                        raise ValueError('Source frame/timestamp mismatch')
                    original = frame.to_image().convert('RGB')
                    src_w, src_h = original.size
                    scale = min(1, 720 / src_h)
                    view_w, view_h = int(src_w * scale) // 2 * 2, int(src_h * scale) // 2 * 2
                    view = original.resize((view_w, view_h), Image.Resampling.LANCZOS)
                    width, height = 2 * view_w, view_h + 90
                    canvas = Image.new('RGB', (width, height), '#11232b')
                    canvas.paste(view, (0, 28)); canvas.paste(view, (view_w, 28))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((8, 5), 'Original RGB', fill='white', font=font)
                    draw.text((view_w + 8, 5), 'Predicted pose', fill='white', font=font)
                    def xy(index):
                        x, y = points[i, index]
                        return (float(x) * view_w / src_w + view_w, float(y) * view_h / src_h + 28)
                    def shown(index):
                        x, y = points[i, index]
                        return scores[i, index] >= .3 and np.isfinite([x, y]).all() and 0 <= x < src_w and 0 <= y < src_h
                    for side, color in [('left', '#62ed93'), ('right', '#ffa550')]:
                        for j1, j2 in [('shoulder','elbow'),('elbow','wrist'),('shoulder','hip'),('hip','knee'),('knee','ankle'),('ankle','big_toe'),('ankle','heel')]:
                            n1, n2 = names[f'{side}_{j1}'], names[f'{side}_{j2}']
                            if shown(n1) and shown(n2): draw.line([xy(n1), xy(n2)], fill=color, width=2)
                    for name, index in names.items():
                        if shown(index):
                            x, y = xy(index)
                            draw.ellipse((x-2,y-2,x+2,y+2),fill='#ffed63')
                    label = f'Native frame {fi} | source time {times[i]:.3f}s | Sapiens2-5B'
                    draw.text((8, view_h+35), label, fill='white', font=font)
                    draw.text((8, view_h+58), 'QC only: occlusion and anatomical accuracy unverified.', fill='#ffcf8a', font=font)
                    if process is None:
                        process = subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error',
                            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
                            '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','20',
                            '-preset','medium','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(part)],stdin=subprocess.PIPE)
                    process.stdin.write(canvas.tobytes())
                    count += 1
            if process is None: raise ValueError('No frames rendered')
            process.stdin.close()
            if process.wait() != 0: raise RuntimeError('FFmpeg encoding failed')
            if count != len(frames): raise ValueError('Incomplete replay')
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate(); process.wait()
            raise
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,avg_frame_rate,duration,width,height','-of','json',str(part)]))['streams'][0]
        if int(probe['nb_read_frames']) != count or Fraction(probe['avg_frame_rate']) != fps:
            raise ValueError('Encoded frame count/rate mismatch')
        part.rename(dest)
        record = dict(clip_id=cid, path=str(dest.resolve()), sha256=digest(dest), frames=count,
            renderer_sha256=digest(__file__),
            fps=str(fps), source_frame_first=int(frames[0]), source_frame_last=int(frames[-1]),
            source_timestamp_first=float(times[0]), source_timestamp_last=float(times[-1]),
            source_sha256=q['sha256'], raw_pose_sha256=digest(raw_path), native_frame_map_sha256=digest(q['frame_map']),
            role='Silent derived QC replay only; not source footage or independent decision reference',
            display='Original and estimated pose side by side; aspect preserved; score 0.3 is display threshold only',
            encoded=probe)
        dest.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        records.append(record)
        print(cid, count, 'frames rendered and count/rate verified', flush=True)
    (args.output/'index.json').write_text(json.dumps(dict(renderer_sha256=digest(__file__),
        queue_sha256=digest(args.queue), clips=records),indent=2)+'\n')


if __name__ == '__main__': main()

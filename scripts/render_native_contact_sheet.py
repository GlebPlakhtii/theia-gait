"""Sample original decoded frames against a native frame map for visual review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('frame_map', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--start', type=float, required=True)
    p.add_argument('--end', type=float, required=True)
    p.add_argument('--step', type=float, default=2)
    a = p.parse_args()
    if a.end < a.start or a.step <= 0:
        raise ValueError('Require end >= start and step > 0')
    if a.output.exists() or a.output.with_suffix('.json').exists():
        raise ValueError('Refuse to overwrite review evidence')
    with a.frame_map.open(newline='') as f:
        frames = list(csv.DictReader(f))
    if a.start < 0 or a.end > float(frames[-1]['timestamp_s']) + .001:
        raise ValueError('Requested range extends beyond native map')
    selected = {}
    for k in range(int((a.end - a.start) / a.step + 1e-8) + 1):
        requested = a.start + k * a.step
        row = min(frames, key=lambda r: abs(float(r['timestamp_s']) - requested))
        selected[int(row['source_frame_index'])] = dict(row, requested_s=requested)
    images = {}
    with av.open(str(a.video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i > max(selected):
                break
            if i not in selected:
                continue
            if abs(float(frame.pts * frame.time_base) - float(selected[i]['source_pts_s'])) > 1e-5:
                raise ValueError(f'Native timestamp mismatch at frame {i}')
            im = frame.to_image().convert('RGB')
            im.thumbnail((352, 260))
            images[i] = im
    if set(images) != set(selected):
        raise ValueError('Missing requested decoded frames')
    height = max(im.height for im in images.values()) + 42
    out = Image.new('RGB', (4 * 370, ((len(images) + 3) // 4) * height), 'white')
    draw = ImageDraw.Draw(out)
    for k, (i, im) in enumerate(images.items()):
        x, y = (k % 4) * 370, (k // 4) * height
        out.paste(im, (x + (370 - im.width) // 2, y + 38))
        draw.text((x + 8, y + 5), f"native t={float(selected[i]['timestamp_s']):.3f}s; frame={i}", fill='black')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(a.output, quality=94)
    def digest(path):
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    record = dict(source=str(a.video.resolve()), source_sha256=digest(a.video),
                  frame_map=str(a.frame_map.resolve()), frame_map_sha256=digest(a.frame_map),
                  renderer_sha256=digest(Path(__file__)), frames=list(selected.values()),
                  role='Sampled original RGB for review; not inferred pose or independent labels')
    a.output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(a.output, len(images), 'native frames')


if __name__ == '__main__':
    main()

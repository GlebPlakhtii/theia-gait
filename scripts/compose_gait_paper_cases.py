"""Crop real browser captures and compose a publication preview; no synthetic pixels."""
from pathlib import Path
import json
import hashlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'deliverables/Theia_Selected_Gait_Paper_2026-11-15'
capture_path = PAPER / 'evidence/paired_case_capture.json'
captures = json.loads(capture_path.read_text())
panels = []
for key, name in [('A', 'case_amputation'), ('B', 'case_reference')]:
    path = PAPER / 'images' / f'{name}_full.jpg'
    original = Image.open(path).convert('RGB')
    meta = captures[key]
    rect = meta['rects'].get('.capture-panel', meta['rects']['.review-grid'])
    vw, vh = meta['viewport']
    sx, sy = original.width / vw, original.height / vh
    bounds = tuple(round(v) for v in (
        rect['x'] * sx, rect['y'] * sy,
        (rect['x'] + rect['width']) * sx,
        (rect['y'] + rect['height']) * sy,
    ))
    panel = original.crop(bounds)
    panel.save(PAPER / 'images' / f'{name}.jpg', quality=97)
    meta['publication_crop_pixels'] = bounds
    meta['source_image_dimensions'] = list(original.size)
    meta['full_image_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    panels.append(panel)
capture_path.write_text(json.dumps(captures, indent=2))

# Caption labels surround the unaltered screenshot crops.
width = panels[0].width + 40
height = sum(p.height for p in panels) + 155
out = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(out)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
y = 15
labels = [
    '(a) Prosthetic gait: left-knee and left-arm flags',
    '(b) Non-amputee walking reference: no regional flag',
]
for panel, label in zip(panels, labels):
    draw.text((20, y), label, fill='#1d1f23', font=font)
    y += 34
    out.paste(panel, (20, y))
    y += panel.height + 22
draw.text((20, y), 'Same density-model fold and calibration; threshold at the 95th percentile.', fill='#6b6e75', font=small)
draw.text((20, y + 20), 'Illustrative cases, without independent clinical form labels.', fill='#6b6e75', font=small)
out.save(PAPER / 'images/paired_cases.png')
print('Cropped two original screenshots; preserved full captures and metadata.')

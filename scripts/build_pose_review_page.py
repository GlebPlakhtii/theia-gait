"""Build the final local review page from audited outputs and eligibility records."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
from PIL import Image, ImageDraw


def esc(value):
    return html.escape(str(value), quote=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--queue', default='data/manifests/pose_queue_sapiens2_5b_20261113.csv')
    args = p.parse_args()
    selection = json.loads((args.directory / 'analysis_selection.json').read_text())
    clips = {c['clip_id']: c for c in selection['clips']}
    sections, navigation = [], []
    for q in csv.DictReader(open(args.queue)):
        cid = q['clip_id']
        record = clips[cid]
        d = args.directory / cid
        # Preserve aspect ratio and remove unused landscape cell height.
        thumbs = []
        for path in sorted(d.glob('preview_*.jpg')):
            im = Image.open(path).convert('RGB')
            im.thumbnail((360, 420))
            thumbs.append((path.stem, im))
        if thumbs:
            height = max(im.height for _, im in thumbs) + 35
            contact = Image.new('RGB', (1140, height * ((len(thumbs) + 2) // 3)), 'white')
            draw = ImageDraw.Draw(contact)
            for i, (title, im) in enumerate(thumbs):
                x, y = (i % 3) * 380, (i // 3) * height
                contact.paste(im, (x + (380 - im.width) // 2, y + 25))
                draw.text((x + 10, y + 5), title, fill='black')
            contact.save(d / 'contact.jpg', quality=94)
        held = not record['included_in_provisional_export']
        status = 'HELD — excluded from provisional export' if held else 'PROVISIONAL — visibility review still required'
        src = quote(os.path.relpath(q['local_path'], args.directory)) + f"#t={q['start_s']},{q['end_s']}"
        navigation.append(f'<li><a href="#{esc(cid)}">{esc(cid)}</a> — {record["frames"]} frames; {"held" if held else "provisional"}</li>')
        sections.append(f'''<section id="{esc(cid)}">
<h2>{esc(cid)}</h2><p class="{'held' if held else 'provisional'}"><strong>{status}</strong></p>
<p>{esc(record['details'])}</p>
<p>{record['frames']} native frames · {record['csv_rows']} body/feet rows</p>
<p><a href="{src}">Original RGB</a> · <a href="{cid}/poses.csv">Body/feet CSV</a> · <a href="{cid}/raw_308.npz">All 308 raw landmarks</a></p>
<details><summary>Play original recording</summary><video controls preload="none" src="{src}"></video><p>Browser seeking is approximate. Source frame indices and the native frame map are authoritative.</p></details>
<p>Derived overlays for review; the original source is unchanged:</p>
<a href="{cid}/contact.jpg"><img loading="lazy" src="{cid}/contact.jpg" alt="Sampled pose overlays"></a>
<a href="{cid}/signals.png"><img loading="lazy" src="{cid}/signals.png" alt="Projected knee-angle and raw-score timelines"></a>
<details><summary>Inspect the three largest adjacent knee-angle steps</summary><p>Review candidates only; a large step is not automatically an error or a clinical event.</p><a href="{cid}/temporal_jump_review.jpg"><img loading="lazy" src="{cid}/temporal_jump_review.jpg" alt="Before-and-after pose comparisons"></a></details>
</section>''')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Theia pose review</title>
<style>body{{font:16px system-ui;max-width:1120px;margin:36px auto;padding:0 20px;color:#16252b;background:#f7f9fa}}section{{background:white;padding:24px;margin:24px 0;border:1px solid #dce4e7;border-radius:8px}}h2{{font-size:18px;overflow-wrap:anywhere}}img{{max-width:100%;height:auto}}video{{max-width:100%;max-height:560px}}a{{color:#076273}}p,li{{line-height:1.5}}.held{{background:#fff0e8;color:#7b2900;padding:12px}}.provisional{{background:#edf3f5;padding:12px}}summary{{cursor:pointer}}</style>
<h1>Theia — completed pose extraction</h1>
<p>Sapiens2-5B · 1024×768 model input · BF16 · horizontal flip test · original-pixel coordinates · no temporal smoothing or interpolation.</p>
<p><strong>{selection['all_clips']} clips / {selection['all_frames']:,} native frames extracted.</strong> {selection['held_clips']} clips are held for observed body masking. The provisional review subset contains {selection['provisional_clips']} clips / {selection['provisional_frames']:,} frames.</p>
<p><a href="poses_provisional.csv">Provisional body/feet CSV</a> · <a href="analysis_selection.json">Inclusion and hold details</a> · <a href="poses_all.csv">All predictions, including held clips (audit only)</a> · <a href="integrity_report.json">Integrity checks</a> · <a href="run.json">Run provenance</a></p>
<p><strong>No clinical transfer or decision benefit is established.</strong> Gym participants have undocumented health status; ARED is development footage. Provisional inclusion does not certify clean visibility or anatomical accuracy. Scores are not visibility probabilities; the plotted 0.3 cutoff is only a display rule. Projected angles are not anatomical flexion, load, pain or fatigue. Manually fixed crops are for offline feasibility.</p>
<ul>{''.join(navigation)}</ul>{''.join(sections)}</html>'''
    (args.directory / 'index.html').write_text(page)
    print(args.directory / 'index.html')


if __name__ == '__main__':
    main()

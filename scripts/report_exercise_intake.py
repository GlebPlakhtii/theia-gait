"""Create a local, read-only intake report with prepared RGB and audit evidence."""
import html
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/prepared/exercise_intake_20261114'
AUDIT=ROOT/'audit/new_video_intake_20261114'


def main():
    manifest=json.load(open(ROOT/'data/manifests/exercise_intake_20261114.json'))
    selections=json.load(open(OUT/'selections.json'))
    review=json.load(open(AUDIT/'pilot_review.json'))
    status=json.load(open(AUDIT/'pipeline_status.json')) if (AUDIT/'pipeline_status.json').exists() else {}
    approved=set(review['approved_full_inference']);esc=html.escape
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Theia — additional exercise video intake</title><style>
body{font:16px system-ui;background:#f5f7f9;color:#142532;max-width:1120px;margin:32px auto;padding:0 20px;line-height:1.5}section{background:white;border:1px solid #d2dce3;border-radius:10px;padding:22px;margin:20px 0}h1{font-size:29px}h2{font-size:21px}video{max-width:100%;max-height:510px;background:#101820}table{border-collapse:collapse;width:100%}td,th{text-align:left;vertical-align:top;border-bottom:1px solid #d2dce3;padding:10px}a{color:#075e86}.tag{display:inline-block;background:#e4edf3;padding:4px 10px;border-radius:6px}.small{font-size:14px}code{overflow-wrap:anywhere}</style>
<h1>Additional exercise footage — screening and preparation</h1>
<p>10 official recordings acquired: all six supplied sources plus four additional recordings. Source duration: 24 min 10 s; source files: 1.41 GB. These are assembled media sources, not an existing annotated clinical or NASA benchmark.</p>
<p>9 prepared RGB segments. Five segments tested with Sapiens2-5B (60 sampled frames); four approved for a full native-frame run, totaling 2,221 frames. The low-resolution Pesquet 2017 inset is held after incorrect far-leg predictions.</p>
<p>Ground training, in-mission ARED, treadmill and limited-visibility material remain separate. No clinical rehabilitation recordings or independent form labels were added.</p>''']
    parts.append('<section><h2>Processing status</h2><p>Report generated '+esc(datetime.now(timezone.utc).isoformat())+'</p><p>Pipeline phase at generation: <strong>'+esc(status.get('phase','not started'))+'</strong>.</p><p><a href="../../../audit/new_video_intake_20261114/pipeline_status.json">Latest pipeline status</a> · <a href="pose_queue_full.csv">Approved native-frame queue</a> · <a href="pose_rois_full.json">Reviewed person crops</a> · <a href="../../../audit/new_video_intake_20261114/pilot_review.json">Pilot review</a></p><p>The separate doctor packet is built only after full pose inference, native-clock checks, short-clip rendering and decoding checks. Existing doctor packets are preserved.</p></section>')
    parts.append('<section><h2>Source screening</h2><table><tr><th>Recording</th><th>Duration</th><th>Finding</th></tr>')
    for r in sorted(manifest['sources'],key=lambda r:r['id']):
        parts.append('<tr><td><a href="'+esc(r['page'],quote=True)+'">'+esc(r['id'])+'</a><br><span class="small">'+esc(r['discovery_status'])+'</span></td><td>'+f"{r['duration_s']:.1f} s"+'</td><td>'+esc(r['screening'])+'</td></tr>')
    parts.append('</table></section><section><h2>What preprocessing preserves</h2><p>Each cut has a SHA-256 hash, native rational frame rate, decoded frame map, original-frame lineage and source URL. All nine derived videos passed strict decoding checks. Spatial changes are rigid 90° rotations or a documented inset crop. There is no temporal interpolation, missing-body reconstruction or perspective calibration.</p><p>NASA 2025 at 43.01–45.178 s is an edited still photograph of a squat, so it is excluded from repetition analysis. Thirsk and Glover partial views are excluded from the full hip–knee–ankle queue.</p><p>2D projected angles remain viewpoint-dependent. Model scores do not establish true joint visibility. Same-person and repeated-footage dependencies must remain grouped; a different publication is not automatically an independent subject.</p></section>')
    for s in selections:
        state='Approved for full native pose inference' if s['id'] in approved else ('Held after pose pilot' if s['pose'] else 'RGB retained; excluded from current squat pose run')
        parts.append('<section><h2>'+esc(s['id'])+'</h2><p><span class="tag">'+esc(s['domain'])+'</span> '+esc(state)+'</p><p>'+esc(s['notes'])+'</p><video controls preload="metadata" src="'+esc(str(Path(s['prepared_path']).relative_to(OUT)),quote=True)+'"></video><p>'+f"Source window {s['start']}–{s['end']} s; {s['frames']} frames at {s['native_rate']} fps."+'</p><p><a href="'+esc(s['domain']+'/'+s['id']+'/provenance.json',quote=True)+'">Source and transform provenance</a> · <a href="'+esc(s['domain']+'/'+s['id']+'/original_frame_lineage.csv',quote=True)+'">Original frame lineage</a></p></section>')
    parts.append('<p>Credits: NASA, ESA/NASA and CSA/NASA. Source terms continue to apply. This page contains screening findings, not doctor conclusions.</p></html>')
    (OUT/'index.html').write_text('\n'.join(parts))
    (OUT/'README.txt').write_text('Open index.html for all screened sources and prepared clips.\nUse pose_queue_full.csv and pose_rois_full.json for the four approved inference segments.\npose_queue.csv is the five-segment pilot queue; P17_INSET_01 is held and must not be promoted automatically.\nThe full background pipeline status is audit/new_video_intake_20261114/pipeline_status.json in the workspace.\nThis preparation folder is not the final doctor packet.\nSource clocks, crop/rotation transforms and all original-frame mappings are preserved in the per-clip folders.\n')


if __name__=='__main__':main()

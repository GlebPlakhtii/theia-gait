"""Build a local browseable evidence catalog and a candidate pose-input manifest."""
import html
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/prepared/healthy_locomotion_20261114'


def build():
    manifest=json.loads((OUT/'manifest.json').read_text())
    catalog=json.loads((OUT/'source_catalog.json').read_text())
    qc_path=OUT/'visual_qc.json'
    qc=json.loads(qc_path.read_text()) if qc_path.exists() else {}
    for clip in manifest['clips']:
        clip.update(qc.get(clip['source'],{}))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    clips=manifest['clips']
    counts=Counter((x['environment'],x['activity']) for x in clips)
    people=len({(x['source'],x['subject']) for x in clips})
    trials=len({x['trial_group'] for x in clips})
    queue={'stage':'lower_body_auxiliary_candidates_not_pose_inferred','clinical_form_labels':False,'full_body_eligible':False,
        'instructions':[
            'Use original timestamps and frames. Detect/track the participant; exclude empty-room, entry and exit frames.',
            'Keep simultaneous views in one trial and all clips of one participant in the same split.',
            'Require per-joint visibility. Do not fill a cropped head or trunk with invented reference anatomy.',
            'Compare walking and running separately, stratified by treadmill/overground and speed.',
            'Healthy-cohort status is not a correct-form label. Prosthetic gait needs clinician-defined goals rather than forced equality to this reference.'
        ], 'clips':clips}
    (OUT/'pose_input_candidates.json').write_text(json.dumps(queue,indent=2))
    esc=html.escape
    cards=[]
    for x in clips:
        note=x.get('visual_limitations','Visual check pending.')
        speed=f"{x['speed_kmh']} km/h" if x.get('speed_kmh') else x.get('speed_label','unspecified speed')
        cards.append(f'''<article data-environment="{x['environment']}" data-activity="{x['activity']}">
        <video controls preload="none" playsinline poster="{esc(x['poster'])}" src="{esc(x['path'])}"></video>
        <div class="body"><span class="tag">{x['environment']} · {x['activity']} · {esc(x['view'])}</span>
        <h3>{esc(x['subject'])} · {esc(str(speed))}</h3>
        <p>{x['duration_s']:.2f} s · {x['fps']:g} fps · {x['width']}×{x['height']} · {x['frames']} frames</p>
        <p class="muted">{esc(note)}</p>
        <details><summary>Provenance and preparation</summary><p>{esc(x['id'])}</p>
        <p>Original frame range [{x['source_start_frame']}, {x['source_end_frame_exclusive']}); original time {x['source_start_s']:.3f}–{x['source_end_s']:.3f} s.</p>
        <p>Pose estimation: not run. Clinical form verdict: unassigned.</p>
        <p>Group: {esc(x['trial_group'])}</p><a href="{esc(x['path'])}" download>Download clip</a></details></div></article>''')
    source_rows=[]
    for s in catalog['sources']:
        source_rows.append(f'''<tr><td><a href="{esc(s['dataset_url'])}">{esc(s['title'])}</a><br><a href="{esc(s['paper_url'])}">Paper</a></td>
        <td>{esc(s['environment'])}<br>{esc(', '.join(s['activities']))}</td>
        <td>{esc(s['population'])}</td><td>{esc(s['access'])}<br><small>{esc(s['license'])}</small></td></tr>''')
    summary=' · '.join(f'{n} {env} {act}' for (env,act),n in sorted(counts.items()))
    exclusions=''.join(f'<li><a href="{esc(x["url"])}">{esc(x["id"])}</a>: {esc(x["finding"])}</li>' for x in catalog['excluded_or_unconfirmed'])
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Theia | Healthy locomotion references</title><link rel="icon" href="data:,">
    <style>*{{box-sizing:border-box}}body{{margin:0;background:#0c1620;color:#e9f0f5;font:16px/1.5 system-ui}}main{{max-width:1400px;margin:auto;padding:36px 24px}}h1{{font-size:clamp(30px,4vw,48px);margin:8px 0}}h2{{margin-top:36px}}p{{margin:8px 0}}a{{color:#75e0cf}}.eyebrow,.tag{{color:#75e0cf;font-size:13px;letter-spacing:.04em}}.muted,small{{color:#b4c4d1}}.notice{{border-left:3px solid #e7bb69;padding:12px 18px;background:#192532;margin:22px 0}}.stats{{display:flex;gap:30px;flex-wrap:wrap;margin:24px 0}}.stats strong{{display:block;font-size:30px}}.controls{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:20px 0}}select{{padding:10px;background:#1a2a38;color:white;border:1px solid #3e566b;border-radius:6px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:20px}}article{{background:#162433;border:1px solid #2b4052;border-radius:12px;overflow:hidden}}article[hidden]{{display:none}}video{{width:100%;height:230px;background:black}}.body{{padding:16px}}h3{{margin:10px 0 4px}}details{{font-size:14px;overflow-wrap:anywhere}}summary{{cursor:pointer;color:#82d9d1}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:700px}}th,td{{padding:14px;text-align:left;vertical-align:top;border-bottom:1px solid #334858}}li{{margin:12px 0}}footer{{margin:40px 0;color:#b4c4d1}}</style>
    <main><span class="eyebrow">THEIA · REFERENCE SOURCE REVIEW · 14 NOV 2026</span><h1>Healthy walking & running</h1>
    <div class="notice"><b>Lower-body auxiliary collection.</b> These {len(clips)} clips do not meet the full-body requirement. <a href="../fullbody-reference/">Open the new full-body selection and 18-source comparison</a>.</div>
    <p class="muted">Treadmill and overground kept separate. Original RGB, source provenance, and clips for pose extraction.</p>
    <div class="stats"><div><strong>{people}</strong>sampled participants</div><div><strong>{trials}</strong>source trials</div><div><strong>{len(clips)}</strong>prepared video clips</div><div><strong>0</strong>clinical form verdicts</div></div>
    <div class="notice"><b>Reference motion, with clear limits.</b> These participants come from healthy study cohorts. Each stride has not been certified “correct.” No pose inference or new model training has been run on this collection. Use it to estimate healthy variation; the doctor must define which differences matter for prosthetic gait.</div>
    <p>{esc(summary)}</p><p><a href="manifest.json">Download provenance manifest</a> · <a href="pose_input_candidates.json">Pose input candidates</a> · <a href="source_catalog.json">Full source catalog</a> · <a href="README.txt">Read collection notes</a></p>
    <h2>Collected video</h2><div class="controls"><label>Environment <select id="environment"><option value="all">Both</option><option>treadmill</option><option>overground</option></select></label><label>Activity <select id="activity"><option value="all">Both</option><option>walk</option><option>run</option></select></label><span id="shown"></span></div>
    <section class="grid">{''.join(cards)}</section>
    <h2>Dataset shortlist</h2><div class="table-wrap"><table><thead><tr><th>Source</th><th>Coverage</th><th>Participants / evidence</th><th>Access and use</th></tr></thead><tbody>{''.join(source_rows)}</tbody></table></div>
    <details style="margin-top:24px"><summary>Sources excluded or still unconfirmed</summary><ul>{exclusions}</ul></details>
    <footer>Credits: Riglet, Delphin, Claquesin, Orliac, Ornetti, Laroche & Gueugnon (2024), Figshare CC0; Wang, Basu, Durandau & Sartori (2023), video release CC BY 4.0. Face anonymization is preserved. Clinical labels remain unassigned.</footer></main>
    <script>const env=document.getElementById('environment'),act=document.getElementById('activity');function filter(){{let n=0;for(const c of document.querySelectorAll('article')){{c.hidden=!((env.value==='all'||env.value===c.dataset.environment)&&(act.value==='all'||act.value===c.dataset.activity));if(!c.hidden)n++;else c.querySelector('video').pause()}}document.getElementById('shown').textContent=n+' clips shown'}}env.addEventListener('change',filter);act.addEventListener('change',filter);filter();</script></html>'''
    (OUT/'index.html').write_text(page)
    readme=f'''THEIA — HEALTHY LOCOMOTION REFERENCE COLLECTION\nPrepared 2026-11-14\n\n{people} sampled participants; {len(manifest['sources'])} camera recordings; {trials} source trials; {len(clips)} clips.\n{summary}\n\nOpen index.html or http://localhost:8006/healthy-reference/ while the existing local server is running.\n\nThis is a source/reference collection. No newly inferred poses, trained model, accuracy result or clinician-certified correct-form labels are included. Healthy eligibility is supported by the papers at cohort level, not by visual diagnosis or individual medical records.\n\nRiglet: 6 participants, Session1, comfortable overground walking and running. Three nonblack passes selected per camera recording. Black gaps removed without joining separate passes; all native frames in each interval retained. Mostly lower-body views: do not use as full-body/trunk ground truth. Body entry/exit still requires masking.\n\nTwente: retained study participant IDs only (04 onward; 01–03 are excluded pilots). Treadmill views of the same trial share trial_group. Study paper reports 9 retained healthy participants; archive description reports 12 recorded. Face anonymization remains intact.\n\nUse the JSON manifest to preserve frame/time mapping, camera, speed, subject and source. All clips from a participant must stay in one train/test split, including both camera views. Frame count verified by decoding each MP4. View sample-level QC in visual_qc.json; it is not a clinical review.\n\nFor reference comparison, condition on activity, treadmill/overground, speed, age/body dimensions where available, camera geometry and visible joints. For amputation rehabilitation, healthy gait provides a comparator, not an automatic target that every prosthesis user must match. Your clinician should set appropriate goals.\n\nSources and licenses: source_catalog.json. Original downloads: data/raw/healthy_locomotion_20261114 under project root. Acquisition evidence and archive indexes: audit/healthy_locomotion_20261114. Selective archive members were checked by CRC; whole-archive hashes were not verified because the archives were not downloaded in full. Source hashes and derivative clip hashes are in the manifests.\n'''
    (OUT/'README.txt').write_text(readme)
    print(summary)


if __name__=='__main__':build()

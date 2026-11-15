"""Local gallery: source diversity, actual downloaded clips, and explicit QC."""
import html
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/prepared/fullbody_locomotion_20261114'
E=html.escape

def build():
    catalog=json.loads((OUT/'source_catalog.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    clips=manifest['clips']
    qcpath=OUT/'visual_qc.json'
    qc=json.loads(qcpath.read_text()) if qcpath.exists() else {}
    for c in clips:
        if c['id'] in qc:c.update(qc[c['id']])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    accepted=[c for c in clips if c['framing_status']=='sampled_full_body_pass']
    healthy=[c for c in accepted if c['role']=='reported_healthy_cohort']
    people=len({(c['source'],c['subject']) for c in accepted})
    trials=len({c['trial_group'] for c in accepted})
    sources={s['id']:s for s in catalog['sources']}
    for s in catalog['sources']:
        s['downloaded_sample']=s['id'] in {'ljmu360','lbmc','movi','mpi3dhp'}
        n=sum(c['source']==s['id'] for c in accepted)
        if n:s['full_body_status']=f'{n} local cuts pass sampled framing review; not a per-frame visibility certification'
    (OUT/'source_catalog.json').write_text(json.dumps(catalog,indent=2,ensure_ascii=False))
    counts=Counter(c['source'] for c in accepted)
    cards=[]
    for c in clips:
        source=sources[c['source']]
        health='Published healthy-cohort candidate' if c['role']=='reported_healthy_cohort' else 'Viewpoint sample · health unverified'
        accepted_flag='yes' if c['framing_status']=='sampled_full_body_pass' else 'no'
        cards.append(f'''<article class="clip" data-source="{E(c['source'])}" data-subject="{E(c['source']+':'+c['subject'])}" data-role="{E(c['role'])}" data-env="{E(c['environment'])}" data-activity="{E(c['activity'])}" data-pass="{accepted_flag}">
<video controls playsinline preload="none" poster="{E(c['poster'])}" src="{E(c['path'])}"></video>
<div class="card-body"><div class="eyebrow">{E(source['title'])}</div><h3>{E(c['activity'].capitalize())} · {E(c['view'])}</h3>
<p><b>{E(c['subject'])}</b> · {E(c['environment'])} · {c['duration_s']:.2f}s · {c['width']}×{c['height']} · {c['fps']:g} fps</p>
<p class="badge">{E(health)}</p><p class="muted">{E(c['notes'])}</p>
{('<p class="notice"><b>Withheld:</b> '+E(c.get('framing_review_note',''))+'</p>') if accepted_flag=='no' else ''}
<details><summary>Framing, source frames & use</summary><p>{E(c['framing_status'])}. {E(c.get('framing_review_note','Final visual review pending.'))}</p>
<p>Original frames [{c['start_frame']}, {c['end_frame_exclusive']}); {c['source_start_s']:.3f}–{c['source_end_s']:.3f}s. {c['frames']} native frames preserved.</p>
<p>{E(c['activity_label_basis'])}</p><p>Participant {E(c['subject'])}; group {E(c['trial_group'])}.</p>
<p>{E(source['rights'])}</p><p><a href="{E(c['qc_sheet'])}">Framing contact sheet</a> · <a href="{E(source['dataset_url'])}">Publisher</a></p>
<p>No clinical form verdict; full-rate pose and biomechanics analysis not run. {('Sparse Keypoint R-CNN framing pre-screen completed.' if c.get('framing_screen') else 'Visual framing review only.')}</p></details></div></article>''')
    rows=[]
    for i,s in enumerate(catalog['sources'],1):
        acquired=s['downloaded_sample']
        status=f'{counts[s["id"]]} prepared framing samples' if acquired else 'Source reviewed · not downloaded'
        rows.append(f'''<tr class="source" data-local="{'yes' if acquired else 'no'}" data-priority="{s['priority']}">
<td><b>{i:02d}</b> <a href="{E(s['dataset_url'])}">{E(s['title'])}</a><p class="badge">{E(status)}</p></td>
<td>{E(s['views'])}</td><td>{E(', '.join(s['activities']))}<p class="muted">{E(s['environment'])}</p></td>
<td>{E(s['population'])}<details><summary>Health evidence</summary>{E(s['health_evidence'])}</details></td>
<td>{E(s['access'])}<details><summary>Limits, terms & evidence</summary><p>{E(s['limitations'])}</p><p>{E(s['rights'])}</p><p>{E(s['full_body_status'])}</p><a href="{E(s['paper_url'])}">Primary reference / official catalog</a></details></td></tr>''')
    opts=''.join(f'<option value="{E(k)}">{E(sources[k]["title"])}</option>' for k in sorted(counts))
    actopts=''.join(f'<option value="{E(k)}">{E(k)}</option>' for k in sorted({c['activity'] for c in clips}))
    personopts=''.join(f'<option value="{E(s+":"+p)}">{E(s+" / "+p)}</option>' for s,p in sorted({(c['source'],c['subject']) for c in accepted}))
    excluded=''.join(f'<li><b>{E(x["id"])}:</b> {E(x["reason"])}</li>' for x in catalog['excluded'])
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">
<title>Theia | Full-body gait sources</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#0a1420;color:#edf4f8;font:16px/1.5 system-ui}}main{{max-width:1500px;margin:auto;padding:30px 24px}}a{{color:#71decd}}h1{{font-size:clamp(30px,4.3vw,52px);line-height:1.1;margin:12px 0 18px}}h2{{margin:36px 0 12px}}h3{{font-size:18px;margin:8px 0}}p{{margin:8px 0}}.eyebrow{{color:#71decd;font-size:12px;letter-spacing:.08em}}.muted{{color:#b1c2d0}}nav{{display:flex;gap:22px;flex-wrap:wrap;margin:20px 0}}.stats{{display:flex;gap:36px;flex-wrap:wrap;margin:24px 0}}.stats strong{{display:block;font-size:32px}}.notice{{padding:16px 20px;background:#192a39;border-left:3px solid #e9bc70;margin:22px 0}}.controls{{display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin:20px 0}}label{{font-size:14px;color:#b1c2d0}}select{{display:block;background:#162839;color:white;border:1px solid #3c566a;border-radius:6px;padding:10px;max-width:100%}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:20px}}article{{border:1px solid #294254;background:#142333;border-radius:12px;overflow:hidden}}article[hidden],tr[hidden]{{display:none}}video{{width:100%;height:340px;object-fit:contain;background:#030a10;display:block}}.card-body{{padding:18px}}.badge{{display:inline-block;border-radius:5px;background:#223a49;color:#a6eadf;padding:4px 8px;font-size:12px}}details{{margin:8px 0;font-size:14px;overflow-wrap:anywhere}}summary{{cursor:pointer;color:#8cdecf}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:1000px}}td,th{{text-align:left;vertical-align:top;padding:15px 12px;border-bottom:1px solid #294254}}td:first-child{{min-width:200px}}td:last-child{{min-width:290px}}th{{color:#92b1c6;font-size:13px}}li{{margin:12px 0}}footer{{margin:40px 0;color:#9fb4c4;font-size:14px}}#shown{{color:#b1c2d0}}
</style><main><div class="eyebrow">THEIA · SOURCE REVIEW · 14 NOVEMBER 2026</div>
<h1>Whole body.<br>More camera angles.</h1><p class="muted">Walking and running source research, with actual RGB samples and honest access status.</p>
<nav><a href="#samples">Watch samples</a><a href="#datasets">Compare all 18 sources</a><a href="source_catalog.json">Source catalog JSON</a><a href="manifest.json">Frame provenance</a><a href="pose_input_candidates.json">Pose input candidates</a></nav>
<div class="stats"><div><strong>{len(accepted)}</strong>clips passing sampled framing</div><div><strong>{people}</strong>sampled participants</div><div><strong>4</strong>sources sampled locally</div><div><strong>18</strong>sources researched</div></div>
<p><b>{len(healthy)} clips from a published healthy-gait cohort.</b> {len(accepted)-len(healthy)} additional clips support camera-view testing with health status unverified. The {len(accepted)} clips cover {trials} source trial/action groups; multiple camera views are not independent trials. Clinical form labels remain unassigned.</p>
<div class="notice"><b>Head-to-foot framing review.</b> The default view shows only accepted candidates. {len(clips)-len(accepted)} prepared candidates are withheld or awaiting review. New cuts use a 5 Hz model framing pre-screen plus visual checks of five frames including both boundaries; earlier cuts use approximately 2 Hz visual review. This is not an every-frame visibility certification. The original 44 cropped clips remain <a href="../healthy-reference/">lower-body auxiliary material</a>.</div>
<p><b>What is ready:</b> local samples from LJMU, LBMC, MoVi and MPI-INF-3DHP. <b>What is researched:</b> 18 independent sources, including gated collections. The source count is not a download count.</p>
<p class="muted">Only LJMU among this local batch is described as a healthy-gait cohort in the reviewed publication. The other three demonstrate camera/framing coverage; their participants have not been medically verified here. Ordinary forward running from a documented healthy cohort remains an acquisition gap; BioCV is the strongest verified source lead. MoVi jogging and running in place are labeled separately.</p>
<h2 id="samples">Downloaded full-body samples</h2><div class="controls">
<label>Dataset<select id="dataset"><option value="all">All downloaded sources</option>{opts}</select></label>
<label>Participant<select id="person"><option value="all">All participants</option>{personopts}</select></label>
<label>Evidence<select id="role"><option value="all">All evidence levels</option><option value="reported_healthy_cohort">Published healthy cohort</option><option value="viewpoint_only">Health unverified · camera testing</option></select></label>
<label>Environment<select id="env"><option value="all">All</option><option>treadmill</option><option>overground</option><option>in-place</option></select></label>
<label>Activity<select id="act"><option value="all">All</option>{actopts}</select></label>
<label>Framing<select id="framing"><option value="yes">Passed sampled review</option><option value="all">All prepared candidates</option></select></label><span id="shown" role="status"></span></div>
<section class="grid">{''.join(cards)}</section>
<h2 id="datasets">18 distinct datasets and source collections</h2><p class="muted">Best healthy-cohort leads come first. Viewpoint benchmarks follow; unknown health status never becomes a “healthy” label based on appearance.</p>
<div class="controls"><label>Source selection<select id="sourceFilter"><option value="all">All 18 reviewed sources</option><option value="local">Downloaded samples only</option><option value="priority">Priority healthy-cohort leads</option></select></label><span id="sourceCount" role="status"></span></div>
<div class="table-wrap"><table><thead><tr><th>Source / actual local status</th><th>Camera coverage</th><th>Activity / environment</th><th>Population</th><th>Access, limitations and evidence</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<details style="margin-top:24px"><summary>Excluded or downgraded sources</summary><ul>{excluded}</ul></details>
<footer>Clips retain native frame counts and resolution; H.264 copies are re-encoded for local playback. Clinical form labels are unassigned. Sparse landmark inference was used only to propose framing intervals; full-rate Sapiens2/biomechanical analysis and model training have not run on this collection. Respect each dataset’s license; this local inspection gallery is not a redistribution package. <a href="README.txt">Collection and verification notes</a>.</footer></main>
<script>
const dataset=document.getElementById('dataset'),env=document.getElementById('env'),act=document.getElementById('act'),framing=document.getElementById('framing'),person=document.getElementById('person'),role=document.getElementById('role');
function filterClips(){{let count=0;for(const c of document.querySelectorAll('.clip')){{const visible=(dataset.value==='all'||c.dataset.source===dataset.value)&&(person.value==='all'||c.dataset.subject===person.value)&&(role.value==='all'||c.dataset.role===role.value)&&(env.value==='all'||c.dataset.env===env.value)&&(act.value==='all'||c.dataset.activity===act.value)&&(framing.value==='all'||c.dataset.pass===framing.value);c.hidden=!visible;if(visible)count++;else c.querySelector('video').pause()}}document.getElementById('shown').textContent=count+' clips shown'}}
for(const c of [dataset,env,act,framing,person,role])c.addEventListener('change',filterClips);filterClips();
const sourceFilter=document.getElementById('sourceFilter');function filterSources(){{let count=0;for(const r of document.querySelectorAll('tr.source')){{r.hidden=!(sourceFilter.value==='all'||(sourceFilter.value==='local'&&r.dataset.local==='yes')||(sourceFilter.value==='priority'&&r.dataset.priority==='1'));if(!r.hidden)count++}}document.getElementById('sourceCount').textContent=count+' sources shown'}}sourceFilter.addEventListener('change',filterSources);filterSources();
document.addEventListener('play',event=>{{if(event.target.tagName==='VIDEO')for(const v of document.querySelectorAll('video'))if(v!==event.target)v.pause()}},true);
</script></html>'''
    (OUT/'index.html').write_text(page)
    pose_dir=OUT/'pose_inputs'
    pose_dir.mkdir(exist_ok=True)
    expected={Path(c['path']).name for c in accepted}
    for old in pose_dir.glob('*.mp4'):
        if old.is_symlink() and old.name not in expected:old.unlink()
    for c in accepted:
        target=pose_dir/Path(c['path']).name
        if not target.exists():target.symlink_to(Path('..')/c['path'])
    queue=dict(stage='candidate_inputs_no_full_rate_pose_analysis',clinical_form_labels=False,
        base_directory=str(OUT),accepted_video_directory=str(pose_dir),
        policy=['Keep full subject and all views in one split.','Require per-joint visibility; no invented cropped anatomy.',
                'Framing pass is based on sampled review, not every-frame visibility certification.',
                'Confirm dataset terms before model training, especially LBMC preview.',
                'Use clinically unverified datasets for pose/view robustness only, not healthy-form supervision.',
                'Preserve treadmill/overground/in-place separation and trial grouping.'],clips=accepted)
    (OUT/'pose_input_candidates.json').write_text(json.dumps(queue,indent=2))
    (OUT/'README.txt').write_text(f'''THEIA — FULL-BODY LOCOMOTION SOURCE EXPANSION
Prepared 2026-11-14

18 distinct sources assessed. Four additional sources actually downloaded: LJMU, LBMC public anonymized preview, BML-MoVi and MPI-INF-3DHP.
{len(accepted)} clips pass sampled framing review; {len(clips)} prepared candidates; {sum(c['frames'] for c in accepted)} selected frames.
{len(healthy)} clips have published healthy-cohort support; {people} sampled participants and {trials} source trial/action groups in total. Camera views of a trial remain grouped.
{len(clips)-len(accepted)} prepared candidates are withheld or pending; consult visual_qc.json for each reason. Rejections before cutting are recorded in audit/fullbody_locomotion_20261114/expansion_*_plan.json. Withheld candidates are excluded from pose_input_candidates.json.
Open http://localhost:8006/fullbody-reference/.
The pose_inputs/ directory contains only the {len(accepted)} retained clips as local symlinks; use pose_input_candidates.json for original paths, roles and provenance. These are candidates for pose extraction, not approved form-training examples. Do not glob the clips/ audit collection, which also contains withheld and superseded cuts.

Full body means the target's head-to-foot extent is inside the image. It does not guarantee that every joint is unoccluded in every frame. See per-clip notes and contact sheets. No clinical good-form labels or training. New candidates use torchvision Keypoint R-CNN ResNet50 FPN COCO_V1 at 5 Hz for framing proposals, followed by five-frame visual review including boundaries. This sparse pre-screen is not full-rate pose/biomechanical analysis. Earlier cuts retain their approximately 2 Hz visual review method.

LJMU: published healthy-gait description, not individually verified medical records. Front/rear source files are trimmed before/after close approach to camera. Other cuts avoid entry/exit. Dark backlighting is retained. Nominal camera labels are not calibrated anatomical yaw. Separate camera files have different trims and must not be treated as automatically frame-aligned.
LBMC: nine views of the same treadmill trial and same participant. Face anonymization remains. These nine videos are not nine independent people/trials. Raw release is restricted; only explicitly public anonymized preview acquired. Confirm use terms before training or redistribution.
MoVi: PG1/PG2 stationary videos and CP1/CP2 handheld videos downloaded. Gallery uses PG videos, with original motions_list/flags30 boundaries. MATLAB inclusive 1-based ranges are converted to half-open 0-based ranges. Handheld clips have different durations and are not falsely time-aligned to PG. Jogging, walking and running in place stay separate.
MPI-INF-3DHP: selected intervals from S1 Seq1, two views. Walking-transition/gesture material is for viewpoint robustness, not clinically approved normal walking.

Each clip records original file/hash, exact frames/time, camera, participant, trial group, role and license in manifest.json/source_catalog.json. All source AVI/MP4 hashes and acquisition checks are under audit/fullbody_locomotion_20261114. ZIP-member CRCs were verified; MoVi source-video MD5s matched publisher catalog. Full large archives were not downloaded and no whole-archive checksum claim is made.

The previous 44 clips are retained as lower-body auxiliary data, not counted as full-body references. No emails, agreement submissions, account registrations, uploads or public publication were performed.
''')
    print('Gallery',len(accepted),'accepted framing candidates;',len(catalog['sources']),'sources')

if __name__=='__main__':build()

"""Build a portable HTML research report from completed, verified artifacts."""
import base64
import csv
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/local_execution_20261114'
OUT=ROOT/'reports/theia_results_2026-11-14.html'


def read(name):
    return json.loads((RUN/name).read_text())


def media(path,mime):
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()


def main():
    verification=read('verification.json')
    assert verification['verified']
    datasets=read('clinical_results.json')
    cycles=read('squat_candidates.json')
    cases=read('clinical_paired_cases.json')
    review=read('ared_visual_review.json')
    protocol=read('protocol.json')
    complete=read('complete.json')
    # Ensure the report reads the original completed run, not modified results.
    for name in ['clinical_results.json','clinical_predictions.json','squat_candidates.json','protocol.json']:
        assert hashlib.sha256((RUN/name).read_bytes()).hexdigest()==complete['artifacts'][name]
    signal=[[float(r['source_time_s']),float(r['median_filtered_deg']) if r['median_filtered_deg'] else None]
            for r in csv.DictReader(open(RUN/'ared_670_779_signals.csv'))]
    names={'CTK_arm_elevation':'Arm elevation · CTK','ELK_pelvis_relative':'Side bending · pelvis-relative',
           'ELK_camera_relative':'Side bending · camera-relative'}
    body='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Theia research results: freshly fitted clinical comparisons, ARED RGB pose extraction, candidate cycles and known failures.">
<title>Theia · Results so far · 14 November 2026</title>
<style>
:root{--ink:#142e3d;--muted:#536b79;--line:#d8e2e7;--paper:#fff;--bg:#f2f6f8;--blue:#126e98;--green:#087765;--amber:#a76a14;--red:#b34251}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{background:#102d40;color:#fff;padding:45px max(24px,calc((100vw - 1180px)/2));border-bottom:5px solid #48a7ad}header .eyebrow{letter-spacing:.16em;text-transform:uppercase;font-size:12px;color:#a6d4e4;font-weight:700}h1{font-size:clamp(32px,5vw,52px);line-height:1.08;letter-spacing:-.035em;margin:13px 0 18px}header p{max-width:920px;color:#d7e6ee;font-size:19px;margin:0}header small{display:block;margin-top:20px;color:#a6c3d3}
nav{position:sticky;top:0;z-index:3;background:#fffefc;border-bottom:1px solid var(--line);display:flex;gap:24px;padding:12px max(24px,calc((100vw - 1180px)/2));overflow:auto;white-space:nowrap}a{color:var(--blue)}nav a{text-decoration:none;font-size:14px;font-weight:650}
main{max-width:1228px;padding:28px 24px 55px;margin:auto}section{background:white;border:1px solid var(--line);border-radius:12px;padding:28px;margin:22px 0;scroll-margin-top:76px}h2{font-size:26px;letter-spacing:-.025em;line-height:1.2;margin:0 0 12px}h3{font-size:18px;line-height:1.3;margin:20px 0 8px}p{margin:10px 0}.muted,.small{color:var(--muted)}.small{font-size:13px}.lead{font-size:18px;max-width:1000px}.notice{padding:14px 18px;border-radius:6px;background:#eaf3f7;border-left:4px solid var(--blue);margin:18px 0}.notice.warn{background:#fff5e8;border-color:#cc8b32}.notice.fail{background:#fff0f1;border-color:var(--red)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stat{padding:20px;background:#fff;border:1px solid var(--line);border-radius:10px}.stat strong{display:block;font-size:32px;line-height:1.2}.stat span{display:block;color:var(--muted);font-size:13px;margin-top:8px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.badge{display:inline-block;border-radius:99px;padding:3px 10px;background:#edf3f6;font-size:12px;font-weight:650}.badge.good{background:#e6f5ef;color:#087060}.badge.warn{background:#fff0dc;color:#965d11}.badge.bad{background:#fde9ec;color:#a53b4a}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{font-weight:700;background:#edf3f6;color:#3f5969}tbody tr:hover{background:#f6fafc}td.num{white-space:nowrap;font-variant-numeric:tabular-nums}.scroll{overflow:auto}.positive{color:var(--green);font-weight:700}.negative{color:var(--red);font-weight:700}.flag{background:#fff4e5}details{margin:18px 0;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;font-weight:650}button,select,input{font:inherit}button{cursor:pointer;border:1px solid #a2bbc8;background:#f5fafc;color:#155e7e;border-radius:6px;padding:7px 12px}button:hover{background:#e8f2f7}button.primary{background:#156a8c;color:#fff;border-color:#156a8c}select,input{padding:7px 10px;border:1px solid #a2bbc8;border-radius:5px;background:white;color:var(--ink)}label{font-size:14px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:15px 0}.figure{width:100%;height:auto;display:block;border-radius:6px}video{display:block;width:100%;max-height:620px;background:#10232d;border-radius:8px}canvas{width:100%;height:200px;background:#f6f9fb;cursor:crosshair;display:block;border-radius:5px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--muted)}.swatch{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px}#clock{font-variant-numeric:tabular-nums}code{font-size:12px;overflow-wrap:anywhere}.downloads{display:flex;gap:10px;flex-wrap:wrap}footer{color:var(--muted);font-size:13px;padding:10px 0}ul{padding-left:23px}li{margin:6px 0}
@media(max-width:760px){.stats{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}main{padding:15px 12px}section{padding:20px 15px}nav{gap:18px;padding:10px 16px}header{padding:32px 20px}th,td{padding:9px 8px}h2{font-size:23px}}
@media print{nav,.controls,.downloads,button,video,canvas{display:none!important}body{background:white;font-size:11px}header{padding:20px;background:#fff;color:#142e3d}header p,header small{color:#536b79}main{padding:0}section{break-inside:avoid;border-radius:0;padding:16px}.stats{grid-template-columns:repeat(4,1fr)}details{display:block}h1{font-size:30px}.no-print{display:none}}
</style></head><body>
<header><div class="eyebrow">Theia / Research progress</div><h1>Results we have so far</h1>
<p>Clinical records improved two exploratory movement-assessment settings. The ARED pipeline now produces reviewable squat candidates, including visible failure cases. Benefit for an in-mission support decision remains untested.</p>
<small>14 November 2026 · Results run 09:49 Europe/Kyiv · Built BUILD_TIME · Local evidence only</small></header>
<nav aria-label="Report sections"><a href="#summary">Summary</a><a href="#clinical">Clinical comparison</a><a href="#ared">ARED replay</a><a href="#gym">Gym footage</a><a href="#limits">Interpretation</a><a href="#evidence">Evidence</a></nav>
<main><div class="stats"><div class="stat"><strong>6</strong><span>Clinical patients, each held out of training in turn</span></div><div class="stat"><strong>108</strong><span>New model fits across 18 evaluation configurations</span></div><div class="stat"><strong>109 s</strong><span>Continuous ARED footage · 2,725 native frames</span></div><div class="stat"><strong>23</strong><span>ARED cycle candidates · 3 suspect transitions</span></div></div>
<section id="summary"><h2>What the current evidence says</h2><div class="grid">
<div><h3><span class="badge good">Measured result</span> Clinical-data comparison</h3><p>With the primary configuration, mean patient accuracy rose from <strong>66.1% to 77.5%</strong> for arm elevation and from <strong>58.9% to 71.5%</strong> for camera-relative side bending. Both false alerts and misses fell.</p><p>Pelvis-relative side bending worsened: mean patient accuracy fell from <strong>43.9% to 37.1%</strong>, with <strong>8 → 27 missed errors</strong>.</p></div>
<div><h3><span class="badge warn">Needs expert review</span> In-mission video</h3><p>Sapiens2-5B poses were processed through a fixed cycle detector. A subsequent AI inspection of selected original frames found <strong>20 plausible squats</strong> and <strong>three suspect transition detections</strong>.</p><p>These are development observations. There are no doctor-reviewed labels yet, so event precision, recall, form quality and operational decision accuracy are unknown.</p></div></div>
<p class="notice">The clinical comparison uses archived KERAAL numerical features for arm elevation and side bending. The ARED experiment uses poses extracted from real RGB video. <strong>No model has yet demonstrated clinical-to-ARED decision transfer.</strong></p></section>
<section id="clinical"><h2>Clinical comparison: adding patient records</h2><p class="lead">Healthy-only training versus healthy plus other patients’ records, evaluated on the same held-out patient.</p>
<p>For each of six patients, that person’s records are excluded from training, scaling and weighting. The augmented arm adds the other five patients. Fixed weighted logistic regression uses a 0.5 decision threshold. The primary L2 penalty, 0.1, and both sensitivity settings were recorded before these new fits.</p>
<div class="controls"><label>Preset regularization <select id="lambda"><option value="0.1">0.1 — primary</option><option value="0.01">0.01 — sensitivity</option><option value="1">1.0 — sensitivity</option></select></label><span class="small">Every preset is retained; no best-result selection.</span></div>
<div class="scroll"><table><thead><tr><th>Task / feature geometry</th><th>False alerts<br>healthy → +clinical</th><th>Missed errors<br>healthy → +clinical</th><th>Mean patient accuracy</th><th>Change</th></tr></thead><tbody id="clinicalRows"></tbody></table></div>
<p class="small">False alerts use negative-record denominators; misses use positive-record denominators. “Mean patient accuracy” first computes accuracy within each patient, then averages across six patients. A negative label means the specified error code is absent, not that the whole movement is correct.</p>
<img class="figure" src="CLINICAL_CHART" alt="Primary-configuration false alerts and missed errors for healthy-only and clinical-augmented training">
<div class="grid"><div><h3>Sample size</h3><ul><li><strong>CTK arm elevation:</strong> 69 patient records: 17 positives, 52 negatives. Healthy training: 20 records from four people.</li><li><strong>ELK side bending:</strong> 53 patient records: 40 positives, 13 negatives. Healthy training: only four records from two people.</li><li>The ELK geometry settings reuse the same records. These are <strong>122 unique patient records</strong> from six people, not 175 independent observations.</li></ul></div>
<div><h3>What changed in training</h3><p>Clinical augmentation adds labeled patient movement examples to an otherwise identical model and feature set. Features have four dimensions for CTK and six for ELK. Training participants receive equal initial total weight, followed by balanced class totals.</p><p class="notice warn">The augmented arm has more data and more people. This run does not isolate a rehabilitation-specific causal effect from sample-size or diversity effects.</p></div></div>
<details><summary>All preset configurations and uncertainty</summary><div class="scroll"><table><thead><tr><th>Feature setting</th><th>L2</th><th>Healthy FP / FN</th><th>+clinical FP / FN</th><th>Mean-accuracy change</th><th>Descriptive 95% interval</th></tr></thead><tbody id="allConfigs"></tbody></table></div><p class="small">Intervals use 10,000 paired patient-cluster bootstrap resamples, seed 20261114. Six previously explored patients and repeated development analyses limit inference; these are not independent confirmatory confidence claims.</p></details>
<details><summary>Every helped, harmed and unchanged primary-configuration case</summary><div class="controls"><label>Outcome <select id="caseFilter"><option value="all">All cases</option><option value="helped">Helped</option><option value="harmed">Harmed</option><option value="unchanged">Unchanged</option></select></label><label>Task <select id="taskFilter"><option value="all">All feature settings</option><option value="CTK_arm_elevation">Arm elevation</option><option value="ELK_pelvis_relative">Side bending / pelvis</option><option value="ELK_camera_relative">Side bending / camera</option></select></label><span id="caseCount" class="small"></span></div><div class="scroll"><table><thead><tr><th>Features / record</th><th>Reference error label</th><th>Healthy prediction</th><th>+clinical prediction</th><th>Outcome</th></tr></thead><tbody id="caseRows"></tbody></table></div><p class="small">1 = specified error detected; 0 = not detected. Unchanged cases include both correct and incorrect predictions. Labels are inherited annotation-derived record labels.</p></details>
<details><summary>How this differs from the earlier reported results</summary><p>The historical baseline already included patient records and tuned thresholds within training folds. Its extra-source comparison did not show a useful gain. The current baseline uses healthy records only, with a fixed threshold. The new result answers a different question and does not replace or reverse the earlier finding.</p>
<div class="scroll"><table><thead><tr><th>Archived experiment</th><th>Baseline FP / FN</th><th>Extra-source FP / FN</th><th>Status</th></tr></thead><tbody><tr><td>CTK patient-transfer v2</td><td>7 / 4</td><td>8 / 5</td><td>Saved predictions recounted; original model code unavailable</td></tr><tr><td>ELK camera/trunk geometry</td><td>5 / 0</td><td>5 / 2</td><td>Post-hoc historical result; not reproduced by the new fixed-threshold model</td></tr></tbody></table></div></details></section>
<section id="ared"><h2>ARED: movement, pose and detected cycles</h2><p>One development recording of Sophie Adenot on ARED. Source interval <strong>670–779 seconds</strong>, 25 frames/s. The replay shows original RGB on the left and estimated pose on the right. Credit: ESA/NASA.</p>
<video id="replay" controls preload="metadata" poster="POSTER" src="VIDEO"></video>
<div class="controls"><strong id="clock">Source time 670.00 s</strong><button id="fullReplay">Play full segment</button><button id="pause">Pause</button><label>Speed <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option></select></label></div>
<canvas id="timeline" width="1120" height="200" aria-label="Projected knee-angle timeline. Click to seek."></canvas>
<div class="legend"><span><i class="swatch" style="background:#126e98"></i>Filtered projected angle</span><span><i class="swatch" style="background:#158670"></i>Candidate completion</span><span><i class="swatch" style="background:#d28a24"></i>Suspect transition</span><span>Click timeline to seek</span></div>
<p class="notice warn"><strong>Three visible failure candidates:</strong> #11, #12 and #23 occur during turning, repositioning or stepping in the sampled original frames. Their detector outputs remain unchanged. This post-prediction AI review is not independent clinical ground truth.</p>
<div class="controls"><button data-candidate="11">Watch #11 · repositioning</button><button data-candidate="12">Watch #12 · shallow transition</button><button data-candidate="23">Watch #23 · stepping</button><span id="playing" class="small"></span></div>
<div class="grid"><div><h3>Automatic measurement</h3><p>The detector uses the mean left/right projected hip–knee–ankle angle, a roughly 0.2-second median filter, and a primary 15° prominence. It requires a trough bounded by return maxima, a 1–8-second duration, and adequate model scores without interpolating gaps.</p><p>This is an offline detector: the centered filter and following maximum use later frames. It is not yet a causal, real-time decision system.</p></div><div><h3>What is not measured</h3><p>Projected angles depend on viewpoint and are not validated anatomical flexion. Heatmap scores are not visibility probabilities. No load, pain, fatigue, injury, muscle atrophy or safe technique is inferred.</p><p>All 2,725 ARED frames pass the numerical score gate; that alone does not establish anatomical accuracy or observability.</p></div></div>
<div class="scroll"><table><thead><tr><th>Prominence preset</th><th>Candidate count</th><th>Reference accuracy</th></tr></thead><tbody><tr><td>10° sensitivity</td><td>24</td><td>Unknown</td></tr><tr><td><strong>15° primary</strong></td><td><strong>23</strong></td><td>Unknown</td></tr><tr><td>20° sensitivity</td><td>21</td><td>Unknown</td></tr></tbody></table></div>
<details><summary>Play any of the 23 detected candidates</summary><p class="small">Buttons play the selected interval with a little context, then pause. These are time ranges within the replay; separate doctor-review clip files have not yet been exported.</p><div class="scroll"><table><thead><tr><th>Watch</th><th>Source start / bottom / completion</th><th>Duration</th><th>Projected excursion</th><th>Sampled-frame review</th></tr></thead><tbody id="eventRows"></tbody></table></div></details>
<details><summary>Original RGB evidence: all 23 candidate triplets</summary><p class="small">Start, minimum-angle and completion frames were inspected after predictions. These selected frames cannot establish missed repetitions or event-level accuracy.</p>CONTACT_SHEETS</details>
<details><summary>Static signal plot</summary><img class="figure" src="ARED_CHART" alt="All ARED candidate cycles over the 109-second projected knee-angle trace"></details></section>
<section id="gym"><h2>Additional squat footage processed</h2><p>The same fixed detector ran on five eligible gym clips: <strong>2,343 frames and 69 candidates</strong>. The participants’ health status is undocumented, so this is not a verified healthy control cohort. Published good/bad filename tags are source labels, not our expert conclusions.</p>
<div class="scroll"><table><thead><tr><th>Source file</th><th>Duration</th><th>Native frames</th><th>Candidates</th><th>Doctor-confirmed form</th></tr></thead><tbody>GYM_ROWS</tbody></table></div>
<details><summary>All five gym signal plots</summary>GYM_CHARTS</details>
<p class="small">Across the original and extension batches, pose inference completed 6,527 native ARED/gym frames. The current analysis uses 5,068 frames after excluding three gym intervals with body-masking concerns. An additional 12-frame walking crop pilot is separate from these squat results.</p></section>
<section id="limits"><h2>Interpretation and next validation</h2><div class="grid"><div><h3>Supported now</h3><ul><li>Fresh held-patient-out fits show a gain from adding clinical examples in two feature settings under the primary configuration.</li><li>The pelvis-relative setting shows a clear failure: fewer false alerts but many more misses.</li><li>High-resolution pose extraction and offline candidate-cycle generation run on actual ARED video.</li><li>Timing, predictions and confusion counts are traceable to saved artifacts.</li></ul></div><div><h3>Still unresolved</h3><ul><li>No demonstrated rehabilitation benefit on an astronaut support decision.</li><li>No compatible clinical/verified-healthy squat RGB comparison has been assembled.</li><li>All six clinical patients and the ARED recording have prior development exposure.</li><li>No doctor conclusions have been received. The doctor’s requested job is to watch short presegmented clips and judge form, with automatic timing and measurements already filled in.</li></ul></div></div>
<p class="notice">Next useful evidence: expert conclusions on presegmented movement clips, including acceptable form, technique concerns, uncertain views and non-squat transitions. Expert review of a development recording remains development evidence; a separate eligible target is needed for an independent transfer test.</p></section>
<section id="evidence"><h2>Execution and evidence</h2><p><strong>108 fits · 18 evaluated clinical configurations · 1,050 saved predictions.</strong> The healthy-only model is refitted for each held-out patient, so 108 fits do not represent 108 independent experiments.</p>
<div class="scroll"><table><tbody><tr><th>Prediction reconstruction</th><td>All saved scores reproduced from new model coefficients; maximum difference 4.44 × 10⁻¹⁶.</td></tr><tr><th>Training/test separation</th><td>Every test patient absent from training IDs and from training-derived scaling.</td></tr><tr><th>Confusion counts</th><td>Independently recounted for every saved configuration.</td></tr><tr><th>Optimization</th><td>Maximum gradient component 1.04 × 10⁻⁷.</td></tr><tr><th>ARED alignment</th><td>Candidate completion times match native source-frame indices.</td></tr><tr><th>Pose estimator</th><td>Sapiens2-5B, 1024 × 768 input, BF16 and flip averaging; existing completed poses reused.</td></tr></tbody></table></div>
<h3>Embedded downloadable artifacts</h3><p class="small">The report, plots, ARED replay and listed JSON data work offline in this single HTML file. Downloads below export the saved results, not a new run.</p>
<div class="downloads"><button data-download="clinical_results.json">Clinical results</button><button data-download="clinical_predictions.json">All predictions</button><button data-download="clinical_paired_cases.json">Paired cases</button><button data-download="squat_candidates.json">Cycle candidates</button><button data-download="protocol.json">Recorded protocol</button><button data-download="verification.json">Verification</button><button data-download="ared_visual_review.json">AI review notes</button></div>
<details><summary>Provenance and reproducibility</summary><p>Source artifacts: <code>results/local_execution_20261114/</code>. Runner: <code>scripts/run_local_experiments.py</code>; the exact executed version is retained as <code>results/local_execution_20261114/executed_runner.py</code>.</p>
<p>Executed runner SHA-256: <code>RUNNER_SHA</code><br>Recorded protocol SHA-256: <code>PROTOCOL_SHA</code></p>
<p>Original ARED RGB SHA-256: <code>f5a6355bb87ee367caa2c3ba9b3e7803f6f538e71c41627025ab42788b480fb1</code>. The embedded side-by-side replay is a derived visualization; original footage and poses remain preserved separately.</p>
<p>Video source: ESA/NASA, Sophie Adenot ARED / EasyMotion-2 catalog item. Recording context does not identify active stimulation, exercise resistance or individual prescription. Historical clinical features and labels retain their original limitations.</p></details></section>
<footer>Prepared for Hlib Plakhtii / Theia. This report includes model outputs and post-prediction observations; it is a research-results report, not an independent reference-label file.</footer></main>
<script type="application/json" id="reportData">REPORT_DATA</script>
<script>
'use strict';
(()=>{
const D=JSON.parse(document.getElementById('reportData').textContent),$=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>(100*x).toFixed(1)+'%',pp=x=>(x>=0?'+':'')+(100*x).toFixed(2)+' pp';
function table(){const selected=Number($('lambda').value);$('clinicalRows').innerHTML=D.results.summaries.filter(r=>r.lambda===selected).map(r=>{const a=r.methods.healthy_only,b=r.methods.healthy_plus_clinical;return `<tr><td><strong>${esc(D.names[r.dataset])}</strong></td><td class="num">${a.fp} → ${b.fp} / ${a.fp+a.tn}</td><td class="num">${a.fn} → ${b.fn} / ${a.tp+a.fn}</td><td class="num">${pct(a.macro_patient_accuracy)} → ${pct(b.macro_patient_accuracy)}</td><td class="num ${r.augmentation_minus_baseline_macro_accuracy>=0?'positive':'negative'}">${pp(r.augmentation_minus_baseline_macro_accuracy)}</td></tr>`}).join('')}
$('lambda').onchange=table;table();
$('allConfigs').innerHTML=D.results.summaries.map(r=>{const a=r.methods.healthy_only,b=r.methods.healthy_plus_clinical;return `<tr><td>${esc(D.names[r.dataset])}</td><td>${r.lambda}${r.primary?' · primary':''}</td><td>${a.fp} / ${a.fn}</td><td>${b.fp} / ${b.fn}</td><td>${pp(r.augmentation_minus_baseline_macro_accuracy)}</td><td>${r.paired_cluster_bootstrap_95_percentile.map(pp).join(' to ')}</td></tr>`}).join('');
function caseTable(){const rows=D.cases.filter(r=>($('caseFilter').value==='all'||r.effect===$('caseFilter').value)&&($('taskFilter').value==='all'||r.dataset===$('taskFilter').value));$('caseCount').textContent=rows.length+' feature-setting cases';$('caseRows').innerHTML=rows.map(r=>`<tr><td>${esc(D.names[r.dataset])}<br><small>${esc(r.id)}</small></td><td>${r.label}</td><td>${r.healthy_only}</td><td>${r.healthy_plus_clinical}</td><td><span class="badge ${r.effect==='helped'?'good':r.effect==='harmed'?'bad':''}">${r.effect==='unchanged'?'unchanged · '+(r.label===r.healthy_only?'correct':'incorrect'):r.effect}</span></td></tr>`).join('')}
$('caseFilter').onchange=caseTable;$('taskFilter').onchange=caseTable;caseTable();
const events=D.cycles[0].parameter_results.find(r=>r.primary).candidates,flagged=new Map(D.review.flagged_candidates.map(r=>[r.candidate,r.reason]));
$('eventRows').innerHTML=events.map((e,i)=>`<tr class="${flagged.has(i+1)?'flag':''}"><td><button data-candidate="${i+1}">#${i+1}</button></td><td class="num">${e.start_s.toFixed(2)} / ${e.bottom_s.toFixed(2)} / ${e.completion_s.toFixed(2)} s</td><td>${e.duration_s.toFixed(2)} s</td><td>${e.projected_excursion_deg.toFixed(1)}°</td><td>${esc(flagged.get(i+1)||'Plausible squat in sampled frames; unvalidated')}</td></tr>`).join('');
const video=$('replay'),canvas=$('timeline'),ctx=canvas.getContext('2d');let stopAt=null;
const left=45,right=1100,top=13,bottom=169,x=t=>left+(t-670)/109*(right-left),y=a=>bottom-a/190*(bottom-top);
function draw(){ctx.clearRect(0,0,1120,200);ctx.font='12px system-ui';ctx.fillStyle='#536b79';ctx.lineWidth=1;for(let a=0;a<=180;a+=45){ctx.strokeStyle='#dce5eb';ctx.beginPath();ctx.moveTo(left,y(a));ctx.lineTo(right,y(a));ctx.stroke();ctx.fillText(a+'°',5,y(a)+4)}for(let t=670;t<=779;t+=10)ctx.fillText(t+'s',x(t)-14,192);ctx.strokeStyle='#126e98';ctx.lineWidth=1.5;ctx.beginPath();let fresh=true;for(const [t,a] of D.signal){if(a===null){fresh=true;continue}fresh?ctx.moveTo(x(t),y(a)):ctx.lineTo(x(t),y(a));fresh=false}ctx.stroke();events.forEach((e,i)=>{ctx.strokeStyle=flagged.has(i+1)?'#d28a24':'#158670';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(e.completion_s),top);ctx.lineTo(x(e.completion_s),bottom);ctx.stroke()});ctx.strokeStyle='#172a36';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x(670+video.currentTime),top);ctx.lineTo(x(670+video.currentTime),bottom);ctx.stroke();$('clock').textContent='Source time '+(670+video.currentTime).toFixed(2)+' s';}
function playRange(start,end,label){stopAt=end;const go=()=>{video.currentTime=Math.max(0,start);video.play().catch(()=>{$('playing').textContent='Press the video play button to start.'});draw()};$('playing').textContent=label;if(video.readyState>=1)go();else video.addEventListener('loadedmetadata',go,{once:true})}
document.querySelectorAll('[data-candidate]').forEach(b=>b.onclick=()=>{const n=Number(b.dataset.candidate),e=events[n-1];playRange(Math.max(0,e.start_s-670-.35),Math.min(109,e.completion_s-670+.35),'Candidate #'+n+' · stops after the interval');video.scrollIntoView({behavior:'smooth',block:'center'})});
$('fullReplay').onclick=()=>playRange(0,null,'Full 109-second development segment');$('pause').onclick=()=>video.pause();$('speed').onchange=()=>video.playbackRate=Number($('speed').value);
video.addEventListener('timeupdate',()=>{if(stopAt!==null&&video.currentTime>=stopAt){video.pause();stopAt=null}draw()});video.addEventListener('seeked',draw);
video.addEventListener('error',()=>{$('playing').textContent='This browser could not play the embedded H.264 replay. The plots, timestamps and data remain available.'});
canvas.onclick=e=>{const r=canvas.getBoundingClientRect(),px=(e.clientX-r.left)*1120/r.width;stopAt=null;video.currentTime=Math.max(0,Math.min(108.96,(px-left)/(right-left)*109));draw()};draw();
document.querySelectorAll('[data-download]').forEach(b=>b.onclick=()=>{const name=b.dataset.download,value=D.downloads[name],url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
})();
</script></body></html>'''
    gymrows=[];gymcharts=[]
    for r in cycles[1:]:
        label=r['clip_id'].removeprefix('mendeley_')
        count=next(x['candidate_count'] for x in r['parameter_results'] if x['primary'])
        gymrows.append(f'<tr><td>{html.escape(label)}</td><td>{r["duration_s"]:.2f} s</td><td>{r["native_frames"]}</td><td>{count}</td><td>Pending</td></tr>')
        gymcharts.append(f'<h3>{html.escape(label)}</h3><img loading="lazy" class="figure" src="{media(RUN/(r["clip_id"]+"_cycles.png"),"image/png")}" alt="Candidate-cycle signal plot for {html.escape(label)}">')
    downloads={name:read(name) for name in ['clinical_results.json','clinical_predictions.json','clinical_paired_cases.json','squat_candidates.json','protocol.json','verification.json','ared_visual_review.json']}
    data={'results':datasets,'cycles':cycles,'cases':cases,'review':review,'names':names,'signal':signal,'downloads':downloads}
    substitutions={
      'BUILD_TIME':datetime.now(ZoneInfo('Europe/Kyiv')).strftime('%H:%M Europe/Kyiv'),
      'CLINICAL_CHART':media(RUN/'clinical_comparison.png','image/png'),
      'ARED_CHART':media(RUN/'ared_670_779_cycles.png','image/png'),
      'POSTER':media(ROOT/'data/poses/ared_continuous_development/replay_poster.png','image/png'),
      'VIDEO':media(ROOT/'data/poses/ared_continuous_development/ared_670_779_pose_review.mp4','video/mp4'),
      'CONTACT_SHEETS':''.join(f'<img class="figure" loading="lazy" src="{media(RUN/f"ared_candidate_review_{i}.jpg","image/jpeg")}" alt="Original RGB triplets for candidates {(i-1)*8+1} to {min(i*8,23)}">' for i in range(1,4)),
      'GYM_ROWS':''.join(gymrows),'GYM_CHARTS':''.join(gymcharts),
      'RUNNER_SHA':complete['script_sha256'],'PROTOCOL_SHA':complete['protocol_sha256'],
      'REPORT_DATA':json.dumps(data,separators=(',',':')).replace('<','\\u003c'),
    }
    for k,v in substitutions.items():body=body.replace(k,v)
    OUT.write_text(body)
    audit={'path':str(OUT),'bytes':OUT.stat().st_size,'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
           'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'source_run':str(RUN),'source_artifacts_verified_against_completed_run':True,
           'embedded_video':True,'embedded_charts':7,'embedded_contact_sheets':3,'embedded_downloads':list(downloads),
           'new_model_training_performed_for_report':False,'doctor_labels_received':False}
    (ROOT/'audit/results_report_20261114.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(OUT,OUT.stat().st_size,'bytes')


if __name__=='__main__':main()

"""Package completed local experiments without changing predictions or thresholds."""
import csv
import html
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from run_local_experiments import ROOT, OUT, sha, write


def main():
    results=json.loads((OUT/'clinical_results.json').read_text())['summaries']
    cycles=json.loads((OUT/'squat_candidates.json').read_text())
    cases=json.loads((OUT/'clinical_paired_cases.json').read_text())
    names={'CTK_arm_elevation':'Arm elevation (CTK)', 'ELK_pelvis_relative':'Side bending: pelvis-relative',
           'ELK_camera_relative':'Side bending: camera-relative'}
    flagged={11:'Turning and repositioning at the apparatus; not a clear squat cycle.',
             12:'Continued repositioning with shallow knee motion; not a clear squat cycle.',
             23:'Stepping/repositioning after the repeated squat sequence.'}
    review={'reviewed_at':datetime.now(timezone.utc).isoformat(), 'reviewer':'Codex',
            'status':'post_prediction_AI_visual_review_not_independent_ground_truth',
            'method':'Reviewed original RGB at start, minimum-angle and completion frames of all 23 primary candidates. Did not exhaustively label all frames or missed events.',
            'source_sha256':'f5a6355bb87ee367caa2c3ba9b3e7803f6f538e71c41627025ab42788b480fb1',
            'source_clock':'Original normalized source seconds = native source frame / 25; original PTS adds 0.044 s.',
            'predictions_unchanged':True,'plausible_squat_candidates':[i for i in range(1,24) if i not in flagged],
            'flagged_candidates':[{'candidate':i,'reason':reason} for i,reason in flagged.items()],
            'reference_event_precision_recall':None}
    write(OUT/'ared_visual_review.json',review)
    lines=['# Existing-data experiments — executed 2026-11-14','',
           'Fresh model training and ARED cycle extraction are complete. The new clinical comparison improves two feature settings and worsens the third. This is exploratory development evidence; the in-mission decision-transfer objective remains untested.','',
           '## Clinical comparison','',
           'Healthy-only training is compared with healthy plus clinical KERAAL records, holding each of six patients out completely. The test records are identical between arms. All preprocessing and class/participant weights use training records only. This is a different question from the historical comparison, whose baseline already included KERAAL patient records and whose extra source did not help. The old findings are preserved.','',
           'The run uses archived numeric features and annotation-derived labels, not saved predictions or freshly extracted clinical RGB poses. It contains 69 CTK records (17 positive / 52 negative) and 53 ELK records (40 / 13) from the same six patients. The two ELK geometries reuse the same 53 records. Healthy training is only 20 CTK records from four people and four ELK records from two people. A negative label means absence of this specific error code.','',
           'Primary configuration was recorded before fitting: weighted logistic regression, L2 penalty 0.1 and threshold 0.5; no outcome-based parameter selection.','',
           '| Task / features | False alerts, healthy → +clinical | Misses, healthy → +clinical | Mean patient accuracy | Helped / harmed / unchanged |',
           '|---|---:|---:|---:|---:|']
    table=[]
    for r in results:
        if not r['primary']:continue
        a,b=(r['methods'][k] for k in ['healthy_only','healthy_plus_clinical'])
        counts=Counter(c['effect'] for c in cases if c['dataset']==r['dataset'])
        row=[names[r['dataset']],f"{a['fp']} → {b['fp']}",f"{a['fn']} → {b['fn']}",
             f"{a['macro_patient_accuracy']:.1%} → {b['macro_patient_accuracy']:.1%}",
             f"{counts['helped']} / {counts['harmed']} / {counts['unchanged']}"]
        table.append(row);lines.append('| '+' | '.join(row)+' |')
    lines += ['', 'Both arm elevation and camera-relative side bending improve at all three preset regularization strengths. Pelvis-relative side bending worsens in mean patient accuracy at all three; its lower false-alert count comes with many more missed errors. All configurations are below.','',
              'The reported six-patient bootstrap intervals are descriptive only. Prior exposure, very small healthy samples, and increased training size in the augmented arm prevent a confirmatory or rehabilitation-specific causal claim. This does not establish improvement on astronauts, squats, or operational actions.','',
              '## All preset configurations','',
              '| Features | L2 penalty | Healthy FP / FN | +clinical FP / FN | Change in mean patient accuracy |',
              '|---|---:|---:|---:|---:|']
    for r in results:
        a,b=(r['methods'][k] for k in ['healthy_only','healthy_plus_clinical'])
        lines.append(f"| {names[r['dataset']]} | {r['lambda']} | {a['fp']} / {a['fn']} | {b['fp']} / {b['fn']} | {100*r['augmentation_minus_baseline_macro_accuracy']:+.2f} pp |")
    lines += ['', '## ARED execution','',
              'Processed all 2,725 native frames from source seconds 670–779 using the completed Sapiens2-5B poses. The fixed detector emitted 23 candidates at the primary 15° prominence, with 24 at 10° and 21 at 20°. These are projected 2D angle changes, not anatomical angles or safety thresholds. The score gate passes all ARED frames but does not establish visibility. The centered median filter and following-maximum endpoint make this an offline detector.','',
              'Post-prediction AI inspection of start/bottom/end RGB triplets found 20 candidates consistent with squats and three suspect transitions: #11 at 714.16–719.48 s, #12 at 719.48–721.28 s, and #23 at 770.88–772.48 s. All 23 outputs remain unchanged. This selected-frame review is not independent reference annotation, so precision, recall and missed-repetition counts are not reported. No advice to change load, technique or pace was generated.','',
              'The same detector also processed five eligible gym clips (2,343 frames). Their primary counts are 11, 20, 13, 17 and 8; these have not been independently verified and the cohort health status is undocumented. No clinical squat data was substituted.','',
              '## Reproducibility and checks','',
              '108 new fits, 18 clinical evaluation configurations and 1,050 saved out-of-patient predictions. Every prediction was independently reconstructed from saved model coefficients; maximum score difference was below 5×10⁻¹⁶. All confusion counts matched, optimizer gradients were below 1.1×10⁻⁷, held-out patient IDs were absent from training, and ARED candidate timestamps matched native frame indices. Hashes, parameters, source features, coefficients, all helped/harmed/unchanged cases and verification outputs are retained. No extra external data or messages were needed.','',
              'The 108 fits include refits of the identical healthy-only training set for each test patient; they are not 108 independent experiments. Training uses the CPU because these models have four or six features. Existing 5B pose inference is reused without recomputation.','',
              'Artifacts: [interactive review](../results/local_execution_20261114/index.html), [protocol](../results/local_execution_20261114/protocol.json), [results](../results/local_execution_20261114/clinical_results.json), [all predictions](../results/local_execution_20261114/clinical_predictions.json), [paired cases](../results/local_execution_20261114/clinical_paired_cases.json), [verification](../results/local_execution_20261114/verification.json), [ARED candidates](../results/local_execution_20261114/squat_candidates.json).','',
              'Re-run into a new directory to preserve this run: `.venv-pose/bin/python scripts/run_local_experiments.py prepare --output results/local_execution_repeat` followed by `.venv-pose/bin/python scripts/run_local_experiments.py run --output results/local_execution_repeat`.','']
    (ROOT/'reports/local_execution_20261114.md').write_text('\n'.join(lines))
    primary=cycles[0]['parameter_results'][0]['candidates']
    with open(OUT/'ared_candidates.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=['candidate','start_s','bottom_s','completion_s','completion_source_frame','post_prediction_review'])
        writer.writeheader()
        for i,e in enumerate(primary,1):
            writer.writerow({'candidate':i,**{k:e[k] for k in ['start_s','bottom_s','completion_s','completion_source_frame']},
                             'post_prediction_review':flagged.get(i,'Plausible squat in sampled RGB; unvalidated')})
    clinical_html=''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in table)
    event_html=''.join(f'''<tr class="{'flagged' if i in flagged else ''}"><td><button data-seek="{e['start_s']-670:.3f}">{i}</button></td>
        <td>{e['start_s']:.2f}–{e['completion_s']:.2f}</td><td>{html.escape(flagged.get(i,'Plausible squat in sampled RGB; unvalidated'))}</td></tr>''' for i,e in enumerate(primary,1))
    paired_html=''.join('<tr>'+''.join('<td>'+html.escape(str(c[k]))+'</td>' for k in ['dataset','id','label','healthy_only','healthy_plus_clinical','effect'])+'</tr>' for c in cases)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Theia — executed local experiments</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#18323d;background:#f4f7f8;max-width:1180px;margin:32px auto;padding:0 20px}
h1,h2{line-height:1.2}section{background:white;padding:24px;margin:24px 0;border:1px solid #dbe4e8;border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #dbe4e8;padding:9px;text-align:left}th{background:#edf3f5}
.scroll{overflow:auto}img,video{max-width:100%;height:auto}.flagged{background:#fff1dc}.note{padding:14px;background:#eaf1f4;border-left:4px solid #507a88}
button{cursor:pointer;padding:5px 13px;border:1px solid #6c929f;border-radius:5px;background:white;color:#185063}a{color:#05657f}canvas{width:100%;height:190px;cursor:crosshair;background:#f7fafb}.small{font-size:14px}details{margin:16px 0}
</style><h1>Theia: experiments executed</h1><p>2026-11-14 · Fresh fits on available clinical features and cycle extraction from in-mission RGB poses.</p>
<p class="note"><strong>Exploratory results.</strong> Clinical augmentation helped two Earth-data feature settings and harmed one. The ARED output is a cycle detector with visible transition errors. In-mission decision benefit remains untested.</p>
<section><h2>Clinical training comparison</h2><p>Healthy-only training versus healthy plus other patients; each of six test patients held out completely. Fixed penalty 0.1 and threshold 0.5. This differs from the old experiment, whose baseline already included patient data. Archived features were used; saved historical predictions were discarded.</p>
<div class="scroll"><table><thead><tr><th>Features</th><th>False alerts</th><th>Misses</th><th>Mean patient accuracy</th><th>Helped / harmed / unchanged</th></tr></thead><tbody>'''+clinical_html+'''</tbody></table></div>
<p class="small">CTK: 69 patient records; ELK: 53 patient records reused for both geometries. Only 20 healthy CTK and four healthy ELK training records. Known prior data exposure and unequal training sizes limit interpretation.</p>
<img src="clinical_comparison.png" alt="False alerts and missed errors for both freshly fitted training arms">
<p><a href="clinical_results.json">All three preset penalties and descriptive uncertainty</a> · <a href="verification.json">Verification</a> · <a href="protocol.json">Protocol</a></p>
<details><summary>All 175 paired feature-setting cases (ELK records appear twice)</summary><div class="scroll"><table><thead><tr><th>Features</th><th>Record</th><th>Label</th><th>Healthy only</th><th>+clinical</th><th>Effect</th></tr></thead><tbody>'''+paired_html+'''</tbody></table></div></details></section>
<section><h2>ARED: original video, poses and cycle candidates</h2><p>109 seconds / 2,725 native frames. The detector found 23 candidates; sampled original RGB shows 20 plausible squats and three suspect transitions. Orange marks the three flagged candidates. This review is not independent ground truth; no accuracy or clinical action is claimed.</p>
<video id="replay" controls preload="metadata" poster="../../data/poses/ared_continuous_development/replay_poster.png" src="../../data/poses/ared_continuous_development/ared_670_779_pose_review.mp4"></video>
<p id="clock">Original source time: 670.00 s</p><canvas id="timeline" width="1120" height="190" aria-label="Projected knee-angle timeline; click to seek"></canvas>
<p class="small">Blue: filtered projected knee angle. Green: candidate completions. Orange: suspect transitions. Gray cursor follows replay. Click the timeline or a numbered candidate to seek. Existing replay has original RGB beside estimated pose.</p>
<details><summary>All candidate timestamps and review findings</summary><table><thead><tr><th>Seek</th><th>Original seconds</th><th>Post-prediction review</th></tr></thead><tbody>'''+event_html+'''</tbody></table></details>
<p><a href="ared_candidates.csv">Candidate CSV</a> · <a href="ared_670_779_signals.csv">Frame-level angles</a> · <a href="squat_candidates.json">All sensitivity outputs and gym clips</a> · <a href="ared_visual_review.json">Review provenance</a></p>
<details><summary>Original RGB evidence for every candidate</summary>'''+''.join(f'<img loading="lazy" src="ared_candidate_review_{i}.jpg" alt="Original RGB triplets for candidate group {i}">' for i in range(1,4))+'''</details></section>
<p><a href="../../reports/local_execution_20261114.md">Full report and reproduction commands</a> · <a href="clinical_models.json">New model coefficients</a> · <a href="clinical_predictions.json">All 1,050 predictions</a></p>
<script>
(()=>{
const events=EVENTS, signal=SIGNAL;
const video=document.getElementById('replay'), canvas=document.getElementById('timeline'),ctx=canvas.getContext('2d');
const left=44,right=1100,top=12,bottom=160;
const x=t=>left+(t-670)/109*(right-left),y=a=>bottom-a/190*(bottom-top);
function draw(){ctx.clearRect(0,0,1120,190);ctx.font='11px system-ui';ctx.fillStyle='#385765';
for(let a=0;a<=180;a+=45){ctx.strokeStyle='#dce5e9';ctx.beginPath();ctx.moveTo(left,y(a));ctx.lineTo(right,y(a));ctx.stroke();ctx.fillText(a+'°',6,y(a)+4)}
for(let t=670;t<=779;t+=10)ctx.fillText(t+'s',x(t)-12,182);
ctx.strokeStyle='#116c82';ctx.lineWidth=1.4;ctx.beginPath();let fresh=true;
for(const [t,a] of signal){if(a===null){fresh=true;continue}if(fresh)ctx.moveTo(x(t),y(a));else ctx.lineTo(x(t),y(a));fresh=false}ctx.stroke();
events.forEach((e,i)=>{ctx.strokeStyle=[11,12,23].includes(i+1)?'#d97e16':'#33976c';ctx.beginPath();ctx.moveTo(x(e.completion_s),top);ctx.lineTo(x(e.completion_s),bottom);ctx.stroke()});
ctx.strokeStyle='#343b42';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x(670+video.currentTime),top);ctx.lineTo(x(670+video.currentTime),bottom);ctx.stroke();
document.getElementById('clock').textContent='Original source time: '+(670+video.currentTime).toFixed(2)+' s';}
video.addEventListener('timeupdate',draw);video.addEventListener('seeked',draw);
document.querySelectorAll('[data-seek]').forEach(b=>b.addEventListener('click',()=>{video.currentTime=Number(b.dataset.seek);draw()}));
canvas.addEventListener('click',e=>{const r=canvas.getBoundingClientRect(),px=(e.clientX-r.left)*1120/r.width;video.currentTime=Math.max(0,Math.min(108.96,(px-left)/(right-left)*109));draw()});draw();
})();
</script></html>'''
    signal=[]
    for r in csv.DictReader(open(OUT/'ared_670_779_signals.csv')):
        signal.append([float(r['source_time_s']),float(r['median_filtered_deg']) if r['median_filtered_deg'] else None])
    page=page.replace('EVENTS',json.dumps(primary)).replace('SIGNAL',json.dumps(signal))
    (OUT/'index.html').write_text(page)
    write(OUT/'package_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),
          'report_sha256':sha(ROOT/'reports/local_execution_20261114.md'),
          'files':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='package_manifest.json'}})
    print(OUT/'index.html')


if __name__=='__main__':main()

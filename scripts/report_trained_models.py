"""Build a local results report, plots and portable trained-model bundle."""
import html
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from train_existing_data import OUT, ROOT, SEEDS, sha, write


def percent(x):
    return f'{100*x:.1f}%'


def video_path(cid):
    if cid.startswith('PG_'):
        _, subset, name = cid.split('_', 2)
        return ROOT / 'data/raw/ProGait/videos' / subset / (name + '.mp4')
    if cid.startswith('mendeley_'):
        return ROOT / 'data/raw/fitness_multiview_squat' / (cid.removeprefix('mendeley_') + '.mp4')
    return None


def build():
    assert (OUT / 'complete.json').exists()
    r = json.loads((OUT / 'results.json').read_text())
    c = json.loads((OUT / 'clinical_results.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    manifest = json.loads((OUT / 'data_manifest.json').read_text())
    preds = json.loads((OUT / 'predictions_ensemble.json').read_text())
    baseline = json.loads((OUT / 'baseline_predictions.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    labels = {'majority': 'Majority', 'feature_logistic': 'Pose features + logistic', 'without_rehab': 'TCN, no ProGait', 'with_rehab': 'TCN + ProGait'}
    for ax, task, title in zip(axes, ['gait', 'squat'], ['ProGait • 401 videos / 4 people', 'Staged squats • 5 videos / 3 people']):
        methods = ['majority', 'feature_logistic'] + (['without_rehab'] if task == 'squat' else []) + ['with_rehab']
        values = [r[task][m]['balanced_accuracy'] for m in methods]
        bars = ax.barh(np.arange(len(methods)), values, color=['#9ba6b7', '#78879f', '#5790c6', '#21a58c'][-len(methods):])
        ax.set_yticks(np.arange(len(methods)), [labels[m] for m in methods])
        ax.invert_yaxis(); ax.set_xlim(0, 1.08)
        ax.set_xticks(np.linspace(0, 1, 6), [f'{x:.0%}' for x in np.linspace(0, 1, 6)])
        ax.set_title(title, loc='left', weight='bold'); ax.set_xlabel('Held-out-person balanced accuracy')
        ax.bar_label(bars, labels=[percent(v) for v in values], padding=5)
    fig.savefig(OUT / 'held_out_results.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    cm = np.array(r['gait']['with_rehab']['confusion_matrix'])
    image = ax.imshow(cm, cmap='Blues')
    for i in range(9):
        for j in range(9):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() * .55 else '#172536')
    ax.set_xticks(range(9), range(1, 10)); ax.set_yticks(range(9), range(1, 10))
    ax.set_xlabel('Predicted publisher CCC category'); ax.set_ylabel('Publisher reference CCC category')
    ax.set_title('ProGait • held-out-person confusion counts')
    fig.colorbar(image, ax=ax, shrink=.8)
    fig.savefig(OUT / 'gait_confusion.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    for seed in SEEDS:
        h = json.loads((OUT / 'fits' / f'final_all_data_seed{seed}' / 'history.json').read_text())
        s = [row for row in h if row['phase'] == 'supervised']
        ax.plot([row['step'] for row in s], [row['mean_loss'] for row in s], marker='o', label=str(seed))
    ax.set_xlabel('Supervised optimizer step'); ax.set_ylabel('Training objective'); ax.set_title('Final all-data fits • loss is not test accuracy', loc='left')
    ax.legend(title='Seed'); fig.savefig(OUT / 'training_loss.png', dpi=180); plt.close(fig)

    rows = []
    for p in preds:
        if p['scope'] == 'held-out-person':
            rows.append(dict(**p, method=p['arm']))
    for p in baseline:
        rows.append(dict(**p, scope='held-out-person'))
    for p in rows:
        path = video_path(p['id'])
        assert path and path.exists(), p['id']
        p['video'] = os.path.relpath(path, OUT)
    write(OUT / 'report_data.json', rows)
    title = 'Theia · trained models and measured results'
    summary = r['coverage']
    table = ''
    for task in ['gait', 'squat']:
        for method in ['majority', 'feature_logistic', 'without_rehab', 'with_rehab']:
            if method not in r[task]:
                continue
            v = r[task][method]
            table += f'<tr><td>{task}</td><td>{labels[method]}</td><td>{v["n"]}</td><td>{percent(v["accuracy"])}</td><td>{percent(v["balanced_accuracy"])}</td><td>{percent(v["macro_f1"])}</td></tr>'
    paired = ''
    for q in r['squat_transfer']['paired_cases']:
        name = q['id'].removeprefix('mendeley_')
        paired += f'<tr><td>{name}</td><td>{["good", "bad"][q["y"]]}</td><td>{["good", "bad"][q["without_rehab"]]}</td><td>{["good", "bad"][q["with_rehab"]]}</td><td>{q["effect"].replace("_", " ")}</td></tr>'
    clinical = ''
    for name, vals in c.items():
        a, b = vals['healthy_only'], vals['healthy_plus_clinical']
        clinical += f'<tr><td>{name}</td><td>{a["n"]}</td><td>{percent(a["macro_patient_accuracy"])}</td><td>{percent(b["macro_patient_accuracy"])}</td><td>{a["fp"]} / {a["fn"]} → {b["fp"]} / {b["fn"]}</td></tr>'
    checkpoints = ''.join(f'<li><a href="fits/final_all_data_seed{s}/model.pt">Final model · seed {s}</a></li>' for s in SEEDS)
    main = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>:root{{color-scheme:light;font:16px/1.55 system-ui;color:#182b3d;background:#eff3f7}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1180px;margin:auto;padding:32px 24px}}h1{{font-size:34px;line-height:1.15;letter-spacing:-.7px}}h2{{margin-top:0;font-size:23px}}p{{max-width:90ch}}a{{color:#075ca3}}.eyebrow{{text-transform:uppercase;letter-spacing:.15em;font-size:12px;color:#2a6a77;font-weight:700}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card,section{{background:white;border:1px solid #d8e1e9;border-radius:12px;padding:22px;margin:18px 0}}.card{{margin:0}}.card strong{{display:block;font-size:29px;color:#145a64}}.note{{border-left:4px solid #b57416;background:#fff6e7;padding:16px 20px;border-radius:5px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px 9px;border-bottom:1px solid #e1e8ef;text-align:left}}th{{background:#f3f6fa}}img{{max-width:100%;height:auto}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}video{{width:100%;max-height:480px;background:#101820;border-radius:8px}}label{{display:inline-flex;gap:6px;align-items:center;margin:4px 12px 12px 0}}select,input,button{{font:inherit;padding:7px;border:1px solid #bac9d8;border-radius:5px}}button{{cursor:pointer;background:#edf5fa}}#list{{max-height:370px;overflow:auto}}#list button{{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid #dde5ed;border-radius:0;padding:10px;background:white;font-size:13px}}#list button:hover,#list button.selected{{background:#dff0f3}}.small{{font-size:13px;color:#52677a}}code{{font-size:13px}}@media(max-width:800px){{.cards,.grid{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}main{{padding:20px 12px}}h1{{font-size:28px}}}}</style>
<main><div class="eyebrow">Local research run · 14 November 2026 · completed</div><h1>{title}</h1>
<p>A shared temporal pose model was trained on all accepted local pose sequences, with separate gait-category and staged-squat heads. Sapiens2 and the publisher pose annotations remain fixed inputs.</p>
<div class="cards"><div class="card"><strong>{summary['pose_sources']}</strong>accepted pose sequences</div><div class="card"><strong>{summary['native_pose_frames']:,}</strong>native input frames</div><div class="card"><strong>30 + 3</strong>held-out fits + final fits</div><div class="card"><strong>0</strong>independent ARED verdicts</div></div>
<p class="note"><b>Mission result remains unmeasured.</b> No doctor conclusions have been returned. Unlabeled ARED poses contributed to reconstruction training; the model abstains from operational form or load recommendations. The current results do not establish rehabilitation benefit for astronauts.</p>
<p><b>What the run found:</b> squat classification was correct on 4 of 5 held-out source videos with and without rehabilitation data, so adding ProGait did not improve this test. The gait classifier failed to generalize: {percent(r['gait']['with_rehab']['accuracy'])} accuracy, below the held-out majority baseline of {percent(r['gait']['majority']['accuracy'])}. These are exploratory results from very few people.</p>
<section><h2>Held-out-person results</h2><p>All views, trials, windows and unlabeled recordings from each test person were excluded from fitting. Predictions average windows per original video and then three seeds. The 1,190 doctor-review cuts are not counted as independent training examples.</p><img src="held_out_results.png" alt="Held-out-person balanced accuracy for each model"><div class="scroll"><table><thead><tr><th>Task</th><th>Model</th><th>Videos</th><th>Accuracy</th><th>Balanced accuracy</th><th>Macro F1</th></tr></thead><tbody>{table}</tbody></table></div><p class="small">Four ProGait participants and three squat participants. Gym labels describe publisher-staged form; they are not clinical verdicts. No hyperparameters were selected using held-out results. Per-person and per-seed metrics are in <a href="results.json">results.json</a>.</p></section>
<section><h2>Did rehabilitation data help the squat model?</h2><p>Both arms used the same held-out people, architecture, 300 reconstruction updates and 600 supervised updates. The added-data arm also learned ProGait's separate gait task. This is an exploratory terrestrial transfer comparison.</p><div class="scroll"><table><thead><tr><th>Original recording</th><th>Publisher label</th><th>Without ProGait</th><th>With ProGait</th><th>Change</th></tr></thead><tbody>{paired}</tbody></table></div><p class="small">Only five source videos: one changed prediction moves pooled accuracy by 20 percentage points. Three seeds measure optimization sensitivity, not additional participants.</p></section>
<section><h2>Inspect every held-out prediction</h2><p>Original RGB and the source-level reference. This results viewer is separate from the doctor's blind labeling page.</p><label>Task <select id="task"><option value="gait">ProGait gait category</option><option value="squat">Staged squat form</option></select></label><label>Model <select id="method"><option value="with_rehab">TCN + ProGait</option><option value="without_rehab">TCN, no ProGait</option><option value="feature_logistic">Pose features + logistic</option><option value="majority">Majority baseline</option></select></label><label><input id="wrong" type="checkbox">Errors only</label><input id="search" aria-label="Search video or participant" placeholder="Search video / person"><p id="count" class="small"></p><div class="grid"><div id="list"></div><div><video id="video" controls preload="metadata"></video><h3 id="case-title"></h3><p id="case-detail"></p></div></div></section>
<section class="grid"><div><h2>Gait category confusion</h2><img src="gait_confusion.png" alt="Nine-category gait confusion matrix"><p class="small">CCC numbers are publisher gait categories. Category 9, described as normal prosthetic gait, does not represent a healthy squat control.</p></div><div><h2>Training completed</h2><img src="training_loss.png" alt="Final model training losses"><p>260,089 trainable parameters; three independently initialized final checkpoints. Each model saw every eligible training window during reconstruction and every labeled source during supervised fitting.</p><p class="small">Publisher RTMW-assisted poses and Sapiens2 poses are different measurement sources. Their joint names were mapped explicitly to the same 23-joint order; confidence scores became binary validity masks.</p></div></section>
<section><h2>Clinical feature models, kept separate</h2><p>New all-data checkpoints and repeated held-out-patient fits use the archived KERAAL features. These reproduce previously explored results; they are not new independent clinical evidence. The two ELK variants share the same 53 patient examples.</p><div class="scroll"><table><thead><tr><th>Task / geometry</th><th>Patient records</th><th>Healthy-only mean patient accuracy</th><th>With other patients</th><th>False positives / misses</th></tr></thead><tbody>{clinical}</tbody></table></div></section>
<section><h2>Files and reproducibility</h2><ul>{checkpoints}<li><a href="protocol.json">Frozen protocol and exact participant splits</a></li><li><a href="data_manifest.json">Every input, checksum and exclusion</a></li><li><a href="predictions_ensemble.json">All ensemble predictions, including explicitly unvalidated mission scores</a></li><li><a href="clinical_models.json">Clinical feature model coefficients</a></li><li><a href="verification.json">Leakage, checkpoint and coverage verification</a></li><li><a href="../../deliverables/Theia_Trained_Models_2026-11-14.zip">Portable model package</a></li><li><a href="../../deliverables/Theia_All_Videos_Review/index.html">Doctor's separate labeling page</a></li></ul><p class="small">ProGait source: <a href="https://huggingface.co/datasets/ericyxy98/ProGait">official dataset</a>, revision beb72713ab6c2ad6dcc460b4e726f040ad536c82, CC BY-NC-SA 4.0. One pose/RGB length mismatch and six publisher-marked non-analysis recordings were excluded; four blank-reference recordings were used only without labels. Existing privacy/quality holds and duplicate pilot frames were excluded.</p></section></main>
<script>const DATA=__DATA__;
const el=id=>document.getElementById(id),label=(task,x)=>task==='gait'?'CCC '+(x+1):['publisher good','publisher bad'][x];
function show(r,button){{document.querySelectorAll('#list .selected').forEach(b=>b.classList.remove('selected'));button.classList.add('selected');el('video').src=r.video;el('case-title').textContent=r.id;let score=r.probabilities?Math.max(...r.probabilities):null;el('case-detail').textContent='Held out: '+r.subject+' · Reference: '+label(r.task,r.y)+' · Prediction: '+label(r.task,r.predicted)+(score===null?'':' · Uncalibrated score: '+score.toFixed(3))+(r.seed_predictions?' · Seed votes: '+r.seed_predictions.map(x=>label(r.task,x)).join(', '):'');}}
function render(){{let q=el('search').value.toLowerCase(),a=DATA.filter(r=>r.task===el('task').value&&r.method===el('method').value&&(!el('wrong').checked||r.y!==r.predicted)&&(r.id+' '+r.subject).toLowerCase().includes(q));el('list').replaceChildren();el('count').textContent=a.length+' source videos';a.forEach((r,i)=>{{let b=document.createElement('button');b.textContent=(r.y===r.predicted?'✓ ':'× ')+r.id;b.onclick=()=>show(r,b);el('list').append(b);if(i===0)show(r,b)}});if(!a.length){{el('video').removeAttribute('src');el('video').load();el('case-title').textContent='No matching cases';el('case-detail').textContent='';}}}}
['task','method','wrong','search'].forEach(id=>el(id).addEventListener('input',render));render();</script></html>'''
    main = main.replace('__DATA__', json.dumps(rows).replace('</', '<\\/'))
    (OUT / 'index.html').write_text(main)

    # Audit actual saved training IDs rather than only trusting split descriptions.
    import torch
    checked = []
    records = manifest['records']
    for job in protocol['jobs']:
        train_ids = {records[i]['id'] for i in job['train_source_indices']}
        test_ids = {records[i]['id'] for i in job['test_source_indices']}
        assert train_ids.isdisjoint(test_ids)
        assert {records[i]['subject'] for i in job['train_source_indices']}.isdisjoint({records[i]['subject'] for i in job['test_source_indices']})
        for seed in SEEDS:
            p = OUT / 'fits' / f'{job["id"]}_seed{seed}'
            complete = json.loads((p / 'complete.json').read_text())
            state = torch.load(p / 'model.pt', weights_only=True)
            assert set(state['training_source_ids']) == train_ids
            assert sha(p / 'model.pt') == complete['checkpoint_sha256']
            assert state['protocol_sha256'] == sha(OUT / 'protocol.json')
            assert complete['checkpoint_reload_identical'] and complete['all_supervised_sources_seen']
            checked.append(p.name)
    held = [p for p in preds if p['scope'] == 'held-out-person']
    assert len(held) == 401 + 5 * 2
    assert len({(p['id'], p['task'], p['arm']) for p in held}) == len(held)
    assert len(rows) == 401 * 3 + 5 * 4
    assert r['mission']['decision_accuracy'] is None
    final_aux = [p for p in preds if p['scope'] != 'held-out-person' and p['y'] is None and p['task'] == 'squat']
    assert len(final_aux) == 7 and all(p['decision'] == 'abstain_pending_independent_reference' for p in final_aux)
    write(OUT / 'verification.json', dict(passed=True, checked_checkpoints=len(checked),
        participant_and_source_disjoint_folds=True, checkpoint_hashes_match=True, checkpoint_reload_logits_identical=True,
        held_out_ensemble_rows=len(held), unique_held_out_rows=True, report_video_paths_exist=True,
        no_invented_mission_accuracy=True, mission_or_ground_unlabeled_abstentions=len(final_aux),
        all_input_hashes_verified_before_training=True, notes=['Numeric baselines emitted missing-feature warnings for fully absent joints; missing summaries were explicitly filled with zero and validity fractions included.',
        'Each final-fit model is trained on all accepted inputs and is not a test checkpoint.']))

    bundle = ROOT / 'deliverables/Theia_Trained_Models_2026-11-14'
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ['results.json', 'clinical_models.json', 'clinical_results.json', 'baseline_models.json', 'protocol.json', 'data_manifest.json', 'verification.json', 'held_out_results.png', 'gait_confusion.png', 'training_loss.png']:
        shutil.copy2(OUT / name, bundle / name)
    for seed in SEEDS:
        shutil.copy2(OUT / 'fits' / f'final_all_data_seed{seed}' / 'model.pt', bundle / f'pose_tcn_seed{seed}.pt')
    shutil.copy2(OUT / 'executed_runner.py', bundle / 'train_existing_data.py')
    shutil.copy2(ROOT / 'scripts/predict_trained_pose.py', bundle / 'predict_trained_pose.py')
    (bundle / 'START_HERE.txt').write_text(f'''THEIA TRAINED RESEARCH MODELS — 2026-11-14

Completed: 33 temporal model fits (30 held-out-person evaluations + 3 final fits).
Final inputs: 417 accepted pose sequences, 200,229 native frames, 3,175 windows.
Two Gerst sequences come from the same recording and participant.
Training labels: 401 ProGait gait categories, 5 staged squat labels; 11 other
recordings have no task labels. Archived clinical feature models remain separate.

Held-out balanced accuracy:
  ProGait TCN: {percent(r['gait']['with_rehab']['balanced_accuracy'])}
  Squat without ProGait: {percent(r['squat']['without_rehab']['balanced_accuracy'])}
  Squat with ProGait: {percent(r['squat']['with_rehab']['balanced_accuracy'])}
See results.json for all folds, seed results, baselines and failures.
Do not interpret these as ARED form accuracy: no independent mission labels exist.

The three .pt files are the FINAL all-data models. Average their probabilities.
The model predicts separate publisher gait-category and staged-squat tasks.
Sapiens2 is the fixed pose extractor and is not bundled or retrained here.
Input: named 23-joint Sapiens pose CSV, or the publisher's 23-joint .npy.gz.
Normalization and joint order are recorded in protocol.json.

Run from this folder with the existing workspace Python environment:
  /home/hlak/Documents/NasaSpaceApp/.venv-pose/bin/python predict_trained_pose.py \\
    --input /absolute/path/to/poses.csv --task squat \\
    --checkpoint pose_tcn_seed20261114.pt \\
    --checkpoint pose_tcn_seed20261115.pt \\
    --checkpoint pose_tcn_seed20261116.pt \\
    --output /absolute/path/to/prediction.json

Requirements: Python {protocol['environment']['python']}, torch {protocol['environment']['torch']}, numpy {protocol['environment']['numpy']}, scipy (use the recorded
environment versions from protocol.json; no downloads needed in this workspace).
Inference supports CUDA or CPU. Run only trusted checkpoint files.
For publisher NPY input, also provide --width and --height for the original RGB.
Scores are uncalibrated research outputs; inference always abstains from an
operational recommendation. Whole-record preprocessing and temporal windows
are offline, not a validated real-time guidance system.

Dataset provenance / restrictions:
ProGait, Yin et al., ICCV 2025, CC BY-NC-SA 4.0, revision
beb72713ab6c2ad6dcc460b4e726f040ad536c82.
https://huggingface.co/datasets/ericyxy98/ProGait
No commercial or unrestricted reuse claim is made for these trained artifacts.
Source videos are not included. Preserve upstream terms and attribution.

Research limits:
- 4 gait participants and 3 staged-squat participants; cuts/views are correlated.
- All data previously screened for development; no confirmatory clinical trial.
- KERAAL models use archived features; ELK alternatives reuse the same records.
- Unlabeled mission recordings participated in reconstruction training. Future
  labels on those recordings are transductive evaluation, not unseen missions.
- No doctor conclusions are modified or invented. Keep the doctor's review
  blind to these predictions until independent annotations are exported.
''')
    entries = {str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    write(bundle / 'manifest.json', dict(files=entries))
    archive = bundle.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                z.write(p, Path(bundle.name) / p.relative_to(bundle))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    write(OUT / 'delivery.json', dict(report=str(OUT / 'index.html'), bundle=str(bundle), archive=str(archive),
                                     archive_bytes=archive.stat().st_size, archive_sha256=sha(archive)))
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    build()

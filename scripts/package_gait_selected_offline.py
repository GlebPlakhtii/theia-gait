"""Build a file:// compatible, offline copy of the selected gait review page."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import os
from datetime import datetime, timezone
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/gait_selected_paper_20261115'
SERVER_ROOT = ROOT / 'deliverables/Theia_Squat_Form_Lab_2026-11-14'
OUT = ROOT / os.environ.get('THEIA_OFFLINE_OUTPUT', 'deliverables/Theia_Gait_Demo_Offline_2026-11-15')
OUT.mkdir(parents=True, exist_ok=True)

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

data = json.loads((SOURCE / 'page_data.json').read_text())
poses, media_provenance = {}, {}
for clip in data['clips']:
    cid = clip['clip_id']
    video_source = (SERVER_ROOT / urlsplit(clip['video_url']).path.lstrip('/')).resolve(strict=True)
    pose_source = (SERVER_ROOT / urlsplit(clip['pose_url']).path.lstrip('/')).resolve(strict=True)
    video_path = f'media/{cid}.mp4'
    pose_path = f'poses/{cid}.json'
    if cid not in poses:
        copy(video_source, OUT / video_path)
        copy(pose_source, OUT / pose_path)
        poses[cid] = json.loads(pose_source.read_text())
        media_provenance[cid] = dict(
            video=video_path, pose=pose_path,
            original_video_route=clip['video_url'], original_pose_route=clip['pose_url'],
            video_sha256=sha(video_source), pose_sha256=sha(pose_source),
            source_first_frame=clip['video']['source_first_frame'], fps=clip['video']['fps'],
        )
    clip['video_url'] = video_path + '?v=' + sha(video_source)[:12]
    clip['pose_url'] = pose_path
    copy(SOURCE / clip['poster'], OUT / clip['poster'])
write_json(OUT / 'page_data.json', data)
write_json(OUT / 'media_provenance.json', media_provenance)

# Classic scripts can load from file://; fetch() and ES-module loading cannot
# reliably do so. Bundle the existing renderer and preload exact JSON records.
anatomy = json.loads((SOURCE / 'anatomy_assets/anatomy.json').read_text())
offline = dict(report=data, poses=poses, anatomy=anatomy)
(OUT / 'offline-data.js').write_text('window.THEIA_OFFLINE_DATA=' + json.dumps(offline, ensure_ascii=False, separators=(',', ':')) + ';\n')

app = (SOURCE / 'app.js').read_text()
old = "const res=await fetch(c.pose_url,{cache:'no-store'});if(!res.ok)throw Error('Pose HTTP '+res.status);cache.set(cid,await res.json());"
new = "const saved=window.THEIA_OFFLINE_DATA.poses[c.clip_id];if(!saved)throw Error('Missing bundled pose '+c.clip_id);cache.set(cid,saved);"
assert old in app
app = app.replace(old, new)
old = "const res=await fetch('page_data.json',{cache:'no-store'});if(!res.ok)throw Error('Data '+res.status);D=await res.json();"
assert old in app
app = app.replace(old, 'D=window.THEIA_OFFLINE_DATA.report;')
assert 'fetch(' not in app
(OUT / 'app.js').write_text(app)

renderer = (SOURCE / 'anatomy.js').read_text()
renderer = renderer.replace("await fetch('anatomy_assets/anatomy.json').then(r=>r.json())", 'window.THEIA_OFFLINE_DATA.anatomy')
assert 'fetch(' not in renderer
(OUT / 'anatomy-source.js').write_text(renderer)
for name in ['three.module.js', 'three.core.js', 'OrbitControls.js', 'LICENSE']:
    copy(SOURCE / 'anatomy_assets/vendor' / name, OUT / 'anatomy_assets/vendor' / name)
for name in ['anatomy.json', 'credits.txt', 'MODEL_README.txt', 'GEOMETRY_README.txt']:
    copy(SOURCE / 'anatomy_assets' / name, OUT / 'anatomy_assets' / name)
subprocess.run([
    'npm', 'exec', '--yes', '--package=esbuild@0.25.12', '--', 'esbuild',
    str(OUT / 'anatomy-source.js'), '--bundle', '--format=iife', '--target=es2020',
    '--legal-comments=eof', '--minify', '--outfile=' + str(OUT / 'anatomy.bundle.js'),
], check=True)

html = (SOURCE / 'index.html').read_text()
html = html.replace('src="app.js?v=pooling-20261115"', 'src="app.js"')
html = html.replace('href="/gait_experiments/"', 'href="#analysis"')
html = html.replace('Full experiment report ↗', 'Analysis ↓').replace('All experiments ↗', 'Evaluation ↓')
html = html.replace('href="/gait_distribution/"', 'href="#demo"').replace('Full distribution demo ↗', 'Video lab ↑')
html = html.replace('POSE-BASED GAIT REVIEW · RESEARCH STUDY', 'POSE-BASED GAIT REVIEW · OFFLINE EDITION')
html = html.replace('<script src="app.js"></script><script type="module" src="anatomy.js"></script>', '<script src="offline-data.js"></script><script src="app.js"></script><script src="anatomy.bundle.js"></script>')
assert 'type="module"' not in html and 'href="/gait_' not in html
(OUT / 'index.html').write_text(html)
(OUT / 'START_DEMO.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=index.html#demo"><title>Open Theia gait demo</title><p><a href="index.html#demo">Open the offline Theia video demo</a></p></html>''')
(OUT / 'START_SPACE_RUNNING.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=index.html?collection=astronaut#demo"><title>ISS treadmill running</title><p><a href="index.html?collection=astronaut#demo">Open ISS treadmill-running examples</a></p></html>''')
(OUT / 'START_SIMULATED_ALERT.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=index.html?collection=astronaut&amp;simulation=left-knee#demo"><title>Simulated knee alert</title><p><a href="index.html?collection=astronaut&amp;simulation=left-knee#demo">Open the explicitly simulated left-knee alert</a></p></html>''')
copy(SOURCE / 'style.css', OUT / 'style.css')
copy(SOURCE / 'paper-results.js', OUT / 'paper-results.js')
copy(SOURCE / 'paper/evidence/pooling_table2/results.json', OUT / 'paper/evidence/pooling_table2/results.json')
copy(SOURCE / 'angles.js', OUT / 'angles.js')
copy(SOURCE / 'landmarks.js', OUT / 'landmarks.js')
copy(SOURCE / 'simulation.js', OUT / 'simulation.js')
copy(SOURCE / 'demo_scenario.json', OUT / 'demo_scenario.json')
for directory in ['figures', 'references', 'fonts']:
    for src in (SOURCE / directory).iterdir():
        if src.is_file(): copy(src, OUT / directory / src.name)
for name in ['protocol.json', 'report_results.json', 'window_results.json', 'inference_verification.json', 'bibliography.json', 'paired_case_provenance.json', 'paired_case_capture.json']:
    copy(SOURCE / name, OUT / name)
if data.get('active_model'):
    for name in ['pooling_window_results.json','pooling_model.json','pooling_demo_verification.json','pooling_predictions.npz']:
        copy(SOURCE/name,OUT/name)
    for member in data['active_model']['checkpoints']:
        copy(ROOT/'results/gait_improvement_refinement_20261115/runs'/member['member']/'best.pt',OUT/'model/checkpoints'/(member['member']+'.pt'))
    for name in ['gait_statistical_pool_model.py','gait_kinematic_model.py','gait_from_scratch_model.py']:
        copy(ROOT/'scripts'/name,OUT/'model/code'/name)
copy(SOURCE / 'astronaut_treadmill_provenance.json', OUT / 'astronaut_treadmill_provenance.json')
for clip in data['clips']:
    if clip['kind'] == 'astronaut':
        copy(SOURCE / 'astronauts' / clip['clip_id'] / 'source_provenance.json', OUT / 'credits' / (clip['clip_id'] + '_source.json'))
for name in ['main.pdf', 'source_bundle.zip']:
    copy(SOURCE / 'paper' / name, OUT / 'paper' / name)
for src in (SOURCE / 'paper/images').iterdir():
    if src.is_file(): copy(src, OUT / 'paper/images' / src.name)
copy(ROOT / 'data/raw/ProGait/README.md', OUT / 'credits/ProGait_SOURCE_README.md')
(OUT / 'credits/DATA_SOURCES.txt').write_text('''VIDEO AND POSE PROVENANCE
ProGait: Xiangyu Yin et al., ProGait: A Multi-Purpose Video Dataset and Benchmark
for Transfemoral Prosthesis Users, ICCV 2025.
https://huggingface.co/datasets/ericyxy98/ProGait
The original dataset card and its license declaration accompany this package.

LJMU: Luke Topham et al., A diverse and multi-modal gait dataset of indoor and
outdoor walks acquired using multiple cameras and sensors, Scientific Data 10,
320 (2023). https://doi.org/10.1038/s41597-023-02161-8
Dataset repository: https://opendata.ljmu.ac.uk/id/eprint/133/
Source terms remain applicable; inclusion does not change them.

ISS treadmill-running excerpts:
ESA / NASA, Horizons mission exercise in space (2018):
https://www.esa.int/media/horizons-mission-exercise-in-space
NASA, How to Run in Microgravity, Frank Rubio recording (2023):
https://images.nasa.gov/details/jsc2023m000211_Frank%20Rubio_Treadmill_Exercise_in_Space_Spanish_HighQuality_1080p_MP4
Source agency attribution and existing embedded logos remain intact.
New poses: Sapiens2-5B, BF16, official UDP decoding and flip averaging.
These are measurement-only examples, excluded from gait classification and
all paper scores. Source frame mappings and rotation are in credits/*_source.json.

Videos are existing prepared excerpts. Body landmarks and model signals were
computed previously by Theia. Exact source frame offsets and content hashes
are recorded in media_provenance.json.

OpenSim Gait2392 geometry: see anatomy_assets/credits.txt (CC BY 3.0).
Three.js and OrbitControls: see anatomy_assets/vendor/LICENSE (MIT).
The renderer is bundled from those exact local source files with esbuild.
''')
(OUT / 'README.txt').write_text('''THEIA GAIT DEMO - SELF-CONTAINED OFFLINE EDITION
15 November 2026

OPEN
1. Extract the entire ZIP to a normal folder.
2. Open START_DEMO.html, or index.html, in a modern browser.
3. Use Collection and Video to select the example. Press Play or click the timeline.
4. Red angles toggles the measured 2D angle arcs and degree labels in flagged regions.
5. Detailed pose adds the 23 cached body/face/foot points, including heels and toes.
   This affects drawing only; prediction inputs and scores stay unchanged.
6. Choose ISS treadmill running, or open START_SPACE_RUNNING.html, for the two
   NASA/ESA running clips. Their neutral angles are measurements, not anomaly alerts.

7. Show simulated knee alert opens a manually staged left-knee alert on the NASA
   clip, from 1.2 to 2.4 seconds. It is explicitly labeled and is not a model result.

No server, Python, installation, internet connection, API key or GPU is required.
Keep the extracted files together. Opening a page from inside a ZIP viewer will
not expose its videos and scripts; extract first. WebGL must be available for 3D.
This edition also works on an ordinary static web server if you prefer one.

INCLUDED
19 model-replay MP4 clips, 19 cached pose records, and 20 collection/clip entries:
- 14 statistical-pooling clips, including 5 historical examples selected with the previous TCN.
- 4 distribution-model case studies (3 ProGait and 1 LJMU reference).
- 2 ISS treadmill-running clips. Sapiens2-5B processed all 378 source frames;
  any trimming for landmark coverage is recorded in astronaut_treadmill_provenance.json.
- All displayed predictions, criteria, scores, charts and source-frame offsets.
- Local Three.js renderer, OpenSim anatomy data, credits and licenses.
- The revised manuscript, 35-run appendix and LaTeX/reproduction source archive.

All inference is precomputed. This package replays the existing results; it does
not train a model or estimate poses for new videos. The page summarizes the current pooling Table 2; replay signals stay separate.
Original evaluation arrays and clinical limitations are unchanged. Model flags are review signals, not diagnoses.
Reference links lead to external publications only when clicked; they are not
required for playback or rendering. Related workspace-only report links were
replaced by sections in this self-contained page.

offline-data.js preloads the same JSON records shipped separately so file://
opening does not require fetch requests. anatomy.bundle.js combines the local
renderer and Three.js as a classic script. No CDN assets are used. IBM Plex fonts and their OFL licenses are bundled.
The page uses the light Theia Space Workbench style. Video display retains original RGB color;
original RGB media remain unchanged. Updated paired screenshots appear in Figure 6.
Red angles show current-frame projected geometry, not a new model decision.
Knee angles use 180 degrees for straight; arm angles are at the shoulder.

Verification scope: the original package was tested from an extracted folder
on a loopback server. Direct file:// launch could not be tested because the
automation browser permits only HTTP(S). See PACKAGE_VERIFICATION.json for
recorded checks and later feature revisions.

See PACKAGE_MANIFEST.json for file hashes and media_provenance.json for video
and pose origins. Dataset attribution and source terms are in credits/.
''')
space_clips = [c for c in data['clips'] if c['kind'] == 'astronaut']
if data.get('active_model'):
    readme=OUT/'README.txt'
    readme.write_text('''STATISTICAL-POOLING MODEL EDITION
The default collection now uses the three-seed statistical-pooling ensemble,
with threshold 0.5, for all 14 supervised clips. Predictions were rerun and
matched the frozen experiment outputs exactly. Original test AP is 0.627,
precision 0.474 and recall 0.404 against automatic pose-derived criteria.
This is the highest observed test-AP family, not the validation-selected winner.
It has been installed in this experimental demo at the user's request.

Four distribution-model examples use their separate original density model.
Two ISS examples show measured poses; the simulated alert remains labeled.
The historical five-clip subset was selected using the previous TCN and was
not reselected to favor the new model. All disagreements remain visible.

Open START_DEMO.html after extracting the complete ZIP. No GPU, installation
or network connection is required for playback. The browser replays stored
predictions; it does not run pose extraction or inference on uploaded videos.
The three checkpoint files and model definitions are also in model/ for audit;
using them outside the replay requires Python, PyTorch and prepared pose inputs.
Current model: pooling_model.json. Verified replay: pooling_demo_verification.json.
Current window predictions: pooling_window_results.json and pooling_predictions.npz.
The older window_results.json, report_results.json and inference_verification.json
are retained only as the previous-TCN research record.

''' + readme.read_text())
with (OUT / 'README.txt').open('a') as stream:
    stream.write('\nISS running coverage: ' + str(sum(c['video']['frames'] for c in space_clips)) +
        ' retained native frames across two clips. Source frames and any trim\n'
        'are recorded separately. Neutral ink angles are measured geometry; no\n'
        'astronaut form-error classifier is applied. Confidence coverage does not\n'
        'establish anatomical accuracy or clinical correctness.\n')

manifest = dict(
    built_at=datetime.now(timezone.utc).isoformat(),
    entrypoint='START_DEMO.html', unique_videos=len(poses), model_clip_entries=len(data['clips']),
    total_mp4_files=len(list(OUT.rglob('*.mp4'))),
    requires_network=False, requires_server=False, inference='precomputed',
    files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'PACKAGE_MANIFEST.json', 'PACKAGE_VERIFICATION.json'}},
)
(OUT / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(folder=str(OUT), files=len(manifest['files']), unique_videos=len(poses), bytes=sum(x['bytes'] for x in manifest['files'].values())), indent=2))

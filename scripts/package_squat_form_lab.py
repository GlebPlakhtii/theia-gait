"""Package the completed local squat analysis and verify portable assets."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/squat_form_lab_20261114'
DEST = ROOT / 'deliverables/Theia_Squat_Form_Lab_2026-11-14'

def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def sha(path): return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()

def main():
    data = read(OUT / 'assessments.json')
    publisher = []
    for source in data['sources']:
        if source['publisherLabel'] is None:
            continue
        reps = [c for c in data['clips'] if c['source_id'] == source['id'] and c['features']['completeRep']]
        counts = Counter(c['assessment']['status'] for c in reps)
        binary = counts['good'] + counts['bad']
        prediction = None
        if binary >= len(reps) / 2 and counts['good'] != counts['bad']:
            prediction = 'good' if counts['good'] > counts['bad'] else 'bad'
        publisher.append(dict(id=source['id'], reference=source['publisherLabel'], prediction=prediction,
                              binaryReps=binary, candidateReps=len(reps), counts=dict(counts)))
    data['validation'] = dict(publisherSources=publisher, independentClinicalReference=False,
        meaning='Descriptive publisher-file agreement only; no independent per-repetition or ARED clinical validation.')
    for src, target in [('squat_form_lab.html','index.html'),('squat_form_lab.css','style.css'),
                        ('squat_form_lab.js','app.js'),('squat_form_rules.js','rules.js')]:
        shutil.copy2(ROOT / 'scripts' / src, DEST / target)
    assets = {}
    ids = set()
    source_frames = 0
    for source in data['sources']:
        folder = DEST / 'opensim' / source['id']
        tree = ET.parse(folder / 'ik_setup.xml')
        tool = tree.getroot().find('InverseKinematicsTool')
        for key, value in [('model_file','model.osim'),('marker_file','estimated_markers.trc'),
                           ('output_motion_file','kinematics.mot'),('results_directory','.')]:
            element = tool.find(key)
            if element is None:
                element = ET.SubElement(tool, key)
            element.text = value
        tree.write(folder / 'ik_setup.xml', encoding='utf-8', xml_declaration=True)
        source['opensimSetup'] = f'opensim/{source["id"]}/ik_setup.xml'
        completed = read(OUT / 'sources' / source['id'] / 'opensim/complete.json')
        assert completed['frames'] == source['frames']
        source_frames += completed['frames']
        # The original audit hashes refer to the original run, not the portable setup XML.
        completed['original_run_file_sha256'] = completed.pop('file_sha256')
        completed['portable_file_sha256'] = {p.name:sha(p) for p in folder.iterdir() if p.name != 'complete.json'}
        completed['portable_setup_change'] = 'Explicit model_file and relative paths; numerical outputs unchanged.'
        write(folder / 'complete.json', completed)
        for path in folder.iterdir(): assets[str(path.relative_to(DEST))] = sha(path)
    for clip in data['clips']:
        assert clip['clip_id'] not in ids
        ids.add(clip['clip_id'])
        assert clip['assessment']['clinicalVerdict'] is None
        for key in ['video','track']:
            path = DEST / clip[key]
            assert path.is_file() and path.stat().st_size > 0, path
            assets[clip[key]] = sha(path)
        text = (DEST / clip['track']).read_text()
        track = json.loads(text.split('=',2)[2].strip().removesuffix(';'))
        for key in ['times','pose2d','scores','stabilized','world','opensim']:
            assert len(track[key]) == clip['frames'], (clip['clip_id'],key)
        assert all(a < b for a,b in zip(track['times'],track['times'][1:]))
        assert abs(track['times'][-1] - (clip['frames']-1)/clip['fps']) < .003
        assert clip['source_last_frame'] - clip['source_first_frame'] + 1 == clip['frames']
    assert source_frames == 11139 and len(ids) == 161
    assert sum(c['features']['completeRep'] for c in data['clips']) == 116
    for path in ['app.js','rules.js']:
        subprocess.run(['node','--check',str(DEST/path)],check=True)
    subprocess.run(['node',str(ROOT/'scripts/squat_form_rules.js'),'--test'],check=True)
    write(DEST/'data.json',data)
    (DEST/'data.js').write_text('window.THEIA_DATA='+json.dumps(data,separators=(',',':'),allow_nan=False)+';\n')
    scripts = ['squat_form_pipeline.py','opensim_squat_fit.py','squat_form_features.py','squat_form_rules.js',
               'squat_form_lab.html','squat_form_lab.css','squat_form_lab.js','package_squat_form_lab.py','serve_squat_form_lab.py']
    provenance = dict(createdAt=datetime.now(timezone.utc).isoformat(),
        scope=data['scope'], protocol=read(OUT/'protocol.json'),
        source_inputs=read(OUT/'sources.json'),
        motionbert_code=read(ROOT/'vendor/motionbert/source_manifest.json'),
        motionbert_checkpoint=read(ROOT/'models/motionbert/provenance.json'),
        scripts={s:sha(ROOT/'scripts'/s) for s in scripts},
        defaults=data['defaults'], thresholdProvenance=data['thresholdProvenance'],
        evidence=data['evidence'], assets=assets)
    for name in ['index.html','style.css','app.js','rules.js','data.js','data.json']:
        provenance['assets'][name] = sha(DEST/name)
    write(DEST/'provenance.json',provenance)
    verify = dict(checkedAt=datetime.now(timezone.utc).isoformat(), sources=len(data['sources']),
        nativeFrames=source_frames, clips=len(ids), candidateRepetitions=116, contextClips=45,
        counts=data['summary']['counts'],
        assetIntegrity='PASS: all video/track assets exist, SHA256 recorded, frame counts and timestamps consistent',
        ruleChecks='PASS: quality precedence, depth, deadband, camera gate, distinct ARED profile',
        opensimRuns='PASS: completed actual OpenSim IK for all 12 source sequences',
        portableOpenSimReplay='pending', browserChecks='pending',
        clinicalValidation='Not available; these checks validate software behavior and packaging only.')
    if (DEST/'verification.json').exists():
        previous = read(DEST/'verification.json')
        for key in ['portableOpenSimReplay','browserChecks']:
            if previous.get(key) != 'pending': verify[key] = previous[key]
    write(DEST/'verification.json',verify)
    (DEST/'START_HERE.txt').write_text('THEIA SQUAT FORM LAB\n\nOpen index.html in Chrome or another modern browser. Keep this folder intact.\n'
        '161 original RGB clips; 116 candidate repetitions and 45 context segments.\n'
        'Use Projection & model view to compare foot-relative 2D, MotionBERT and OpenSim.\n'
        'Rule settings immediately recompute all clip results; Export saves the current decisions.\n'
        'The 90-degree ground target is a demonstration prescription, not a universal good-squat standard.\n'
        'ARED has no absolute ground-depth requirement. Thresholds are provisional and editable.\n'
        'No independent clinical validation or calibrated 3D ground truth is available.\n'
        'OpenSim assets use nominal scale and have no force/muscle analysis.\n'
        'Original doctor-labeling materials are separate and unchanged.\n')
    print(json.dumps({'clips':len(ids),'frames':source_frames,'counts':data['summary']['counts'],'publisher':publisher}))

if __name__ == '__main__': main()

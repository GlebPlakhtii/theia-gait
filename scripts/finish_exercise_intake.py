"""Run the reviewed intake queue through native poses and a separate doctor packet.

Never changes previously delivered packets or sends anything to a third party.
"""
import csv
import hashlib
import json
import subprocess
import shutil
import sys
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import build_doctor_clips as builder
import package_doctor_clips as packager
from run_local_experiments import load_signal, candidates

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'audit/new_video_intake_20261114'
PREP=ROOT/'data/prepared/exercise_intake_20261114'
POSE=ROOT/'data/poses/sapiens2_5b_intake_full'
REPLAY=ROOT/'data/pose_replays/exercise_intake_20261114'
OUT=ROOT/'deliverables/Theia_Doctor_Extra_Clips_2026-11-14'
QUEUE=PREP/'pose_queue_full.csv'
STATUS=AUDIT/'pipeline_status.json'


def save(phase,**extra):
    data=json.loads(STATUS.read_text()) if STATUS.exists() else {}
    data.update(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**extra)
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(STATUS)
    print(phase,extra,flush=True)


def run(args):
    print('COMMAND',args,flush=True)
    subprocess.run(args,cwd=ROOT,check=True)


def main():
    if OUT.exists():raise ValueError('Refuse to overwrite a doctor packet; inspect previous pipeline status')
    queue=list(csv.DictReader(open(QUEUE)))
    selections={r['id']:r for r in json.load(open(PREP/'selections.json'))}
    expected=sum(selections[q['clip_id']]['frames'] for q in queue)
    save('native_pose_inference',status='running',expected_frames=expected)
    run([sys.executable,'scripts/run_sapiens2_pose.py','--queue',str(QUEUE),'--rois',str(PREP/'pose_rois_full.json'),
         '--output',str(POSE),'--checkpoint-every','25'])
    save('auditing_pose')
    run([sys.executable,'scripts/audit_pose_batch.py',str(POSE),'--queue',str(QUEUE)])
    integrity=json.load(open(POSE/'integrity_report.json'));assert integrity['total_frames']==expected and not integrity['errors']
    run([sys.executable,'scripts/plot_pose_review.py',str(POSE),'--queue',str(QUEUE)])
    save('rendering_pose_replays')
    run([sys.executable,'scripts/render_pose_replay.py',str(POSE),'--queue',str(QUEUE),'--output',str(REPLAY)])
    custom={};cycles=[]
    for q in queue:
        cid=q['clip_id'];spec=selections[cid];path=POSE/cid/'poses.csv'
        frames,times,raw,valid=load_signal(path);assert len(frames)==spec['frames']
        parameter_results=[]
        for prominence in [15.,10.,20.]:
            smooth,events=candidates(times,raw,valid,prominence)
            for event in events:event['completion_source_frame']=int(frames[event['end_index']])
            parameter_results.append(dict(prominence_deg=prominence,primary=prominence==15.,candidate_count=len(events),candidates=events))
        cycles.append(dict(clip_id=cid,native_frames=len(frames),reference_labels_available=False,parameter_results=parameter_results))
        custom[cid]=dict(id=cid,domain=spec['domain'],pose_csv=str(path),raw=spec['prepared_path'],replay=str(REPLAY/(cid+'_pose_review.mp4')))
    builder.OUT=OUT;builder.SOURCE_MAPPING=AUDIT/'doctor_source_mapping.json'
    save('cutting_and_prelabeling')
    builder.plan(custom_sources=custom,cycle_results=cycles);builder.render()
    builder.write(AUDIT/'cycle_predictions.json',cycles)
    packager.OUT=OUT;packager.main()
    records=json.load(open(OUT/'automatic_labels/clips.json'));count=len(records)
    mission=sum(c['domain']=='in_mission_ared' for c in records);ground=count-mission
    package_id='theia-doctor-extra-clips-20261114-v1'
    page=(OUT/'REVIEW_CLIPS.html').read_text().replace('theia-doctor-short-clips-20261114-v2',package_id)
    page=page.replace('Priority 1: ARED — 32 short clips',f'In-mission ARED — {mission} clips').replace('Optional: gym — 80 short clips',f'Ground ARED — {ground} clips').replace('All 112 clips',f'All {count} clips')
    page=page.replace("c.source_id==='ARED'","c.domain==='in_mission_ared'").replace("c.source_id!=='ARED'","c.domain!=='in_mission_ared'")
    page=page.replace('ARED is in microgravity; load and prescription are unknown.','The in-mission group is in microgravity; the ground group is terrestrial training. Load and prescription are unknown.')
    page=page.replace('candidate squat repetition','candidate knee-flexion cycle').replace('ARED footage: ESA/NASA.','Sources: NASA, ESA/NASA and CSA/NASA; see provenance.')
    (OUT/'REVIEW_CLIPS.html').write_text(page)
    (OUT/'index.html').write_text('<!doctype html><meta http-equiv="refresh" content="0;url=REVIEW_CLIPS.html"><a href="REVIEW_CLIPS.html">Open form review</a>')
    blank=json.load(open(OUT/'automatic_labels/blank_doctor_review.json'));blank['package_id']=package_id
    builder.write(OUT/'automatic_labels/blank_doctor_review.json',blank)
    (OUT/'START_HERE.txt').write_text(f'''THEIA — ADDITIONAL SHORT CLIPS FOR FORM REVIEW
Extract the entire ZIP and open REVIEW_CLIPS.html. Watch each short clip,
choose the visible-form conclusion, optionally add notes, then Save & next.
Export doctor conclusions as JSON and return it to Hlib Plakhtii / Theia.
No manual timestamps, repetition counts or skeleton tracing are requested.

{count} review items: {mission} in-mission ARED, {ground} ground ARED.
Each has original and pose versions; review each item once. The ground group
is auxiliary material, not mission evidence or clinical rehabilitation data.
All {expected} selected native frames are covered; padding overlaps are context.

Pose geometry, movement phases and cycle boundaries are automatic candidates.
Angles are projected 2D angles; camera perspective and occlusion remain.
Short CSA b-roll has limited context and feet near the image edge.
Original frame indices, crop/rotation transforms and source URLs are in provenance.
The prepared RGB may be cropped or rigidly rotated; no missing pixels were filled.
No clinical conclusion, pain, injury, safety or joint-load label is inferred.
This is model-assisted review, not a blinded test of pose accuracy.

Credit NASA, ESA/NASA, CSA/NASA. Source reuse terms continue to apply.
''')
    builder.write(OUT/'provenance/intake_sources.json',[selections[q['clip_id']] for q in queue])
    for q in queue:
        cid=q['clip_id'];origin=Path(q['frame_map']).parent
        target=OUT/'provenance'/cid;target.mkdir()
        shutil.copy2(origin/'original_frame_lineage.csv',target/'original_frame_lineage.csv')
        shutil.copy2(origin/'provenance.json',target/'preprocessing.json')
    plan=json.load(open(OUT/'provenance/plan.json'));plan['user_correction']='Additional reviewed sources; existing doctor packets preserved.'
    plan['frame_clock']='Native rational frame rates retained, including 30000/1001 and 60000/1001.'
    builder.write(OUT/'provenance/plan.json',plan)
    # Replace the old packager's attribution, which describes its original dataset.
    (OUT/'provenance/SOURCES.txt').write_text('\n\n'.join(selections[q['clip_id']]['source_page']+'\n'+selections[q['clip_id']]['notes'] for q in queue))
    save('checking_delivered_videos',review_items=count)
    files=sorted(OUT.glob('clips*/*.mp4'))
    def verify(path):
        result=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1','-xerror','-err_detect','explode','-i',str(path),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
        assert result.returncode==0 and not result.stderr,(str(path),result.stderr)
    with ThreadPoolExecutor(3) as pool:list(pool.map(verify,files))
    assert len(files)==2*count and all(r['doctor_conclusion'] is None for r in records)
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(builder.digest(p)+'  '+str(p.relative_to(OUT)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt')+'\n')
    archive=OUT.with_suffix('.zip');save('creating_zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.relative_to(OUT)),compress_type=zipfile.ZIP_STORED if p.suffix=='.mp4' else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    digest=builder.digest(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    save('complete',status='complete',pose_frames=expected,review_items=count,in_mission_items=mission,ground_items=ground,zip=str(archive),zip_sha256=digest,
         full_postinference_visual_qc='pending; only pilot preview review and automated integrity/decode checks completed',clinical_benefit_established=False)
    from report_exercise_intake import main as report
    report()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save('failed',status='failed',error=repr(error),traceback=traceback.format_exc());raise

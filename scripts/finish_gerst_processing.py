"""Follow the specific Gerst job through audit, clips and doctor ZIP creation.

Preserves prior delivered packets. Does not send messages or publish anything.
"""
import argparse,csv,json,os,subprocess,sys,time,traceback,zipfile,hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import build_doctor_clips as builder
import package_doctor_clips as packager
from run_local_experiments import load_signal,candidates

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'audit/gerst_processing_20261114'
POSE=ROOT/'data/poses/sapiens2_5b_gerst_full'
OUT=ROOT/'deliverables/Theia_Doctor_Gerst_Clips_2026-11-14'
REPLAYS=ROOT/'data/pose_replays/gerst_20261114'
RAW=ROOT/'data/prepared/gerst_20261114'
STATUS=AUDIT/'pipeline_status.json'


def save(phase,**kw):
    data=json.loads(STATUS.read_text()) if STATUS.exists() else {}
    data.update(updated_at=datetime.now(timezone.utc).isoformat(),phase=phase,**kw)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(STATUS)
    print(phase,kw,flush=True)


def command(args):
    print('COMMAND',args,flush=True)
    subprocess.run(args,cwd=ROOT,check=True)


def main(wait=True):
    queue=list(csv.DictReader(open(AUDIT/'queue.csv')))
    job=json.load(open(AUDIT/'full_job.json'));pid=job['pid']
    if wait:
        save('waiting_for_native_pose_run',inference_pid=pid,expected_frames=3850,status='running')
        while not all((POSE/q['clip_id']/'complete.json').exists() for q in queue):
            try:os.kill(pid,0)
            except ProcessLookupError:raise RuntimeError('Pose process exited before all completion files; inspect full.log. No success claimed.')
            # This is a background pipeline monitor, not a blocking chat/tool wait.
            time.sleep(15)
    assert all((POSE/q['clip_id']/'complete.json').exists() for q in queue)
    save('auditing_pose')
    command([sys.executable,'scripts/audit_pose_batch.py',str(POSE),'--queue',str(AUDIT/'queue.csv')])
    command([sys.executable,'scripts/plot_pose_review.py',str(POSE),'--queue',str(AUDIT/'queue.csv')])
    integrity=json.load(open(POSE/'integrity_report.json'));assert integrity['total_frames']==3850 and not integrity['errors']
    save('rendering_pose_replays')
    # Renderer is deliberately non-overwriting; skip only an audited complete run.
    if not (REPLAYS/'index.json').exists():
        command([sys.executable,'scripts/render_pose_replay.py',str(POSE),'--queue',str(AUDIT/'queue.csv'),'--output',str(REPLAYS)])
    RAW.mkdir(parents=True,exist_ok=True)
    custom={};cycle_results=[]
    for n,q in enumerate(queue,1):
        cid=q['clip_id'];csvpath=POSE/cid/'poses.csv';frames,times,raw,valid=load_signal(csvpath)
        assert len(frames)==int(round((float(q['end_s'])-float(q['start_s']))*25))
        cut=RAW/(cid+'.mp4')
        if not cut.exists():
            command(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',q['local_path'],'-map','0:v:0',
                '-vf',f'trim=start_frame={frames[0]}:end_frame={frames[-1]+1},setpts=PTS-STARTPTS',
                '-frames:v',str(len(frames)),'-an','-c:v','libx264','-crf','18','-preset','fast','-threads','2',
                '-fps_mode','passthrough','-movflags','+faststart','-n',str(cut)])
        parameter_results=[]
        for prominence in [15.,10.,20.]:
            smooth,events=candidates(times,raw,valid,prominence)
            for e in events:e['completion_source_frame']=int(frames[e['end_index']])
            parameter_results.append({'prominence_deg':prominence,'primary':prominence==15.,'candidate_count':len(events),'candidates':events})
        cycle_results.append({'clip_id':cid,'native_frames':len(frames),'source_start_s':float(times[0]),
                              'duration_s':len(frames)/25,'score_gate_coverage':float(valid.mean()),
                              'reference_labels_available':False,'parameter_results':parameter_results})
        custom[cid]={'id':f'GERST_{n:02d}','domain':'in_mission_ared','pose_csv':str(csvpath),
                     'raw':str(cut),'replay':str(REPLAYS/(cid+'_pose_review.mp4'))}
    builder.OUT=OUT;builder.SOURCE_MAPPING=AUDIT/'short_clip_source_mapping.json'
    save('cutting_and_prelabeling')
    builder.plan(custom_sources=custom,cycle_results=cycle_results)
    builder.render()
    builder.write(AUDIT/'cycle_predictions.json',cycle_results)
    packager.OUT=OUT;packager.main()
    records=json.load(open(OUT/'automatic_labels/clips.json'));count=len(records)
    repetitions=sum(r['automatic_type']=='candidate_repetition' for r in records);contexts=count-repetitions
    new_id='theia-doctor-gerst-short-clips-20261114-v1'
    page=(OUT/'REVIEW_CLIPS.html').read_text()
    page=page.replace('theia-doctor-short-clips-20261114-v2',new_id)
    page=page.replace('Priority 1: ARED — 32 short clips',f'Gerst ARED — {count} short clips')
    page=page.replace('<option value="GYM">Optional: gym — 80 short clips</option>','')
    page=page.replace('All 112 clips',f'All {count} clips')
    page=page.replace("c.source_id==='ARED'","c.domain==='in_mission_ared'").replace("c.source_id!=='ARED'","c.domain!=='in_mission_ared'")
    page=page.replace('candidate squat repetition','candidate knee-flexion cycle')
    (OUT/'REVIEW_CLIPS.html').write_text(page)
    blank=json.load(open(OUT/'automatic_labels/blank_doctor_review.json'));blank['package_id']=new_id
    builder.write(OUT/'automatic_labels/blank_doctor_review.json',blank)
    (OUT/'START_HERE.txt').write_text(f'''THEIA — GERST SHORT-CLIP FORM REVIEW
Extract this ZIP and open REVIEW_CLIPS.html. Watch the short cut, choose your
form conclusion and write notes. Timing, poses and projected knee angles are
already filled in. Save & next advances; Export doctor conclusions saves JSON
to return to Hlib Plakhtii / Theia. CSV is an alternative. Nothing is sent online.

This supplement has {count} short review items ({repetitions} automatic cycle
candidates and {contexts} context segments), each with original-only and
original/pose side-by-side versions. Review each item once. The model may label
a non-squat movement as a knee cycle; use Not a squat or Cannot judge as needed.
No manual timing annotation is required.

Source: Alexander Gerst, Blue Dot mission 2014, ESA/NASA ARED footage. The two
processed source windows are 8–75 s and 145–232 s, totaling 154 s / 3,850 native
frames. They include setup, squats and other full-body ARED exercise. The rest
of the 12.5-minute recording was screened, not pose-processed in this run.

Projected 2D angles depend on camera viewpoint and occlusion. They do not measure
anatomical joint loading, pain, fatigue or safety. ARED is in microgravity; load
and personal prescriptions are unknown. Your form conclusions remain blank.
This is model-assisted expert review, not a pose-blinded benchmark. The earlier
doctor packet is preserved, with distinct clip IDs and a distinct review file.
All clips are silent, with original privacy/appearance retained. Credit ESA/NASA.
''')
    plan=json.load(open(OUT/'provenance/plan.json'))
    plan['user_correction']='Gerst supplement; same pre-cut/prefilled doctor workflow, no manual timestamps required.'
    plan['source_context']='Blue Dot mission 2014; two full-body ARED views; no EasyMotion-2 context inferred.'
    builder.write(OUT/'provenance/plan.json',plan)
    (OUT/'provenance/SOURCE.txt').write_text('ESA/NASA — Alexander Gerst, Blue Dot mission 2014.\nhttps://dlmultimedia.esa.int/download/public/videos/2024/01/026/2401_026_BR_004.mp4\nSource SHA256 d7bec40ee47691bd04692e4d8d99984f8b386aa3ad06944d62442b8f8841d33c\nDerived silent temporal cuts. Preserve attribution; source reuse terms apply.\n')
    save('checking_delivered_videos',review_items=count)
    files=sorted(OUT.glob('clips*/*.mp4'))
    def check(p):
        r=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-xerror','-err_detect','explode',
                          '-i',str(p),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
        if r.returncode or r.stderr:raise RuntimeError(str(p)+': '+r.stderr)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check,files))
    assert len(files)==2*count and all(r['doctor_conclusion'] is None for r in records)
    manifest_files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(builder.digest(p)+'  '+str(p.relative_to(OUT)) for p in manifest_files)+'\n')
    archive=OUT.with_suffix('.zip')
    save('creating_zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.relative_to(OUT)),compress_type=zipfile.ZIP_STORED if p.suffix=='.mp4' else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    digest=builder.digest(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    save('complete',status='complete',zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=digest,
         review_items=count,cycle_candidates=repetitions,context_clips=contexts,pose_frames=3850,
         video_files=len(files),strict_decode_passed=True,zip_crc_passed=True,
         full_video_pose_processed=False,clinical_decision_benefit_established=False,
         full_postinference_visual_qc='not yet performed; automated checks and 24-frame pilot only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--no-wait',action='store_true');a=p.parse_args()
    try:main(not a.no_wait)
    except BaseException as e:
        save('failed',status='failed',error=repr(e),traceback=traceback.format_exc())
        raise

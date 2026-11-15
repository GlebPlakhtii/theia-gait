"""Finish source QA and refresh the combined offline doctor package."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import prepare_progait as prep
import build_unified_doctor_review as portal

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'audit/progait_20261114'


def main():
    status_path=AUDIT/'preparation_status.json'
    while json.loads(status_path.read_text()).get('status')=='processing':time.sleep(15)
    status=json.loads(status_path.read_text())
    if status['status']=='failed':
        print('Resuming source preparation; completed sources are reused.',flush=True)
        subprocess.run([sys.executable,'-u','scripts/prepare_progait.py'],cwd=ROOT,check=True)
    status=json.loads(status_path.read_text());assert status['status']=='complete' and status['complete_sources']==412,status
    metadata=[json.loads(x) for x in (prep.RAW/'metadata.jsonl').read_text().splitlines()]
    metas={'PG_'+m['subset']+'_'+m['ID']:m for m in metadata}
    all_records=[];mismatched=[];tracking=[];native_frames=0;native_duration=0;publisher_frames=0;counts=Counter();rates=Counter();short=[]
    for cid,m in sorted(metas.items()):
        path=prep.PREP/cid/'complete.json';data=json.loads(path.read_text());rows=data['records'];n=data['frames']
        with gzip.open(prep.RAW/'annotations'/m['subset']/(m['ID']+'_keypoints.npy.gz'),'rb') as f:kp=np.load(f,allow_pickle=False)
        publisher_frames+=len(kp);native_frames+=n;native_duration+=sum(r['video_duration_s'] for r in rows);rates[data['rate']]+=1
        assert rows[0]['source_first_frame']==0 and rows[-1]['source_last_frame']==n-1
        assert all(a['source_last_frame']+1==b['source_first_frame'] for a,b in zip(rows,rows[1:]))
        assert sum(r['frames'] for r in rows)==n
        if data.get('reference_problem'):mismatched.append(dict(clip_id=cid,problem=data['reference_problem']))
        if data.get('missing_reference_roi_frames'):tracking.append(dict(clip_id=cid,missing_frames=len(data['missing_reference_roi_frames'])))
        aligned=kp.shape==(n,23,3) and np.isfinite(kp).all()
        data['reference_frame_alignment']='Publisher file/frame ordering assumed; matching native RGB and pose-array counts' if aligned else 'unverified; overlay withheld'
        if aligned:
            good=(kp[:,:,2]>=.3)&(kp[:,:,0]>=0)&(kp[:,:,0]<data['width'])&(kp[:,:,1]>=0)&(kp[:,:,1]<data['height'])
            angle_input=kp.copy();angle_input[:,:,2]=np.where(good,kp[:,:,2],0);signals=prep.angles(angle_input)
        category=re.search(r'CCC\s*:\s*(\d+)',m['txt'])
        reference_status='available' if category else ('missing' if not m['txt'].strip() else 'publisher_marks_not_for_analysis')
        counts[reference_status]+=1
        for r in rows:
            assert (prep.OUT/r['original_video']).is_file()
            if r.get('pose_data'):
                raw=(prep.OUT/r['pose_data']).read_text();obj=json.loads(raw.removeprefix('window.THEIA_POSE=').rstrip(';\n'))
                assert len(obj['frames'])==r['frames'] and obj['clip_id']==r['clip_id']
                assert all(len(a)==69 for a in obj['frames'])
            r['reference_assessment_status']=reference_status
            if not aligned:r['context']='Prosthesis walking trial. Source privacy masking preserved. Pose overlay unavailable because annotation alignment could not be verified. Please review the original video.'
            if r['video_duration_s']<1:
                short.append(r['clip_id'])
                note=' Very short source-end context; use the preceding clip or Cannot judge.' if n>=30 else ' The entire source is under one second; it does not show a complete gait cycle.'
                if note not in r['context']:r['context']+=note
            if aligned:
                first,last=r['source_first_frame'],r['source_last_frame']+1
                for side in ['left','right']:
                    values=signals[side][first:last];values=values[np.isfinite(values)]
                    for name,v in [('min',values.min() if len(values) else None),('max',values.max() if len(values) else None),('excursion',np.ptp(values) if len(values) else None)]:
                        r['automatic_measurements'][f'{side}_projected_knee_{name}_deg']=round(float(v),1) if v is not None else None
            assert not r.get('doctor_conclusion') and not r.get('doctor_notes')
        prep.write(path,data);all_records.extend(rows)
    prep.write(prep.OUT/'automatic_labels/clips.json',all_records)
    # Clearly name the runnable subset, rather than implying that missing crops are usable.
    old_queue=prep.PREP/'pose_queue_all.csv'
    if old_queue.exists():old_queue.replace(prep.PREP/'pose_queue_ready.csv')
    report=dict(status='complete',original_videos=412,participants=4,review_clips=len(all_records),
        decoded_native_frames=native_frames,publisher_pose_frames=publisher_frames,source_duration_seconds=native_duration,
        native_rates=dict(rates),reference_assessment_status=dict(counts),pose_alignment_exceptions=mismatched,
        sources_needing_target_tracking_review=tracking,short_context_clips_under_1s=short,
        all_selected_source_frames_covered_once=True,publisher_pose_frame_arrays_checked=True,doctor_conclusions_blank=True,
        pose_reference_alignment_basis='Publisher filename and array order, equal decoded frame counts, and sampled visual checks. NPY arrays supply no independent timestamps; equality of counts alone does not prove anatomical or temporal correctness.',
        data_revision=json.load(open(AUDIT/'acquisition_status.json'))['revision'],
        reference_provenance='Publisher RTMW-assisted annotations, partly manually corrected. These are not new Sapiens2 predictions or wholly manual ground truth.',
        grouped_filename_prefixes=len({(m['subset'],'_'.join(m['ID'].split('_')[:3])) for m in metadata}),
        grouping_limit='148 filename-prefix groups in this release; paper reports 144 trials. Trial equivalence and camera synchronization have not been established.',
        completed_at=datetime.now(timezone.utc).isoformat())
    prep.write(AUDIT/'final_source_audit.json',report)
    (prep.PREP/'README.txt').write_text(f'''PROGAIT — PREPARED RGB AND POSE INPUTS
412 RGB sources; {native_frames} decoded frames; {native_duration/60:.2f} minutes.
All sources have short original-video review cuts in deliverables/Theia_Doctor_ProGait_2026-11-14.
Publisher pose overlays are provided only where native frame counts align.
See audit/progait_20261114/final_source_audit.json for alignment and tracking exceptions.
pose_queue_ready.csv is the runnable subset with complete reference-derived subject crops.
pose_rois_all.json contains the reference-derived crops; do not call this independent person detection.
pose_queue_needs_tracking_review.json lists sources whose crops require review.
Original annotations remain unchanged in data/raw/ProGait. Gait assessment text is not copied to the doctor form.
Only 402 sources have a CCC gait category; four have no text and six are marked not for analysis.
The available categories describe prosthetic gait. They are not good/bad ARED squat ground truth.
Preserve participant and trial grouping in future studies; no subject should cross a claimed subject-held-out split.
ProGait: Yin et al., ICCV 2025; CC BY-NC-SA 4.0.
''')
    # Let the original intake pipeline finish its own integrity audit and packet.
    intake=ROOT/'audit/new_video_intake_20261114/pipeline_status.json'
    while json.loads(intake.read_text()).get('status')=='running':time.sleep(15)
    assert json.loads(intake.read_text())['status']=='complete','Additional intake needs recovery'
    # The watcher may be creating its first archive. Never write that ZIP concurrently.
    watcher=json.loads((ROOT/'audit/unified_review_20261114/watch_job.json').read_text())['pid']
    import os
    while True:
        try:os.kill(watcher,0)
        except ProcessLookupError:break
        time.sleep(10)
    portal.build();portal.archive()
    prep.write(ROOT/'audit/unified_review_20261114/delivery_complete.json',dict(status='complete',at=datetime.now(timezone.utc).isoformat(),source_audit=report))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

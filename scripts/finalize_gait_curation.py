"""Assemble only reviewed clips, preserve source provenance and native pose alignment."""
from pathlib import Path
from collections import Counter, defaultdict
import concurrent.futures as cf
import json, shutil, sys, subprocess
import numpy as np
from curate_gait_inputs import ROOT, AUDIT, OUT, write, sha, load_reference, source_inventory

JOINTS=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']

def main():
    plan=json.loads((AUDIT/'selection_plan.json').read_text())
    review=json.loads((AUDIT/'visual_review.json').read_text())
    pending=[c['clip_id'] for c in plan['clips'] if review.get(c['clip_id'],{}).get('status') not in ['accept','reject']]
    accepted=[c for c in plan['clips'] if review.get(c['clip_id'],{}).get('status')=='accept']
    missing=[c['clip_id'] for c in accepted if not (OUT/'staging'/f'{c["clip_id"]}.json').exists() and not (OUT/'with_amputation'/c['source']/f'{c["clip_id"]}.json').exists()]
    print(json.dumps(dict(accepted=Counter(c['source'] for c in accepted),pending_clips=len(pending),pending_sources=len(plan['pending_sources']),missing_exports=len(missing)),indent=2),flush=True)
    if '--check' in sys.argv:
        print('pending',pending);return
    assert not plan['pending_sources'], 'Source screen incomplete'
    assert not pending, pending
    assert not missing, missing
    sources=source_inventory();assert len(sources)==445
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        source_hashes=list(pool.map(lambda s:(s['id'],sha(ROOT/s['original'])),sources))
    actual=dict(source_hashes)
    assert all(actual[s['id']]==s['source_sha256'] for s in sources), 'Original changed'
    rows=[];seen={};duplicates=[]
    for c in accepted:
        dest=OUT/'with_amputation'/c['source']/f'{c["clip_id"]}.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
        stage=OUT/'staging'/dest.name
        meta=stage.with_suffix('.json') if stage.with_suffix('.json').exists() else dest.with_suffix('.json')
        encoded=json.loads(meta.read_text());video=stage if stage.exists() else dest
        assert sha(video)==encoded['encoded_sha256'],str(video)
        if encoded['encoded_sha256'] in seen:
            duplicates.append(dict(clip_id=c['clip_id'],duplicate_of=seen[encoded['encoded_sha256']],reason='Identical encoded video SHA-256'))
            continue
        seen[encoded['encoded_sha256']]=c['clip_id']
        if stage.exists():
            assert not dest.exists(), str(dest)
            stage.replace(dest)
        subject=c.get('subject')
        row=dict(c,video=str(dest.relative_to(OUT)),video_absolute=str(dest),sha256=encoded['encoded_sha256'],encoded_sha256=encoded['encoded_sha256'],
                 decoded_frames=encoded['decoded_frames'],transformations=encoded['transformations'],
                 review_status='accepted_sampled_visual_and_automated_screen',visual_review=review[c['clip_id']],
                 subject=subject,split_group=subject or 'additional_input_identity_unresolved',
                 clinical_form_label=None,pose_inference_status='quality_screen_only',
                 quality_screen=str((AUDIT/'screen'/f'{c["id"]}.json').relative_to(ROOT)))
        row['limitations']=['Sparse visual review does not certify visibility in every frame.','Natural self-occlusion, walking aids and hands-on assistance may remain.','Quality inclusion does not establish clinically correct gait.']
        if c['source']=='progait':
            ref=load_reference(c);assert ref is not None
            start,end=c['source_first_frame'],c['source_end_frame_exclusive']
            arr=ref[start:end];assert arr.shape==(c['clip_frames'],23,3)
            pose=OUT/'annotations/progait'/f'{c["clip_id"]}.npz';pose.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(pose,keypoints=arr,source_frame=np.arange(start,end),clip_frame=np.arange(end-start),
                                source_time_s=np.arange(start,end)/c['fps'],joint_names=np.array(JOINTS),
                                provenance=np.array('Official ProGait publisher 2D annotations; temporally sliced, not newly inferred'))
            row.update(publisher_pose=str(pose.relative_to(OUT)),publisher_pose_sha256=sha(pose),
                       pose_inference_status='publisher_23_keypoints_available_and_quality_screen',
                       pose_coordinate_system='Native source pixels (x,y,confidence); no spatial transform',
                       clinical_reference_scope='Source-video level; no new clip-level clinical annotation')
        write(dest.with_suffix('.json'),row);rows.append(row)
    healthy_base=ROOT/'data/prepared/fullbody_locomotion_20261114'
    healthy=json.loads((healthy_base/'pose_input_candidates.json').read_text())['clips']
    recheck_path=AUDIT/'cohort_recheck/decisions.json'
    recheck=json.loads(recheck_path.read_text()) if recheck_path.exists() else dict(sources={},clips={})
    unverified=[]
    for c in healthy:
        extra=recheck['clips'].get(c['id'])
        if not (c['source']=='ljmu360' and c['role']=='reported_healthy_cohort') and not (extra and extra['status']=='accept'):
            unverified.append(dict(id=c['id'],source=c['source'],existing_video=str(healthy_base/c['path']),reason='Amputation/health cohort not established; kept out of both cohort labels.'))
            continue
        src=healthy_base/c['path'];assert sha(src)==c['sha256']
        dest=OUT/'without_amputation'/c['source']/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(src,dest)
        assert sha(dest)==c['sha256']
        evidence=recheck['sources'].get(c['source'],dict(cohort_evidence_level='publisher_reported_healthy_cohort',needs_cohort_confirmation=False,
            cohort_basis='LJMU publisher-reported healthy cohort; no clip-level correct-form diagnosis',role=c['role']))
        row=dict(c,clip_id=c['id'],cohort='without_amputation',
                 video=str(dest.relative_to(OUT)),video_absolute=str(dest),source_sha256=c['original_sha256'],
                 clip_frames=c['frames'],clip_duration_s=c['duration_s'],source_first_frame=c['start_frame'],source_end_frame_exclusive=c['end_frame_exclusive'],
                 split_group=c['source']+'_'+c['subject'],review_status='accepted_existing_visual_review',
                 clinical_form_label=None,pose_inference_status='not_run',all_frames_visibility_certified=False,
                 prior_review_manifest=str((healthy_base/'pose_input_candidates.json').relative_to(ROOT)),
                 prior_verification='audit/fullbody_locomotion_20261114/expansion_verification.json')
        row.update(evidence)
        row.update(prior_clip_path=c['path'],path=row['video'],medical_status='not_individually_verified',eligible_as_correct_form_target=False)
        for key in ['poster','qc_sheet']:
            if c.get(key): row[key]=str(healthy_base/c[key])
        if extra:
            row.update(visual_review=extra,review_status='accepted_rechecked_visual_review',short_clip=extra['short_clip'],
                contains_turns_or_transitions=extra['contains_turns_or_transitions'])
            d=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=nb_read_frames,width,height','-of','json',str(dest)]))['streams'][0]
            assert int(d['nb_read_frames'])==c['frames'] and (d['width'],d['height'])==(c['width'],c['height'])
            row['decoded_frames']=int(d['nb_read_frames'])
        write(dest.with_suffix('.json'),row);rows.append(row)
    by_source=defaultdict(list)
    for c in rows:by_source[c.get('original')].append(c)
    rejected_sources={c['id']:c['reason'] for c in plan['rejected_sources']}
    source_results=[]
    for s in sources:
        kept=by_source[s['original']];intervals=sorted((c['source_first_frame'],c['source_end_frame_exclusive']) for c in kept)
        assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),(s['id'],intervals)
        excluded=[];cursor=0
        for a,b in intervals:
            if cursor<a:excluded.append([cursor,a])
            cursor=b
        if cursor<s['frames']:excluded.append([cursor,s['frames']])
        candidates=[c for c in plan['clips'] if c['id']==s['id']]
        source_results.append(dict(s,accepted_clip_ids=[c['clip_id'] for c in kept],
             outcome='retained_intervals' if kept else 'excluded_from_clean_set',
             exclusion_reason=None if kept else rejected_sources.get(s['id'],'Candidates failed visual full-body/occlusion/duplicate review.'),
             reviewed_candidates=[dict(clip_id=c['clip_id'],decision=review[c['clip_id']]) for c in candidates],
             not_retained_native_frame_intervals=excluded,
             not_retained_interval_note='Complement of retained intervals; may include usable material omitted by conservative selection. Not a clinical abnormality label.'))
    stats={}
    for cohort,source in sorted(set((c['cohort'],c['source']) for c in rows)):
        selected=[c for c in rows if (c['cohort'],c['source'])==(cohort,source)]
        stats[cohort+'/'+source]=dict(clips=len(selected),source_videos=len({c['original'] for c in selected}),
          seconds=round(sum(c['clip_duration_s'] for c in selected),3),frames=sum(c['clip_frames'] for c in selected),
          activities=dict(Counter(c['activity'] for c in selected)),environments=dict(Counter(c['environment'] for c in selected)),
          known_subjects=len({c['subject'] for c in selected if c.get('subject')}))
    rejected_candidates=[dict(clip_id=c['clip_id'],original=c['original'],source_first_frame=c['source_first_frame'],source_end_frame_exclusive=c['source_end_frame_exclusive'],**review[c['clip_id']]) for c in plan['clips'] if review[c['clip_id']]['status']=='reject']
    for c in plan['rejected_candidates']:rejected_candidates.append(c)
    summary=dict(status='complete',sources_screened=len(sources),new_source_groups=dict(Counter(s['source'] for s in sources)),
       clips=len(rows),statistics=stats,
       new_sources_with_retained_clips=dict(Counter(s['source'] for s in source_results if s['accepted_clip_ids'])),
       new_sources_without_retained_clips=dict(Counter(s['source'] for s in source_results if not s['accepted_clip_ids'])),
       rejected_candidates=len(rejected_candidates),exact_duplicate_exports_removed=len(duplicates),
       existing_clips_held_for_unverified_cohort=len(unverified),
       without_amputation_sources=len({c['source'] for c in rows if c['cohort']=='without_amputation'}),
       without_amputation_evidence_counts=dict(Counter(c['cohort_evidence_level'] for c in rows if c['cohort']=='without_amputation')),
       originals_preserved_and_sha256_verified=True,all_frames_visibility_certified=False,
       note='Full-body criteria concern footage usability, not gait normality. New full-rate pose inference was not run; ProGait publisher poses are included.')
    policy=['No diagnosis or automatic good/bad-form labels.','Keep each participant and all views of each physical trial in one train/test partition.','ProGait has only four participants; many clips do not imply many independent people.','Additional footage identities are unresolved: keep the complete additional_input group together until identities are established.','Face privacy masks, physical aids, natural self-occlusion, and existing watermarks are retained.','ProGait subset name outside can denote hallway walking indoors; environment is overground, not necessarily outdoors.']
    write(OUT/'manifest.json',dict(version=1,summary=summary,policy=policy,clips=rows))
    write(OUT/'source_decisions.json',source_results)
    write(OUT/'rejected_candidates.json',dict(clips=rejected_candidates,exact_duplicate_exports=duplicates))
    write(OUT/'cohort_unverified_existing_clips.json',unverified)
    confirmation_pending=[dict(clip_id=c['clip_id'],source=c['source'],video=c['video'],cohort_basis=c['cohort_basis']) for c in rows if c.get('needs_cohort_confirmation')]
    write(OUT/'cohort_confirmation_pending.json',confirmation_pending)
    if recheck['clips']:write(OUT/'cohort_recheck.json',recheck)
    write(OUT/'summary.json',summary)
    queue=[dict(clip_id=c['clip_id'],video=c['video'],video_absolute=c['video_absolute'],cohort=c['cohort'],source=c['source'],activity=c['activity'],environment=c['environment'],view=c.get('view','mixed/unspecified'),subject=c.get('subject'),split_group=c['split_group'],frames=c['clip_frames'],fps=c['fps'],width=c['width'],height=c['height'],source_first_frame=c['source_first_frame'],source_end_frame_exclusive=c['source_end_frame_exclusive'],publisher_pose=c.get('publisher_pose'),clinical_form_label=None) for c in rows]
    for q,c in zip(queue,rows):
        for key in ['cohort_evidence_level','needs_cohort_confirmation','medical_status','role','short_clip','contains_turns_or_transitions']:
            if key in c:q[key]=c[key]
        q['eligible_as_correct_form_target']=False
    write(OUT/'pose_input_queue.json',dict(base_directory=str(OUT),clips=queue))
    text='THEIA — curated locomotion clips\n\n'+json.dumps(summary,indent=2)+'\n\n'
    text+='Folders: '+ '; '.join(sorted(stats))+'.\nOnly accepted clips are in these folders. Source originals remain unchanged.\n\n'
    text+='Each MP4 has a JSON sidecar with source frame range [start, end), hashes, activity and selection evidence. Videos retain native image dimensions and cadence; cuts are temporal only, with audio removed.\n\n'
    text+='pose_input_queue.json lists all prepared inputs. annotations/progait contains the publisher 23-joint poses sliced to each clip: keypoints [N,23,3] = native pixel x,y,confidence, plus exact source/clip frame indices. These are existing annotations, not a new Sapiens run. Additional videos received 3 Hz Keypoint R-CNN quality screening and ProGait received 1 Hz screening plus native-frame annotation geometry checks.\n\n'
    text+='New candidate clips were visually sampled at five frames including boundaries, with denser source-scene review for additional footage. This is conservative quality selection, not certification that every joint is visible in every frame. Head/feet cropping, substantial foreground occlusion, body-masking blur, non-locomotion scenes and a confirmed repeated sequence were excluded. Natural self-occlusion and walking aids remain.\n\n'
    text+='Clinical form labels remain empty: a healthy cohort does not prove ideal form, and atypical prosthetic gait is valuable data. LJMU and LBMC have publisher-reported healthy-cohort support. MoVi and MPI are included as provisional visual comparison candidates: absence of a visible prosthesis does not verify amputation history or medical health. These '+str(len(confirmation_pending))+' clips are explicitly flagged needs_cohort_confirmation=true in the manifest and pose queue; see cohort_confirmation_pending.json. None is an approved correct-form training target.\n\n'
    text+='LBMC adds eight camera views of the SAME participant/trial, not eight independent participants. MoVi retains walking, jogging and running in place as separate activities, including two short 1.17-second examples marked short_clip. MPI is a walking transition with gestures, for viewpoint testing rather than steady-gait measurement. Running in place and transitions must not be pooled with steady forward locomotion.\n\n'
    text+='Cohort recheck primary sources: LBMC https://pmc.ncbi.nlm.nih.gov/articles/PMC12477944/ (Limitations explicitly describes healthy participants); MoVi https://pmc.ncbi.nlm.nih.gov/articles/PMC8211257/ ; MPI https://vcai.mpi-inf.mpg.de/3dhp-dataset/ . Source rights remain in source_catalog.json; the additions are local research copies, not a new redistribution license.\n\n'
    text+='Originals are indexed in source_decisions.json; rejected_candidates.json records interval-level exclusions. Audit evidence is in '+str(AUDIT)+'.\n\n'
    text+='\n'.join(policy)+'\n\nProGait: Xiangyu Yin et al., ICCV 2025, https://huggingface.co/datasets/ericyxy98/ProGait ; CC BY-NC-SA 4.0. Publisher source labels and privacy masking are preserved. Other source rights are inherited; this local curation grants no new redistribution rights.\n'
    (OUT/'README.txt').write_text(text)
    cohort_text='WITHOUT AMPUTATION — reference and provisional comparison clips\n\n'
    cohort_text+='This folder groups publisher-supported healthy-cohort clips together with explicitly provisional visual comparison clips. It is not a verified medical-history label for every participant.\n\n'
    for key,counts in stats.items():
        if key.startswith('without_amputation/'):
            cohort_text+=key.split('/')[1]+': '+str(counts['clips'])+' clips; '+str(counts['known_subjects'])+' participants.\n'
    cohort_text+='\nLJMU + LBMC: publisher-reported healthy cohorts; correct gait form remains unvalidated.\nMoVi + MPI: visual comparison candidates, needs_cohort_confirmation=true; medical/amputation history remains unverified. These should not be treated as confirmed negative medical labels.\n\n'
    cohort_text+='Use the parent pose_input_queue.json and per-video JSON sidecars. Walking, jogging, running in place, treadmill/overground and transitions have separate labels. All views of one participant belong in the same train/test split. LBMC is eight views of one person/trial. Two MoVi clips are only 1.17 seconds long.\n'
    (OUT/'without_amputation/README.txt').write_text(cohort_text)
    shutil.copy2(ROOT/'data/raw/ProGait/README.md',OUT/'ProGait_SOURCE_README.md')
    catalog=json.loads((healthy_base/'source_catalog.json').read_text())
    for source in catalog['sources']:
        if source['id'] in recheck['sources']:
            source['curated_cohort_recheck']=recheck['sources'][source['id']]
            if source['id']=='lbmc':
                source['health_evidence']=recheck['sources']['lbmc']['cohort_basis']
                source['limitations']='Public anonymized preview of one participant. Primary paper reports a healthy cohort; individual screening records and correct-form labels are unavailable. Preserve publisher anonymization and physical markers; camera views are not independent participants.'
    write(OUT/'source_catalog.json',catalog)
    # Keep generated rejected/superseded candidates out of the deliverable root.
    staging=OUT/'staging';quarantine=AUDIT/'generated_candidates_not_in_final';quarantine.mkdir(exist_ok=True)
    if staging.exists():
        for p in staging.iterdir():
            dest=quarantine/p.name
            if dest.exists():assert sha(p)==sha(dest);p.unlink()
            else:p.replace(dest)
        staging.rmdir()
    write(AUDIT/'finalization.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

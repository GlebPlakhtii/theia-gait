"""Apply frozen matching/QC and aggregate repeated clips without pseudoreplication."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from gait_geometry import LABELS
from build_gait_review_highlights import build as build_review_highlights

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gait_metrics_20261114'


def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(path)


def numeric(value):
    return isinstance(value,(float,int)) and np.isfinite(value)


def clip_value(record, outcome, rules):
    """No cross-view, transition, missing-value or low-coverage contribution."""
    if record.get('status')!='observations_extracted':return None,['pose_pending'],[]
    review=record.get('pose_visual_review') or {}
    regional_failure=(review.get('status')=='regional_pose_failure'
        and review.get('weight_sha256')==record.get('pose',{}).get('weight_sha256'))
    lower_failure=(review.get('status')=='regional_pose_failure'
        and review.get('weight_sha256')==record.get('pose',{}).get('weight_sha256')
        and review.get('normalization_scale_unreliable',bool(set(review.get('affected_metric_ids',[])) & {'M04','M05','M06','M07','M08','M09','M10','M11','M12','M13','M14'})))
    if lower_failure and outcome.get('unit')=='L0':
        return None,['leg_scale_unreliable_after_visual_pose_failure'],[]
    samples=[];excluded=[]
    for segment in record['segments']:
        why=[]
        duration=(segment['end_clip_frame_exclusive']-segment['first_clip_frame'])/record['fps']
        if segment['view']!=outcome['view']:why.append('view_incompatible')
        if duration<rules['minimum_segment_duration_s']:why.append('segment_too_short')
        if (segment.get('fixed_L0_pixels') or 0)<rules['minimum_fixed_L0_pixels']:why.append('projected_body_scale_too_small')
        if segment.get('quality_flags'):why.append('segment_quality_flag')
        metric=next(m for m in segment['metrics'] if m['metric_id']==outcome['family'])
        if metric['measurement_status']=='not_applicable':why.append('outside_straight_walking_scope')
        if regional_failure and outcome['family'] in review.get('affected_metric_ids',[]):
            why.append(review.get('reason_code','visual_pose_localization_failure'))
        elif 'visual_pose_localization_failure' in metric.get('reason_codes',[]):why.append('visual_pose_localization_failure')
        source,key,statistic=outcome['source'].split(':')
        value=None;coverage=0
        if source=='observation':
            component=metric.get('observations',{}).get(key)
            if component:coverage=component['valid_fraction'];value=component.get(statistic)
        else:
            component=segment.get('paired_geometry',{}).get(key)
            if component:coverage=component['valid_fraction'];value=component.get(statistic)
        if coverage<rules['minimum_component_valid_fraction']:why.append('component_coverage_too_low')
        if not numeric(value):why.append('numeric_observation_unavailable')
        if why:excluded.append(dict(segment=segment['segment_id'],reasons=why));continue
        samples.append(dict(value=value,duration_s=duration,segment=segment['segment_id'],valid_fraction=coverage))
    if not samples:return None,sorted({r for e in excluded for r in e['reasons']}) or ['no_segments'],excluded
    duration=sum(s['duration_s'] for s in samples)
    if duration/record['clip_duration_s']<rules['primary_minimum_source_clip_retained_duration_fraction']:
        return None,['insufficient_eligible_duration_fraction'],excluded
    return dict(value=float(np.average([s['value'] for s in samples],weights=[s['duration_s'] for s in samples])),
                retained_duration_s=duration,retained_duration_fraction=duration/record['clip_duration_s'],segments=samples),[],excluded


def aggregate_units(rows):
    """Each trial/camera block first, then each known participant, equal weight."""
    trials=defaultdict(list)
    for row in rows:trials[(row['subject'],row['trial_block'])].append(row)
    participants=defaultdict(list)
    for (subject,trial),clips in sorted(trials.items(),key=str):
        unit=dict(value=float(np.median([c['value'] for c in clips])),trial=trial,
                  clips=sorted({c['clip_id'] for c in clips}),sources=sorted({c['source'] for c in clips}))
        participants[('participant:'+subject) if subject else trial].append(unit)
    units=[]
    for key,blocks in sorted(participants.items()):
        units.append(dict(unit_id=key,unit_type='participant' if key.startswith('participant:') else 'source_file_identity_unresolved',
                          value=float(np.median([b['value'] for b in blocks])),trial_blocks=len(blocks),
                          clips=sorted({c for b in blocks for c in b['clips']}),sources=sorted({s for b in blocks for s in b['sources']})))
    return units


def summarize(rows):
    units=aggregate_units(rows);values=[u['value'] for u in units]
    known=all(u['unit_type']=='participant' for u in units)
    return dict(unit_count=len(units),participant_count=len(units) if known else None,
                known_participant_count=sum(u['unit_type']=='participant' for u in units),
                unknown_identity_source_blocks=sum(u['unit_type']!='participant' for u in units),
                clip_count=len({r['clip_id'] for r in rows}),trial_block_count=len({r['trial_block'] for r in rows}),
                source_counts=dict(Counter(r['source'] for r in rows)),
                median=float(np.median(values)) if values else None,
                q25=float(np.percentile(values,25)) if values else None,
                q75=float(np.percentile(values,75)) if values else None,units=units)


def interval(a,b,protocol):
    if a['participant_count'] is None or b['participant_count'] is None:
        return dict(bounds=None,reason='Participant identities unresolved; source files are not independent participants.')
    minimum=protocol['minimum_known_subjects_each_group']
    if min(a['participant_count'],b['participant_count'])<minimum:
        return dict(bounds=None,reason=f'Fewer than {minimum} known participants in at least one group; descriptive results only.')
    rng=np.random.default_rng(protocol['random_seed']);av=np.array([u['value'] for u in a['units']]);bv=np.array([u['value'] for u in b['units']])
    n=protocol['bootstrap_resamples']
    delta=np.median(rng.choice(av,(n,len(av)),replace=True),axis=1)-np.median(rng.choice(bv,(n,len(bv)),replace=True),axis=1)
    return dict(bounds=np.percentile(delta,[2.5,97.5]).tolist(),method='Participant bootstrap of difference in group medians',
                resamples=n,confidence_level=.95,reason='Sampling variability only; does not quantify measurement/calibration error.')


def availability(records):
    matrix=[];segment_rows=[]
    for r in records:
        row=dict(clip_id=r['clip_id'],source=r['source'],cohort=r['cohort'],status=r['status'],families=[])
        segments=r.get('segments') or [dict(segment_id=None,metrics=r['metrics'])]
        for i in range(1,18):
            mid=f'M{i:02d}';items=[]
            for s in segments:
                m=next(m for m in s['metrics'] if m['metric_id']==mid);items.append(m)
                segment_rows.append(dict(clip_id=r['clip_id'],segment_id=s['segment_id'],metric_id=mid,
                    measurement_status=m['measurement_status'],detection_status=m['detection_status'],
                    observation_units=m.get('observation_units',{}),numeric_observation_fields=list(m.get('observations',{})),
                    reason_codes=m['reason_codes']))
            row['families'].append(dict(metric_id=mid,states=dict(Counter(m['measurement_status'] for m in items)),
                numeric_observation_segments=sum(bool(m.get('observations')) for m in items),
                reason_codes=sorted({x for m in items for x in m['reason_codes']})))
        matrix.append(row)
    write(OUT/'metric_availability.json',dict(clips=len(records),metric_families=17,rows=matrix))
    write(OUT/'metric_availability_segments.json',dict(rows=segment_rows))
    return dict(Counter(x['measurement_status'] for x in segment_rows))


def supplementary(frozen,records,harmonized):
    path=OUT/'extended_outcomes_frozen.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==(OUT/'extended_outcomes_frozen.sha256').read_text().strip()
    definitions=json.loads(path.read_text());results=[];exclusions=[];values=[]
    for mode in frozen['modes']:
        candidates=[r for r in frozen['rows'] if mode in r['comparison_modes']]
        for outcome in definitions['outcomes']:
            relevant=[r for r in candidates if r['view']==outcome['view']]
            environments=sorted({r['environment'] for r in relevant})
            for environment in environments:
                selected=[r for r in relevant if r['environment']==environment];rows=[]
                for row in selected:
                    record=harmonized[row['clip_id']] if row['same_model_subset'] else records[row['clip_id']]
                    if outcome.get('free_arm_only') and row['support_context'] not in ('none','no_aid_recorded_in_curated_reference'):
                        estimate,reasons=None,['guarded_or_supported_arms']
                    elif record.get('pose') and record['pose'].get('weight_sha256')!='2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f':
                        estimate,reasons=None,['pose_method_mismatch']
                    else:estimate,reasons,_=clip_value(record,outcome,frozen['quality_rules'])
                    if estimate is None:
                        exclusions.append(dict(mode=mode,outcome=outcome['id'],clip_id=row['clip_id'],reasons=reasons));continue
                    item=dict(**row,**estimate,mode=mode,outcome=outcome['id'],unit=outcome['unit'])
                    rows.append(item);values.append(item)
                groups={g:summarize([r for r in rows if r['cohort']==g]) for g in ['with_amputation','without_amputation']}
                a,b=[groups[g] for g in ['with_amputation','without_amputation']];matched=bool(a['unit_count'] and b['unit_count'])
                results.append(dict(mode=mode,environment=environment,view=outcome['view'],outcome=outcome,groups=groups,
                    status='descriptive_comparison_available' if matched else 'not_available',
                    median_difference_with_minus_without=a['median']-b['median'] if matched else None,
                    interpretation='Supplementary projected variation only. Parent clinical metric, anatomical axes and measurement error remain unvalidated.',
                    uncertainty=interval(a,b,frozen['uncertainty']) if matched else dict(bounds=None,reason='Both quality-eligible groups are not present.')))
    write(OUT/'extended_observation_clip_values.json',dict(rows=values))
    write(OUT/'extended_observation_exclusions.json',dict(rows=exclusions))
    result=dict(registration=definitions['registration'],selection_rule=definitions['selection_rule'],
                definition_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),results=results)
    write(OUT/'extended_observation_results.json',result)
    return result


def family_coverage(records,primary,extended,harmonized):
    reasons={
        1:'Phase-specific pelvic hiking/drop unavailable; hip-landmark line ROM is only a raw proxy, not an ASIS-based pelvic angle.',
        2:'Projected frontal trunk changes only; camera roll and anatomical axes are uncalibrated.',
        3:'Projected side trunk changes only; absolute forward lean relative to gravity is uncalibrated.',
        4:'Swing phase is unvalidated; raw ankle-to-hip lateral observations do not establish swing circumduction.',
        5:'Projected side knee bend only; mechanical prosthetic joint correspondence and anatomical accuracy are unverified.',
        6:'Signed anatomical axis and prosthetic geometry are unverified; no hyperextension or rapid-extension classification.',
        7:'Ground plane, flat-foot baseline and midstance are unavailable; no validated vaulting measurement.',
        8:'Ground plane and contact events are unvalidated; no heel/toe/flat contact classification.',
        9:'Ground-plane and sole-clearance references are unavailable; no metric ground clearance.',
        10:'Ground homography and anatomical axes are unavailable; no foot-progression/ankle angle.',
        11:'Validated contact events are unavailable; no stance or step-time asymmetry.',
        12:'Contact events and ground scale are unavailable; no calibrated step length or step-to classification.',
        13:'Double-support phases are unvalidated; raw frontal ankle separation is not a validated base-of-support measurement.',
        14:'No validated gait cycles; no gait-cycle irregularity metric. Temporal signal SD is kept distinct.',
        15:'Shoulder-landmark line and shoulder-to-hip changes only; phase-specific hiking is unvalidated.',
        16:'Projected wrist-to-shoulder variation only; supported/guarded arms are excluded from free-arm comparisons.',
        17:'Projected ear-to-shoulder variation only; head rotation and clinical forward-head posture remain unverified.'}
    rows=[]
    for i,label in enumerate(LABELS,1):
        mid=f'M{i:02d}'
        def counts(items):
            numeric_counts=Counter();proxy_counts=Counter()
            for r in items:
                if r['status']!='observations_extracted':continue
                metrics=[next(m for m in s['metrics'] if m['metric_id']==mid) for s in r['segments']]
                key=r['pose']['method']+':'+r['cohort']
                if any(m['measurement_status']=='uncertain' and m['value'] is not None for m in metrics):numeric_counts[key]+=1
                if any(m.get('observations') for m in metrics):proxy_counts[key]+=1
            return dict(numeric_counts),dict(proxy_counts)
        numeric_counts,proxy_counts=counts(records.values())
        harmonized_numeric,harmonized_proxy=counts(harmonized.values())
        matches=[r for r in primary if r['outcome']['family']==mid and r['sensitivity']=='base' and r['status']=='descriptive_comparison_available']
        secondary=[r for r in extended['results'] if r['outcome']['family']==mid and r['status']=='descriptive_comparison_available']
        rows.append(dict(metric_id=mid,name=label,numeric_projected_clips_by_method_and_cohort=numeric_counts,
            raw_observation_clips_by_method_and_cohort=proxy_counts,
            harmonized_numeric_projected_clips_by_method_and_cohort=harmonized_numeric,
            harmonized_raw_observation_clips_by_method_and_cohort=harmonized_proxy,
            harmonized_count_scope='Repeated measurement of the predefined ProGait subset, not additional unique input clips or participants.',
            primary_comparison_rows=len(matches),supplementary_comparison_rows=len(secondary),
            clinical_detection_available=False,interpretation=reasons[i]))
    write(OUT/'metric_comparison_coverage.json',dict(rows=rows))
    return rows


def main():
    frozen_path=OUT/'comparison_manifest_frozen.json'
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()==(OUT/'comparison_manifest_frozen.sha256').read_text().strip()
    frozen=json.loads(frozen_path.read_text())
    catalog=json.loads((OUT/'catalog.json').read_text())
    records={c['clip_id']:json.loads((OUT/'clips'/c['clip_id']/'metrics.json').read_text()) for c in catalog['clips']}
    hcat=json.loads((OUT/'harmonized/catalog.json').read_text())
    harmonized={c['clip_id']:json.loads((OUT/'harmonized/clips'/c['clip_id']/'metrics.json').read_text()) for c in hcat['clips']}
    counts=availability(list(records.values()))
    results=[];clip_rows=[];quality_exclusions=[]
    row_index={r['clip_id']:r for r in frozen['rows']}
    sensitivities=['base','strict_coverage','static_camera_only']
    for sensitivity in sensitivities:
        rules=dict(frozen['quality_rules'])
        if sensitivity=='strict_coverage':rules.update(minimum_component_valid_fraction=.9,minimum_fixed_L0_pixels=120)
        for mode in frozen['modes']:
            candidates=[r for r in frozen['rows'] if mode in r['comparison_modes']]
            for outcome in frozen['outcomes']:
                buckets=defaultdict(list)
                for row in candidates:
                    if row['view']!=outcome['view']:continue
                    record=harmonized[row['clip_id']] if row['same_model_subset'] else records[row['clip_id']]
                    context=row.get('context_review') or {}
                    why=[]
                    if outcome.get('free_arm_only') and row['support_context'] not in ('none','no_aid_recorded_in_curated_reference'):
                        why.append('guarded_or_supported_arms')
                    if sensitivity=='static_camera_only' and context.get('camera_motion') in ('moving','panning'):
                        why.append('moving_or_panning_camera')
                    if record.get('pose') and record['pose'].get('weight_sha256')!='2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f':
                        why.append('pose_method_mismatch')
                    estimate,reasons,segment_exclusions=clip_value(record,outcome,rules) if not why else (None,why,[])
                    key=dict(clip_id=row['clip_id'],mode=mode,sensitivity=sensitivity,outcome=outcome['id'])
                    if estimate is None:
                        quality_exclusions.append(dict(**key,reasons=reasons,segment_exclusions=segment_exclusions));continue
                    item=dict(**row,**estimate,**{k:v for k,v in key.items() if k!='clip_id'},unit=outcome['unit'])
                    clip_rows.append(item)
                    buckets[(row['activity_group'],row['environment'],row['view'],row['pose_method'])].append(item)
                # Include empty compatible strata; do not silently drop failures to match.
                keys=set(buckets)|{(r['activity_group'],r['environment'],r['view'],r['pose_method']) for r in candidates if r['view']==outcome['view']}
                for stratum in sorted(keys):
                    rows=buckets.get(stratum,[]);groups={}
                    for group in ('with_amputation','without_amputation'):
                        groups[group]=summarize([r for r in rows if r['cohort']==group])
                    a,b=groups['with_amputation'],groups['without_amputation'];matched=bool(a['unit_count'] and b['unit_count'])
                    cell_candidates=[r for r in candidates if (r['activity_group'],r['environment'],r['view'],r['pose_method'])==stratum]
                    candidate_counts=dict(Counter(r['cohort'] for r in cell_candidates))
                    pending_counts=dict(Counter(r['cohort'] for r in cell_candidates if
                        (harmonized[r['clip_id']] if r['same_model_subset'] else records[r['clip_id']])['status']!='observations_extracted'))
                    if matched:status='descriptive_comparison_available';unmatched_reason=None
                    elif not all(candidate_counts.get(g,0) for g in groups):
                        status='metadata_unmatched';unmatched_reason='No compatible metadata candidates in at least one cohort for this task/environment/view/model.'
                    elif any(pending_counts.values()):
                        status='waiting_for_eligible_poses';unmatched_reason='Compatible candidates exist, but pose processing is still pending and both eligible groups are not yet present.'
                    else:
                        status='no_quality_eligible_group';unmatched_reason='Poses are available, but frozen QC or documented localization failures exclude all values in at least one group.'
                    results.append(dict(mode=mode,sensitivity=sensitivity,stratum=dict(zip(frozen['matching_fields'],stratum)),outcome=outcome,
                        status=status,unmatched_reason=unmatched_reason,metadata_candidate_counts=candidate_counts,pose_pending_counts=pending_counts,
                        groups=groups,median_difference_with_minus_without=a['median']-b['median'] if matched else None,
                        uncertainty=interval(a,b,frozen['uncertainty']) if matched else dict(bounds=None,reason=unmatched_reason),
                        clinical_interpretation='No clinical good/bad threshold; projected observations only.'))
    paired=[]
    for cid,hr in harmonized.items():
        row=row_index[cid];pr=records[cid]
        for outcome in frozen['outcomes']:
            if outcome['view']!=row['view']:continue
            p,preasons,_=clip_value(pr,outcome,frozen['quality_rules']);s,sreasons,_=clip_value(hr,outcome,frozen['quality_rules'])
            paired.append(dict(clip_id=cid,subject=row['subject'],trial_block=row['trial_block'],outcome=outcome['id'],unit=outcome['unit'],
                publisher_value=p['value'] if p else None,sapiens_value=s['value'] if s else None,
                sapiens_minus_publisher=s['value']-p['value'] if s and p else None,
                reasons=preasons+sreasons,scope='Same video and frozen QC rules, each estimator may retain different frames/segments; descriptive method sensitivity, not independent accuracy.'))
    strata=defaultdict(Counter)
    for r in frozen['rows']:
        strata[(r['activity_group'],r['environment'],r['view'],r['pose_method'])][r['cohort']]+=1
    strata=[dict(stratum=dict(zip(frozen['matching_fields'],k)),counts=dict(v),two_groups_present=len(v)==2) for k,v in sorted(strata.items())]
    method_summary=[]
    for outcome in frozen['outcomes']:
        pairs=[p for p in paired if p['outcome']==outcome['id'] and p['sapiens_minus_publisher'] is not None]
        values=[dict(p,value=p['sapiens_minus_publisher'],source='progait') for p in pairs]
        method_summary.append(dict(outcome=outcome,paired_clips=len(pairs),participant_summary=summarize(values),
            scope='Sapiens minus publisher; each estimator retains its own QC-passing frames. Not independent accuracy or a clinical error bound.'))
    supplementary_results=supplementary(frozen,records,harmonized)
    coverage=family_coverage(records,results,supplementary_results,harmonized)
    harmonization_status=[]
    for cid,record in harmonized.items():
        row=row_index[cid];review=record.get('pose_visual_review') or {}
        harmonization_status.append(dict(clip_id=cid,subject=row['subject'],view=row['view'],status=record['status'],
            pose_review_status=review.get('status'),affected_metric_ids=review.get('affected_metric_ids',[]),
            metadata_exclusions=row['metadata_exclusions'],no_primary_mode_reason=row['no_primary_mode_reason'],
            eligible_primary_outcomes=sorted({r['outcome'] for r in clip_rows if r['clip_id']==cid and r['sensitivity']=='base'}),
            primary_qc_exclusions=[{k:r[k] for k in ('mode','outcome','reasons','segment_exclusions')}
                for r in quality_exclusions if r['clip_id']==cid and r['sensitivity']=='base']))
    payload=dict(updated_at=datetime.now(timezone.utc).isoformat(),status='final' if not catalog['counts'].get('waiting_for_pose',0) and not hcat['counts'].get('waiting_for_pose',0) else 'interim',
        frozen_manifest_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),feature_version=catalog['version'],
        processing_counts=catalog['counts'],harmonization_counts=hcat['counts'],harmonization_status=harmonization_status,metric_family_record_counts=counts,
        modes=frozen['modes'],results=results,source_view_method_strata=strata,
        supplementary_observations=supplementary_results,metric_comparison_coverage=coverage,
        unavailable_comparisons=frozen['unavailable_comparisons'],paired_pose_method_sensitivity=paired,paired_pose_method_summary=method_summary,
        clinical_thresholds=None,causal_effect_estimate=False,
        limitations=['Sampling five native frames for context review cannot certify every intermediate frame.',
                     'ROI and nominal-view matching do not calibrate camera projection, gravity, or prosthetic joint centers.',
                     'Walking speeds, demographics and prosthetic components are not matched.',
                     'ROM95 and SD describe image-plane time series, not validated anatomical ROM or gait-cycle variability.',
                     'Short clips can include gait initiation or termination; sampled context review does not certify steady-state walking.',
                     'Source identity and method are displayed; datasets differ in recording and assistance conditions.'])
    write(OUT/'comparison_results.json',payload)
    write(OUT/'comparison_clip_values.json',dict(rows=clip_rows))
    write(OUT/'comparison_exclusions.json',dict(metadata=[dict(clip_id=r['clip_id'],reasons=r['metadata_exclusions'],no_primary_mode_reason=r['no_primary_mode_reason']) for r in frozen['rows']],quality=quality_exclusions))
    write(OUT/'comparison_tables.json',dict(rows=[dict(mode=r['mode'],sensitivity=r['sensitivity'],**r['stratum'],outcome=r['outcome']['id'],unit=r['outcome']['unit'],status=r['status'],
        with_median=r['groups']['with_amputation']['median'],with_q25=r['groups']['with_amputation']['q25'],with_q75=r['groups']['with_amputation']['q75'],with_units=r['groups']['with_amputation']['unit_count'],with_participants=r['groups']['with_amputation']['participant_count'],
        without_median=r['groups']['without_amputation']['median'],without_q25=r['groups']['without_amputation']['q25'],without_q75=r['groups']['without_amputation']['q75'],without_units=r['groups']['without_amputation']['unit_count'],without_participants=r['groups']['without_amputation']['participant_count'],
        median_difference=r['median_difference_with_minus_without']) for r in results]))
    build_review_highlights(OUT)
    print(json.dumps(dict(status=payload['status'],clip_rows=len(clip_rows),comparisons=sum(r['status']=='descriptive_comparison_available' for r in results),quality_exclusions=len(quality_exclusions))))


if __name__=='__main__':main()

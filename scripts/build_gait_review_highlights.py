"""Exploratory body-region review cues, separate from clinical classification."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gait_metrics_20261114'
DEFINITIONS={
    'trunk_frontal_rom95':('trunk','Trunk motion'),
    'trunk_side_rom95':('trunk','Trunk motion'),
    'hip_line_rom95_proxy':('hips','Hip-line motion proxy'),
    'shoulder_line_rom95':('shoulders','Shoulder motion'),
    'knee_rom95_mean':('knees','Projected knee range'),
    'knee_rom95_asymmetry':('knees','Left/right knee-range difference'),
    'knee_p95_mean':('knees','Projected knee bend'),
    'knee_temporal_sd_mean':('knees','Projected knee variability'),
    'arm_rom95_mean':('arms','Arm movement proxy'),
    'arm_rom95_asymmetry':('arms','Left/right arm-range difference'),
}


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def reference_fence(units, target_unit):
    """Leave the target participant out; fewer than five references abstains."""
    selected=[u for u in units if u['unit_id']!=target_unit and u['unit_type']=='participant']
    if len(selected)<5:return None
    values=np.array([u['value'] for u in selected],float)
    if not np.isfinite(values).all():return None
    q1,q3=np.percentile(values,[25,75]);iqr=q3-q1
    if iqr<=1e-9:return None
    return dict(q25=float(q1),q75=float(q3),lower=float(q1-1.5*iqr),upper=float(q3+1.5*iqr),
                participant_count=len(selected),participant_ids=[u['unit_id'] for u in selected])


def outside(value, fence):
    return value<fence['lower'] or value>fence['upper']


def build(root=OUT):
    root=Path(root)
    report=json.loads((root/'comparison_results.json').read_text())
    rows=json.loads((root/'comparison_clip_values.json').read_text())['rows']
    catalogs=[('',json.loads((root/'catalog.json').read_text())),
              ('harmonized/',json.loads((root/'harmonized/catalog.json').read_text()))]
    rois=json.loads((root/'production_rois.json').read_text())['clips']
    by_clip={};catalog=[]
    for prefix,cat in catalogs:
        for item in cat['clips']:
            key=prefix+item['clip_id'];by_clip[key]=dict(item=item,prefix=prefix,events=[],evaluated=[])
    candidate_rows={(r['mode'],r['outcome'],r['clip_id']):r for r in rows if r['sensitivity']=='base'}
    cached={}
    def record(key):
        if key not in cached:
            c=by_clip[key];cached[key]=json.loads((root/c['prefix']/'clips'/c['item']['clip_id']/'metrics.json').read_text())
        return cached[key]
    for comparison in report['results']:
        oid=comparison['outcome']['id']
        if comparison['sensitivity']!='base' or oid not in DEFINITIONS:continue
        reference=comparison['groups']['without_amputation']['units']
        region,title=DEFINITIONS[oid]
        for cohort,group in comparison['groups'].items():
            for unit in group['units']:
                fence=reference_fence(reference,unit['unit_id'])
                for cid in unit['clips']:
                    row=candidate_rows.get((comparison['mode'],oid,cid))
                    if row is None:continue
                    key=('harmonized/' if row['same_model_subset'] else '')+cid
                    target=by_clip[key]
                    if fence is None:continue
                    target['evaluated'].append(oid)
                    # Both the balanced participant/source unit and the source clip
                    # must exceed the same exploratory reference fence.
                    if not outside(unit['value'],fence) or not outside(row['value'],fence):continue
                    rec=record(key)
                    for segment_value in row['segments']:
                        if not outside(segment_value['value'],fence):continue
                        seg=next(s for s in rec['segments'] if s['segment_id']==segment_value['segment'])
                        metric=next(m for m in seg['metrics'] if m['metric_id']==comparison['outcome']['family'])
                        if set(metric.get('reason_codes',[])) & {'visual_pose_localization_failure','raw_observations_retained_for_debug_not_comparison'}:continue
                        event=dict(kind='movement_review',region=region,title=title,outcome=oid,
                            start_time_s=seg['start_time_s'],end_time_s=seg['end_time_s']+1/rec['fps'],
                            first_clip_frame=seg['first_clip_frame'],end_clip_frame_exclusive=seg['end_clip_frame_exclusive'],
                            segment_id=seg['segment_id'],segment_value=segment_value['value'],
                            clip_value=row['value'],unit_value=unit['value'],unit_type=unit['unit_type'],
                            target_unit_id=unit['unit_id'],environment=comparison['stratum']['environment'],
                            unit=comparison['outcome']['unit'],reference=fence,
                            comparison_mode=comparison['mode'],view=seg['view'],
                            measured_above_or_below='above' if segment_value['value']>fence['upper'] else 'below',
                            clinical_status='not_validated',severity=None,correction=None,
                            reason='Segment motion lies outside a provisional reference fence; review the video and tracking before interpretation.',
                            temporal_scope='The whole measured segment is highlighted. This does not locate an instantaneous gait error or gait phase.',
                            context_limitation=comparison.get('clinical_interpretation','Projected observations only.'))
                        context=row.get('context_review') or {}
                        event['context_warnings']=[]
                        if context.get('camera_motion') in ('moving','panning','handheld'):
                            event['context_warnings'].append('Camera motion is present; camera effects can contribute to the highlighted signal.')
                        if row['support_context'] not in ('none','no_aid_recorded_in_curated_reference'):
                            event['context_warnings'].append('Guarding or support differs from the reference; assistance can influence movement.')
                        # Reference controls appear in both comparison modes; one cue suffices.
                        identity=(oid,seg['segment_id'])
                        if not any((e.get('outcome'),e.get('segment_id'))==identity for e in target['events']):target['events'].append(event)
    for key,target in by_clip.items():
        item=target['item'];review=item.get('pose_visual_review') or {}
        if review.get('status')=='regional_pose_failure' and review.get('weight_sha256')==item.get('pose',{}).get('weight_sha256'):
            evidence_frames=review.get('failure_clip_frames',review['inspected_clip_frames'])
            with np.load(root/target['prefix']/'clips'/item['clip_id']/'observations.npz') as observations:
                evidence_times=[float(observations['timestamp_s'][i]) for i in evidence_frames]
            regions=review.get('affected_body_regions') or [review.get('affected_body_region','lower_body')]
            for region in dict.fromkeys(regions):
                target['events'].append(dict(kind='pose_quality',region=region,title=review.get('quality_label','Leg tracking unreliable'),
                    start_time_s=0,end_time_s=item['clip_duration_s'],first_clip_frame=0,
                    end_clip_frame_exclusive=item['clip_frames'],reason=review['evidence'],
                    evidence_frames=evidence_frames,evidence_time_s=evidence_times,clinical_status='not_a_movement_judgment',
                    severity=None,correction=None,
                    temporal_scope='Documented regional quality limitation; conservative region warning for the whole clip.'))
        if review.get('manual_visibility_masks') and review.get('weight_sha256')==item.get('pose',{}).get('weight_sha256'):
            with np.load(root/target['prefix']/'clips'/item['clip_id']/'observations.npz',allow_pickle=False) as observations:
                timestamps=observations['timestamp_s']
                for mask in review['manual_visibility_masks']:
                    lo,hi=mask['first_clip_frame'],mask['end_clip_frame_exclusive']
                    target['events'].append(dict(kind='pose_quality',quality_basis='manual_visibility_mask',
                        region=mask.get('body_region','lower_body'),title=mask.get('title','Landmarks occluded'),
                        start_time_s=float(timestamps[lo]),end_time_s=min(item['clip_duration_s'],float(timestamps[hi-1])+1/item['fps']),
                        first_clip_frame=lo,end_clip_frame_exclusive=hi,masked_joint_names=mask['joint_names'],
                        reason=mask['reason'],evidence_frames=[lo,hi-1],evidence_time_s=[float(timestamps[lo]),float(timestamps[hi-1])],
                        clinical_status='not_a_movement_judgment',severity=None,correction=None,
                        temporal_scope='Only the documented occlusion interval is highlighted. Masked landmarks do not contribute to measurements.'))
        target['events'].sort(key=lambda e:(e['start_time_s'],e['kind'],e['region']))
        quality=any(e['kind']=='pose_quality' for e in target['events'])
        payload=dict(clip_id=item['clip_id'],pose_method=item.get('pose',{}).get('method'),
            updated_at=report['updated_at'],protocol_version='exploratory-region-review-1',
            evaluated_outcomes=sorted(set(target['evaluated'])),events=target['events'],
            target_roi=rois.get(item['clip_id']) if quality else None,
            status='review_flags_available' if target['events'] else 'no_flag_in_evaluated_outcomes' if target['evaluated'] else 'insufficient_comparable_reference',
            clinical_classification=None,
            note='Amber = exploratory movement review; purple = pose tracking unreliable. No highlight does not mean correct form. Segment-level cues are not precise error instants.')
        write(root/target['prefix']/'clips'/item['clip_id']/'review_highlights.json',payload)
        catalog.append(dict(key=key,clip_id=item['clip_id'],method='sapiens' if target['prefix'] or item.get('pose',{}).get('method')=='sapiens_inference' else 'publisher',
            status=payload['status'],movement_events=sum(e['kind']=='movement_review' for e in target['events']),
            pose_quality_events=sum(e['kind']=='pose_quality' for e in target['events']),regions=sorted({e['region']for e in target['events']})))
    protocol=dict(version='exploratory-region-review-1',registration='Post-hoc visualization aid; separate from frozen primary outcomes and clinical labels.',
        reference='Same-model, same view/task/environment, base-QC healthy-reference participant summaries. Assistance and camera differences remain.',
        rule='Reference Q1 − 1.5×IQR and Q3 + 1.5×IQR; minimum five known reference participants; leave the target reference participant out.',
        support='Balanced unit, clip and eligible segment must all exceed the exploratory fence. No thresholds are chosen to maximize group separation.',
        time_scope='Highlights cover measured segments; they do not identify exact errors or validated contact phases.',
        source='https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm',
        clinical_threshold=False,correction_recommendation=False,
        limitations=['Fences are descriptive statistical screening, not clinical validation.',
                     'Small, incomplete, source-confounded reference set; flags may change as processing completes.',
                     'A publisher-reported healthy participant can be flagged; an unflagged participant is not certified normal.'])
    write(root/'review_highlights_protocol.json',protocol)
    write(root/'review_highlights_catalog.json',dict(updated_at=report['updated_at'],clips=catalog))
    result=dict(clips=len(catalog),with_movement_review=sum(c['movement_events']>0 for c in catalog),with_pose_quality_review=sum(c['pose_quality_events']>0 for c in catalog))
    print('Body-region review cues: '+json.dumps(result));return result


if __name__=='__main__':build()

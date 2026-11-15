"""Audit native-frame provenance and the 17-family contract, including pending jobs."""
from collections import Counter
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from gait_geometry import NAMES
from compare_gait_groups import write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gait_metrics_20261114'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit_visual_exclusion_contract(record):
    """Check exported values themselves, independently of comparison gating."""
    review=record.get('pose_visual_review') or {}
    if (review.get('status')!='regional_pose_failure'
        or not review.get('weight_sha256')
        or review['weight_sha256']!=(record.get('pose') or {}).get('weight_sha256')):
        return 0
    affected=set(review.get('affected_metric_ids',[]))
    bad_scale=review.get('normalization_scale_unreliable',bool(affected & {f'M{i:02d}' for i in range(4,15)}))
    groups=[s['metrics'] for s in record['segments']] if record['segments'] else [record['metrics']]
    checked=0
    for metrics in groups:
        for metric in metrics:
            if metric['metric_id'] in affected:
                assert metric['value'] is None,'Reviewed regional exclusion still exports a usable value'
                assert metric['measurement_status']=='not_evaluable','Reviewed regional exclusion is not marked unavailable'
                reasons=metric['reason_codes']
                assert review['reason_code'] in reasons and 'raw_observations_retained_for_debug_not_comparison' in reasons,'Reviewed exclusion lacks its quality/debug-only reasons'
                checked+=1
            if bad_scale:
                normalized={name for name,unit in metric.get('observation_units',{}).items() if unit=='L0'}
                assert normalized<=set(metric.get('excluded_observation_components',{})),'Unreliable scale lacks explicit component exclusions'
                if isinstance(metric['value'],dict):
                    assert not normalized.intersection(metric['value']),'Unreliable scale still exports a normalized value'
    return checked


def main():
    clips=json.loads((ROOT/'data/prepared/gait_curated_20261114/manifest.json').read_text())['clips'];index={c['clip_id']:c for c in clips}
    assert len(clips)==len(index)==596
    queue=list(csv.DictReader((OUT/'production_queue.csv').open()));pose_root=OUT/'poses_sapiens2_1b'
    subset=set(json.loads((OUT/'harmonization_subset.json').read_text())['clips'])
    expected_native={c['clip_id'] for c in clips if c['source']!='progait'}|subset
    assert len(queue)==len(expected_native)==219 and {q['clip_id'] for q in queue}==expected_native
    assert {p.parent.name for p in pose_root.glob('*/complete.json')}<=expected_native
    catalog=json.loads((OUT/'catalog.json').read_text())
    assert len(catalog['clips'])==596 and {c['clip_id'] for c in catalog['clips']}==set(index)
    schema=json.loads((pose_root/'joint_schema.json').read_text())['body_feet_23']
    assert set(schema.values())==set(NAMES) and len(schema)==23
    assert schema['62']=='left_wrist' and schema['41']=='right_wrist'
    native=[];failures=[]
    for q in queue:
        cid=q['clip_id'];folder=pose_root/cid
        if not (folder/'complete.json').exists():
            native.append(dict(clip_id=cid,status='pending'));continue
        try:
            complete=json.loads((folder/'complete.json').read_text());c=index[cid]
            frame_map=list(csv.DictReader(open(q['frame_map'])))
            frame_map=[f for f in frame_map if float(q['start_s'])<=float(f['timestamp_s'])<float(q['end_s'])]
            raw=folder/'raw_308.npz'
            assert sha(raw)==complete['raw_npz_sha256'],'raw pose checksum'
            assert complete['source_sha256']==c['sha256'],'curated video checksum provenance'
            assert complete['frames']==len(frame_map)==c['clip_frames'],'frame count'
            with np.load(raw,allow_pickle=False) as a:
                np.testing.assert_array_equal(a['source_frame_index'],np.arange(c['clip_frames']))
                np.testing.assert_array_equal(a['timestamp_s'],[float(f['timestamp_s']) for f in frame_map])
                assert a['keypoints'].shape==(c['clip_frames'],308,2)
                assert a['scores'].shape==(c['clip_frames'],308)
                assert np.isfinite(a['keypoints']).all() and np.isfinite(a['scores']).all()
                assert np.all(np.diff(a['timestamp_s'])>0)
                assert abs(float(a['timestamp_s'][-1])-(c['clip_frames']-1)/c['fps'])<1/c['fps']+.001
                native.append(dict(clip_id=cid,status='passed',frames=complete['frames'],
                    timestamp_first=float(a['timestamp_s'][0]),timestamp_last=float(a['timestamp_s'][-1]),
                    body_feet_score_below_03_fraction=float((a['scores'][:,[int(i) for i in schema]]<.3).mean()),
                    raw_npz_sha256=complete['raw_npz_sha256'],source_original_first_frame=c['source_first_frame']))
        except Exception as e:
            failures.append(dict(clip_id=cid,phase='native_pose',error=type(e).__name__,reason=str(e)[:1000]));native.append(dict(clip_id=cid,status='failed'))
    contract=[];state_counts=Counter();numeric_fields=set();reviewed_metric_exclusions_checked=0
    roots=[(OUT,clips),(OUT/'harmonized',[index[q['clip_id']] for q in queue if index[q['clip_id']]['source']=='progait'])]
    for base,selected in roots:
        for c in selected:
            cid=c['clip_id'];folder=base/'clips'/cid
            try:
                r=json.loads((folder/'metrics.json').read_text());assert r['clip_id']==cid
                assert r['clinical_form_label'] is None and not r['clinical_detection_available']
                reviewed_metric_exclusions_checked+=audit_visual_exclusion_contract(r)
                groups=[s['metrics'] for s in r['segments']] if r['segments'] else [r['metrics']]
                for metrics in groups:
                    assert len(metrics)==17
                    assert {m['metric_id'] for m in metrics}=={f'M{i:02d}' for i in range(1,18)}
                    for m in metrics:
                        assert m['detection_status'] in ('not_evaluable','not_applicable')
                        assert m['reason_codes']
                        if m['value'] is not None:
                            assert m['measurement_status']=='uncertain' and m['uncertainty']['type']=='unquantified'
                            assert set(m['value'])<=set(m.get('observation_units',{}))
                            numeric_fields.update((m['metric_id'],k) for k in m['value'])
                            assert not (set(m['value']) & set(m.get('excluded_observation_components',{})))
                        state_counts[m['measurement_status']]+=1
                if r['status']=='observations_extracted':
                    assert r['processed_frames']==c['clip_frames']
                    with np.load(folder/'observations.npz',allow_pickle=False) as a:
                        assert a['xy_pixels'].shape==(c['clip_frames'],23,2)
                        assert a['quality_mask'].shape==(c['clip_frames'],23)
                        np.testing.assert_array_equal(a['source_frame'],np.arange(c['source_first_frame'],c['source_end_frame_exclusive']))
                        assert np.all(np.diff(a['timestamp_s'])>0)
                        assert np.isnan(a['filtered_xy_up'][~a['quality_mask']]).all()
                        review=r.get('pose_visual_review') or {}
                        if review.get('weight_sha256')==r['pose'].get('weight_sha256'):
                            for mask in review.get('manual_visibility_masks',[]):
                                lo,hi=mask['first_clip_frame'],mask['end_clip_frame_exclusive']
                                assert not a['quality_mask'][lo:hi,[NAMES.index(n) for n in mask['joint_names']]].any()
                    viewer=json.loads((folder/'viewer.json').read_text());assert len(viewer['joint_names'])==23
                    assert viewer['clip_frame'][0]==0 and viewer['clip_frame'][-1]==c['clip_frames']-1
                    with np.load(folder/'observations.npz',allow_pickle=False) as a:
                        # Viewer JSON is serialized to six decimal places; raw
                        # observation timestamps retain their original precision.
                        np.testing.assert_allclose(viewer['native_timestamp_s'],a['timestamp_s'],rtol=0,atol=0.500001e-6)
                contract.append(dict(clip_id=cid,branch='harmonized' if base!=OUT else 'main',status='passed',pose_status=r['status'],segments=len(r['segments'])))
            except Exception as e:
                failures.append(dict(clip_id=cid,phase='metric_contract',branch=str(base),error=type(e).__name__,reason=str(e)[:1000]))
    matrix=json.loads((OUT/'metric_availability.json').read_text())
    assert len(matrix['rows'])==596 and all(len(r['families'])==17 for r in matrix['rows'])
    assert {r['clip_id'] for r in matrix['rows']}==set(index)
    reviews=json.loads((OUT/'pose_visual_review.json').read_text())['clips']
    frozen=json.loads((OUT/'comparison_manifest_frozen.json').read_text())
    extended=json.loads((OUT/'extended_outcomes_frozen.json').read_text())['outcomes']
    outcomes={o['id']:o for o in frozen['outcomes']+extended}
    covered_fields={(o['family'],o['source'].split(':')[1]) for o in outcomes.values() if o['source'].startswith('observation:')}
    assert numeric_fields<=covered_fields,'A numeric projected field has no registered comparison definition'
    comparison_rows=json.loads((OUT/'comparison_clip_values.json').read_text())['rows']
    comparison_rows+=json.loads((OUT/'extended_observation_clip_values.json').read_text())['rows']
    for row in comparison_rows:
        review=reviews.get(row['clip_id'],{})
        assert not (review.get('status')=='regional_pose_failure' and outcomes[row['outcome']]['family'] in review['affected_metric_ids']), 'Documented regional quality exclusion contributed to comparison'
        if review.get('status')=='regional_pose_failure' and review.get('normalization_scale_unreliable',bool(set(review['affected_metric_ids']) & {f'M{i:02d}' for i in range(4,15)})):
            assert outcomes[row['outcome']]['unit']!='L0','Unreliable leg-length normalization contributed to comparison'
    report=json.loads((OUT/'comparison_results.json').read_text())
    reruns=report['harmonization_status']
    assert len(reruns)==13 and {r['clip_id'] for r in reruns}==subset
    quality_exclusions=json.loads((OUT/'comparison_exclusions.json').read_text())['quality']
    for row in reruns:
        metrics=json.loads((OUT/'harmonized/clips'/row['clip_id']/'metrics.json').read_text())
        assert row['status']==metrics['status']
        expected={r['outcome'] for r in comparison_rows if r['clip_id']==row['clip_id'] and r.get('sensitivity')=='base'}
        assert set(row['eligible_primary_outcomes'])==expected
        expected_exclusions=[{k:r[k] for k in ('mode','outcome','reasons','segment_exclusions')}
            for r in quality_exclusions if r['clip_id']==row['clip_id'] and r['sensitivity']=='base']
        assert row['primary_qc_exclusions']==expected_exclusions, 'Displayed ProGait exclusions differ from the comparison records'
    ref_rows={(r['mode'],r['outcome']['id'],r['stratum']['environment']):r for r in report['results'] if r['sensitivity']=='base'}
    primary_rows={(r['mode'],r['outcome'],r['clip_id']):r for r in comparison_rows if r.get('sensitivity')=='base'}
    highlight_catalog=json.loads((OUT/'review_highlights_catalog.json').read_text())
    assert len(highlight_catalog['clips'])==609
    assert {r['key'] for r in highlight_catalog['clips']}==set(index)|{'harmonized/'+cid for cid in subset}
    checked_events=0
    for item in highlight_catalog['clips']:
        cid=item['clip_id'];prefix='harmonized' if item['key'].startswith('harmonized/') else ''
        h=json.loads((OUT/prefix/'clips'/cid/'review_highlights.json').read_text())
        assert h['clip_id']==cid and h['clinical_classification'] is None
        assert h['updated_at']==report['updated_at'],'Review cues do not match current comparison generation'
        regional_events=[e for e in h['events'] if e['kind']=='pose_quality' and e.get('quality_basis')!='manual_visibility_mask']
        review=reviews.get(cid,{})
        expected_regions=[]
        if review.get('status')=='regional_pose_failure':
            record=json.loads((OUT/prefix/'clips'/cid/'metrics.json').read_text())
            if review.get('weight_sha256') and review['weight_sha256']==record.get('pose',{}).get('weight_sha256'):
                expected_regions=review.get('affected_body_regions') or [review.get('affected_body_region','lower_body')]
        assert {e['region'] for e in regional_events}==set(expected_regions),'Reviewed body region missing from warning display or leaked to another pose method'
        assert len(regional_events)==len(set(expected_regions)),'Duplicate regional tracking warning'
        assert set(expected_regions)<={'lower_body','full_body','left_foot','right_foot','arms','left_arm','right_arm','head'},'Tracking warning names an unsupported display region'
        for event in h['events']:
            assert event['correction'] is None and event['severity'] is None
            assert 0<=event['start_time_s']<event['end_time_s']<=index[cid]['clip_duration_s']+1/index[cid]['fps']
            if event['kind']=='movement_review':
                assert event['clinical_status']=='not_validated'
                row=primary_rows[event['comparison_mode'],event['outcome'],cid]
                segment=next(s for s in row['segments'] if s['segment']==event['segment_id'])
                assert abs(segment['value']-event['segment_value'])<1e-8
                assert abs(row['value']-event['clip_value'])<1e-8
                reference=ref_rows[event['comparison_mode'],event['outcome'],event['environment']]
                target=next(u for group in reference['groups'].values() for u in group['units'] if u['unit_id']==event['target_unit_id'])
                assert cid in target['clips'] and target['unit_type']==event['unit_type']
                assert abs(target['value']-event['unit_value'])<1e-8
                units=[u for u in reference['groups']['without_amputation']['units'] if u['unit_id']!=event['target_unit_id'] and u['unit_type']=='participant']
                fence=event['reference'];assert len(units)==fence['participant_count']>=5
                assert set(fence['participant_ids'])=={u['unit_id'] for u in units}
                q1,q3=np.percentile([u['value'] for u in units],[25,75])
                np.testing.assert_allclose([fence['lower'],fence['upper']],[q1-1.5*(q3-q1),q3+1.5*(q3-q1)])
                assert all(v<fence['lower'] or v>fence['upper'] for v in [event['unit_value'],event['clip_value'],event['segment_value']])
            else:
                assert event['kind']=='pose_quality' and event['clinical_status']=='not_a_movement_judgment'
                assert prefix=='harmonized' or index[cid]['source']!='progait','Sapiens tracking warning leaked into publisher branch'
                folder=OUT/prefix/'clips'/cid
                record=json.loads((folder/'metrics.json').read_text())
                assert reviews[cid]['weight_sha256']==record['pose']['weight_sha256']
                frames=event['evidence_frames']
                if event.get('quality_basis')=='manual_visibility_mask':
                    mask=next(m for m in reviews[cid]['manual_visibility_masks'] if m['first_clip_frame']==event['first_clip_frame'] and m['end_clip_frame_exclusive']==event['end_clip_frame_exclusive'])
                    assert event['masked_joint_names']==mask['joint_names']
                    assert frames==[mask['first_clip_frame'],mask['end_clip_frame_exclusive']-1]
                else:
                    assert reviews[cid]['status']=='regional_pose_failure'
                    assert frames and frames==reviews[cid].get('failure_clip_frames',reviews[cid]['inspected_clip_frames'])
                assert all(isinstance(f,int) and 0<=f<index[cid]['clip_frames'] for f in frames)
                with np.load(folder/'observations.npz',allow_pickle=False) as observations:
                    np.testing.assert_array_equal(event['evidence_time_s'],observations['timestamp_s'][frames])
                viewer=json.loads((folder/'viewer.json').read_text())
                for frame,time in zip(frames,event['evidence_time_s']):
                    assert frame in viewer['clip_frame'],'Evidence frame missing from display poses'
                    assert abs(viewer['timestamp_s'][viewer['clip_frame'].index(frame)]-time)<=0.500001e-6
            checked_events+=1
    out=dict(updated_at=datetime.now(timezone.utc).isoformat(),status='failed' if failures else 'passed_for_completed_outputs',
        native_jobs=len(queue),native_completed=sum(r['status']=='passed' for r in native),native_pending=sum(r['status']=='pending' for r in native),
        native_frames_verified=sum(r.get('frames',0) for r in native),native=native,
        main_clips=596,harmonized_clips=13,contract_records=len(contract),contract=contract,
        availability_matrix_cells=596*17,metric_states=dict(state_counts),failures=failures,
        comparison_visual_exclusion_check='passed',comparison_clip_rows_checked=len(comparison_rows),
        reviewed_metric_exclusions_checked=reviewed_metric_exclusions_checked,
        normalized_measurement_visual_exclusion_check='passed',review_highlight_records_checked=609,review_highlight_events_checked=checked_events,
        numeric_projected_fields_covered=len(numeric_fields),
        limits='Counts, mappings, timestamps, hashes and output contract verified. Passing does not certify anatomical landmark accuracy or absence of every tracking failure.')
    write(OUT/'output_audit.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ('native','contract','metric_states','limits')}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()

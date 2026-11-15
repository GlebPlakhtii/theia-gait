"""Describe a fixed interim snapshot without changing production or frozen rules."""
import argparse
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    out = args.snapshot.resolve()
    read = lambda name: json.loads((out / name).read_text())
    report, catalog = read('comparison_results.json'), read('catalog.json')
    matrix, review = read('metric_availability.json'), read('pose_visual_review.json')
    native, timing = read('production_status.json'), read('video_timestamp_audit.json')
    exclusions = read('comparison_exclusions.json')
    assert len(catalog['clips']) == 596 and len(matrix['rows']) == 596
    assert all(len(x['families']) == 17 for x in matrix['rows'])
    counts = Counter((c['source'], c['status']) for c in catalog['clips'])
    source_rows = []
    for source in sorted({c['source'] for c in catalog['clips']}):
        clips = [c for c in catalog['clips'] if c['source'] == source]
        source_rows.append(dict(source=source, total=len(clips),
            extracted=counts[source, 'observations_extracted'],
            pending=counts[source, 'waiting_for_pose'],
            known_source_subjects=len({c['subject'] for c in clips if c['subject']}) or None,
            cohort=clips[0]['cohort'],
            pose_method='Publisher annotations' if source == 'progait' else 'Sapiens2-1B'))
    primary = [r for r in report['results'] if r['sensitivity'] == 'base'
        and r['stratum']['environment'] == 'overground'
        and r['status'] == 'descriptive_comparison_available']
    method = {r['outcome']['id']: r for r in report['paired_pose_method_summary']}
    families = []
    names = {r['metric_id']: r['name'] for r in read('metric_comparison_coverage.json')['rows']}
    for i in range(1, 18):
        mid = f'M{i:02d}'
        values, raw = Counter(), Counter()
        for row in matrix['rows']:
            f = next(f for f in row['families'] if f['metric_id'] == mid)
            if f['states'].get('uncertain', 0): values[row['source']] += 1
            if f['numeric_observation_segments']: raw[row['source']] += 1
        families.append(dict(id=mid, name=names[mid], clips_with_uncertain_numeric_value=dict(values),
                             clips_with_any_raw_observation=dict(raw)))
    strict = [r for r in exclusions['quality']
        if r['mode'] == 'guarded_known_subject_sensitivity' and r['sensitivity'] == 'strict_coverage'
        and r['outcome'] == 'trunk_frontal_rom95' and r['clip_id'].startswith('ljmu360_')
        and 'pose_pending' not in r['reasons']]
    strict_counts = dict(Counter(reason for r in strict for reason in r['reasons']))
    failures = {k:v for k,v in review['clips'].items() if v['status'] == 'regional_pose_failure'}
    subject_counts = dict(Counter(c['subject'] for c in catalog['clips'] if c['source'] == 'progait'))
    analysis = dict(snapshot_at=read('snapshot_manifest.json')['captured_at'],
        derived_results_at=report['updated_at'], status='interim_fixed_snapshot',
        main_processing_counts=report['processing_counts'], harmonization_counts=report['harmonization_counts'],
        native_completed_jobs=native['completed_clips'], native_total_jobs=native['total_jobs'],
        decoded_clips_passed=timing['counts']['passed'], source_coverage=source_rows,
        progait_clips_per_known_participant=subject_counts,
        primary_comparisons=primary, pose_method_sensitivity=[method[r['outcome']['id']] for r in primary],
        strict_reference_exclusions=strict_counts, visual_reviewed_clips=len(review['clips']),
        regional_pose_failures=failures, metric_availability=families,
        interpretation=dict(reliable_cohort_difference_established=False,
            clinical_good_bad_classifier_validated=False, main_run_resumed=True,
            selection_limitation='Short clips finish first; this is an incomplete, non-random snapshot.',
            cohort_limitation='Available comparisons use one guarded ProGait participant and seven LJMU participants. Camera, speed and assistance are not matched.',
            normalization_limitation='Normalized observations depend on projected leg length. No L0-normalized values from either visually failed leg track are interpreted in this review.',
            next_steps=['Finish the existing native-cadence queue.',
                'Review prosthetic leg localization before interpreting knee or left/right asymmetry.',
                'Repeat frozen participant/source-balanced comparisons and quality sensitivities with all available outputs.',
                'Keep ground/contact-dependent and clinical metrics unavailable until their required references are validated.']))
    (out/'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False))
    fmt = lambda v: '—' if v is None else f'{v:.2f}'
    tr = []
    for r in primary:
        a,b = r['groups']['with_amputation'],r['groups']['without_amputation']
        delta = method[r['outcome']['id']]['participant_summary']['median']
        tr.append(f"<tr><th>{escape(r['outcome']['label'])}</th><td>{fmt(a['median'])}°</td>"
            f"<td>{fmt(b['median'])}°<small>IQR {fmt(b['q25'])}–{fmt(b['q75'])}°</small></td>"
            f"<td>{fmt(r['median_difference_with_minus_without'])}°</td><td>{fmt(delta)}°</td></tr>")
    source_table = ''.join(f"<tr><th>{escape(s['source'])}</th><td>{s['extracted']} / {s['total']}</td>"
        f"<td>{s['pending']}</td><td>{s['known_source_subjects'] or 'Unknown'}</td><td>{s['pose_method']}</td></tr>"
        for s in source_rows)
    family_table = ''.join(f"<tr><th>{f['id']} · {escape(f['name'])}</th>"
        f"<td>{sum(f['clips_with_uncertain_numeric_value'].values())}</td>"
        f"<td>{sum(f['clips_with_any_raw_observation'].values())}</td></tr>" for f in families)
    clip = 'PG_outside_1_2_1_f_2_616_691'
    failure_cards = ''
    for cid, item in failures.items():
        # Reference saved visual evidence; previews are immutable completed-pose outputs.
        evidence = item.get('frame_evidence', [])
        src = evidence[0] if evidence else f'poses_sapiens2_1b/{cid}/preview_000056.jpg'
        if src.startswith('results/gait_metrics_20261114/'): src = src.split('results/gait_metrics_20261114/',1)[1]
        failure_cards += (f'<article><h3>{escape(cid)}</h3><p>{escape(item["evidence"])}</p>'
            f'<a href="../../{escape(src)}"><img loading="lazy" src="../../{escape(src)}" alt="Sampled pose failure evidence"></a>'
            f'<p><a href="../../?clip={escape(cid)}&amp;method=sapiens">Review video and pose</a></p></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Theia · Interim evidence review</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#172b3a;font:16px/1.55 system-ui,sans-serif}
main{max-width:1140px;margin:auto;padding:34px 24px 60px}h1{font-size:36px;line-height:1.15;max-width:850px;margin:12px 0 18px}h2{font-size:23px;margin:0 0 14px}h3{font-size:16px;overflow-wrap:anywhere}
p{max-width:1000px}a{color:#05675f}nav{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:20px}.eyebrow{letter-spacing:.12em;font-size:12px;font-weight:700;color:#59717d}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:25px 0}.card,section{background:white;border:1px solid #dce3e7;border-radius:12px;padding:23px;margin-bottom:20px}.card{margin:0}.card strong{display:block;font-size:30px}.muted,small{color:#526b79}small{display:block;font-size:12px}.notice{background:#fff2d9;border-left:4px solid #be7d19;padding:15px 20px;border-radius:5px}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dfe6e9;vertical-align:top}thead th{background:#edf4f4}tbody th{font-weight:500}.evidence{display:grid;grid-template-columns:1fr 1fr;gap:20px}.evidence img{width:100%;height:330px;object-fit:contain;background:#15232e;border-radius:8px}li{margin:9px 0}.tag{display:inline-block;background:#e2f1eb;color:#155947;border-radius:30px;padding:4px 12px;font-size:13px}code{overflow-wrap:anywhere}@media(max-width:740px){.cards,.evidence{grid-template-columns:1fr}h1{font-size:29px}main{padding:22px 14px}section{padding:18px}}
</style><main><nav><a href="../../comparison.html">Live comparisons</a><a href="../../">All videos and poses</a><a href="analysis.json">Snapshot analysis JSON</a></nav>
<div class="eyebrow">THEIA / FIXED INTERIM REVIEW / 14 NOVEMBER 2026 · 20:31 UTC</div>
<h1>Useful motion measurements. No reliable cohort difference established yet.</h1>
<p>This review uses the outputs available at the brief processing pause. The main GPU run and automatic metric extraction resumed at 20:31 UTC. These numbers stay fixed while the live report continues updating.</p>
<span class="tag">Main run resumed</span>
<div class="cards"><div class="card"><small>OBSERVATIONS EXTRACTED</small><strong>427 / 596 clips</strong><span>390 publisher pose records + 37 newly inferred clips.</span></div><div class="card"><small>NATIVE INFERENCE COMPLETE</small><strong>40 / 219 jobs</strong><span>37 new clips + 3 same-model ProGait clips. 179 jobs remain.</span></div><div class="card"><small>FULL VIDEO DECODE AUDIT</small><strong>596 / 596 passed</strong><span>Counts, dimensions, timestamps and file hashes. This does not certify pose accuracy.</span></div></div>
<section><h2>What the current measurements show</h2><p>Three primary observations currently have measurements on both sides: frontal trunk motion, hip-landmark-line motion and shoulder-line motion. All use Sapiens2-1B and overground walking. ROM here is the P95–P05 range of a projected signal, not a calibrated anatomical range.</p>
<p class="notice"><b>Sample: 1 ProGait participant / 1 clip versus 7 LJMU participants / 13 clips.</b> ProGait includes guarding and possible belt contact; assistance, speed and camera geometry differ. These are descriptive pilot values, not an amputation effect or a good/bad-form threshold.</p>
<div class="table"><table><thead><tr><th>Projected observation</th><th>With amputation</th><th>Reference median [IQR]</th><th>With − reference</th><th>Same ProGait clip:<br>Sapiens − publisher</th></tr></thead><tbody>PRIMARY_ROWS</tbody></table></div>
<p>The hip-line method difference (−2.83°) is larger in magnitude than its apparent between-group difference (+0.83°). Method sensitivity is not a known error bound, but it shows why estimator consistency and landmark review matter.</p>
<p><a href="../../?clip=PROGAIT_CLIP&amp;method=sapiens">Inspect the contributing ProGait video and Sapiens pose</a> · <a href="comparison_results.json">All primary values and contributing reference clips</a></p></section>
<section><h2>What prevents a strong conclusion</h2><ul>
<li><b>Stricter quality settings remove the reference comparison.</b> None of the 22 processed frontal/rear reference candidates passes the complete strict gate. Twenty have small projected body scale among their exclusion reasons; two fail the retained-duration requirement. Some also contain segments too short for this analysis. The strict gate requires ≥120-pixel projected leg scale and ≥90% landmark coverage.</li>
<li><b>Two reviewed Sapiens tracks misplace the prosthetic leg.</b> One places distal points near the hip; another collapses both leg chains onto the stance leg. Their affected lower-limb measurements are excluded. Confidence scores alone did not identify the problem. Normalized values from these failed tracks are not interpreted here because their leg-length denominator also needs review.</li>
<li><b>The snapshot favors short clips.</b> The production queue completes shorter videos first. There is no eligible same-model knee comparison yet, and no unsupported-walking comparison has both groups available.</li>
<li><b>More clips do not mean more independent people.</b> All 390 ProGait clips represent four participants. Their clip counts are 161, 120, 22 and 87. Additional-input identities remain unresolved; the eight LBMC camera clips come from one participant and one trial.</li></ul></section>
<section><h2>Coverage by source</h2><div class="table"><table><thead><tr><th>Source</th><th>Extracted / all</th><th>Pending</th><th>Known people in source</th><th>Main pose branch</th></tr></thead><tbody>SOURCE_ROWS</tbody></table></div>
<p class="muted">The three ProGait Sapiens reruns are a separate method-matching branch and are not added to the 596 unique videos. LJMU and LBMC are publisher-reported healthy cohorts; MoVi/MPI remain provisional references. Neither label certifies correct gait.</p></section>
<section><h2>Which requested metrics are usable now?</h2><p>Six of the 17 families have some numeric projected observations: frontal/side trunk motion, side-view knee bend, shoulder movement, arm motion and head offsets. Counts below mean at least one numeric segment exists; they do not mean a clip passes comparison QC. Raw proxies can exist even when the requested clinical metric remains unavailable.</p>
<details><summary>Show all 17 families and clip counts</summary><div class="table"><table><thead><tr><th>Requested family</th><th>Clips with uncertain numeric value</th><th>Clips with any raw proxy / observation</th></tr></thead><tbody>FAMILY_ROWS</tbody></table></div></details>
<p>Ground clearance, contact-defined stance/swing timing, step length, foot-progression angle and cycle variability remain unavailable. Phase-specific pelvic hiking, circumduction and vaulting also cannot be inferred from their raw 2D proxies. No clinically validated normal/abnormal labels are produced.</p>
<p><a href="metric_availability.json">Full 596 × 17 matrix</a> · <a href="extended_observation_results.json">All supplementary observations</a></p></section>
<section><h2>Visible pose failures already excluded</h2><p>These examples document measurement failures, not poor walking form. Thirty-three clips had sampled visual review at this snapshot; two had these regional failures. This targeted review is not a random accuracy study.</p><div class="evidence">FAILURE_CARDS</div></section>
<section><h2>What continues now</h2><ol><li>Finish the existing Sapiens2-1B queue at native cadence, keeping checkpoints and originals.</li><li>Review the remaining prosthetic leg tracks, especially knee localization and left/right collapse.</li><li>Repeat the frozen comparisons with all eligible outputs, participant/source balancing and strict-quality sensitivity.</li><li>Deliver numerical observations and exclusions; retain unavailable clinical metrics explicitly.</li></ol>
<p><a href="snapshot_manifest.json">Snapshot provenance and hashes</a> · <a href="pose_visual_review.json">Visual review records</a> · <a href="video_timestamp_audit.json">Completed video audit</a> · <a href="../../comparison.html">Return to the live run</a></p></section></main></html>'''
    page = (page.replace('PRIMARY_ROWS',''.join(tr)).replace('SOURCE_ROWS',source_table)
        .replace('FAMILY_ROWS',family_table).replace('FAILURE_CARDS',failure_cards).replace('PROGAIT_CLIP',clip))
    figures = '''<section id="graphs"><h2>Graphs and participant scatterplots</h2><p>Each dot is one participant after repeated clips and views are aggregated. The comparison currently contains one ProGait participant and seven reference participants. No fitted trend or diagnostic boundary is shown.</p>'''
    for stem,title in [('participant_distributions','Individual participant distributions'),
                       ('trunk_shoulder_scatter','Trunk versus shoulder motion'),
                       ('method_sensitivity','Group difference and method disagreement'),
                       ('source_coverage','Extraction coverage by source')]:
        figures += f'<figure style="margin:24px 0"><h3>{title}</h3><a href="figures/{stem}.svg"><img style="width:100%;height:auto;background:white;border-radius:8px" loading="lazy" src="figures/{stem}.png" alt="{title}"></a><figcaption><a href="figures/{stem}.svg" download>Download SVG</a> · <a href="figures/{stem}.png" download>Download PNG</a></figcaption></figure>'
    figures += '<p><a href="figures/review_figures.json">Scatterplot points and figure provenance</a> · <a href="../../?clip=PG_outside_1_2_1_f_2_616_691&amp;method=sapiens">Open the ProGait example with body-region review highlights</a></p></section>'
    page=page.replace('<section><h2>What prevents a strong conclusion</h2>',figures+'<section><h2>What prevents a strong conclusion</h2>')
    (out/'review.html').write_text(page)
    (out/'analysis_provenance.json').write_text(json.dumps(dict(
        generator=str(Path(__file__).resolve()), generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256((out/'analysis.json').read_bytes()).hexdigest(),
        snapshot_manifest_sha256=hashlib.sha256((out/'snapshot_manifest.json').read_bytes()).hexdigest()),indent=2))
    print(json.dumps(dict(report=str(out/'review.html'),primary_comparisons=len(primary),source_coverage=source_rows),indent=2))


if __name__ == '__main__':
    main()

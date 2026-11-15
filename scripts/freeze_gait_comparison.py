"""Freeze metadata eligibility and quality rules without reading metric outcomes."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gait_metrics_20261114'
MANIFEST = ROOT / 'data/prepared/gait_curated_20261114/manifest.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(c):
    """Conservatively collapse repeated views before participant summaries."""
    source = c['source']
    stem = Path(c['original']).stem
    if source == 'additional_input':
        return None, 'source:' + c['source_sha256']
    subject = c['split_group']
    if source == 'lbmc':
        trial = subject + ':one_preview_trial'
    elif source == 'ljmu360':
        trial = subject + ':' + re.search(r'E\d+T\d+', stem).group()
    elif source == 'progait':
        # Camera and repeat suffixes do not create independent trial units.
        trial = subject + ':' + c['environment'] + ':' + '_'.join(stem.split('_')[:3])
    elif source == 'movi':
        trial = subject + ':' + c['activity']
    else:
        trial = subject + ':selected_sequence'
    return subject, trial


def nominal_view(c, context):
    if context:
        return context['nominal_view']
    label = c.get('view', '').lower()
    if label.startswith('side'):
        return 'side'
    if label in ('front', 'rear', 'front/rear'):
        return 'front_rear'
    if 'oblique' in label:
        return 'oblique'
    return 'unknown_or_changing'


def main():
    target = OUT / 'comparison_manifest_frozen.json'
    if target.exists():
        raise SystemExit('Frozen manifest already exists; do not overwrite after outcome inspection.')
    clips = json.loads(MANIFEST.read_text())['clips']
    subset = set(json.loads((OUT / 'harmonization_subset.json').read_text())['clips'])
    contexts = {}
    for name in ('additional_context_review.json', 'progait_subset_context_review.json'):
        contexts.update(json.loads((OUT / name).read_text())['clips'])
    rows = []
    for c in clips:
        context = contexts.get(c['clip_id'], {})
        view = nominal_view(c, context)
        subject, trial = identity(c)
        support = context.get('visible_support')
        if support is None:
            support = ('no_aid_recorded_in_curated_reference' if c['source'] in ('lbmc', 'ljmu360')
                       else 'parallel_bars' if c['environment'] == 'parallel_bars' else 'unverified')
        reasons = []
        if c['activity'] not in ('walk', 'walking'):
            reasons.append('not_ordinary_walking')
        if c['environment'] not in ('overground', 'treadmill'):
            reasons.append('environment_not_level_overground_or_treadmill')
        if view not in ('side', 'front_rear'):
            reasons.append('view_oblique_unknown_or_changing')
        if context.get('contains_turns_or_transitions') or context.get('scene_edit_visible'):
            reasons.append('turn_transition_or_edit_observed')
        if c['clip_duration_s'] < 2:
            reasons.append('clip_shorter_than_2_seconds')
        if c.get('cohort_evidence_level') == 'provisional_visual_comparison':
            reasons.append('provisional_cohort_not_primary_reference')
        method = 'sapiens2_1b_bf16' if c['source'] != 'progait' or c['clip_id'] in subset else 'publisher_annotation'
        modes = []
        if not reasons and method == 'sapiens2_1b_bf16':
            if c['source'] in ('ljmu360', 'lbmc'):
                modes = ['unsupported_context_descriptive', 'guarded_known_subject_sensitivity']
            elif c['source'] == 'additional_input' and support == 'none':
                modes = ['unsupported_context_descriptive']
            elif c['source'] == 'progait':
                modes = ['guarded_known_subject_sensitivity']
        rows.append(dict(clip_id=c['clip_id'], source=c['source'], cohort=c['cohort'],
                         activity=c['activity'], activity_group='walking' if c['activity'] in ('walk', 'walking') else c['activity'],
                         environment=c['environment'], view=view, support_context=support,
                         subject=subject, trial_block=trial, source_sha256=c['source_sha256'],
                         clip_duration_s=c['clip_duration_s'], pose_method=method,
                         main_pose_method='publisher_annotation' if c['source'] == 'progait' else method,
                         same_model_subset=c['clip_id'] in subset,
                         comparison_modes=modes, metadata_exclusions=reasons,
                         no_primary_mode_reason=None if modes else ('; '.join(reasons) or
                             'unsupported_walking_context_or_same_model_measurements_not_available'),
                         context_review=context or None))
    protocol = dict(
        created_at=datetime.now(timezone.utc).isoformat(), frozen_before_group_differences=True,
        outcome_files_read_during_freeze=False, total_clips=len(rows),
        title='Matched projected gait observations; descriptive research comparison',
        model='Sapiens2-1B BF16, native cadence, flip average',
        quality_rules=dict(minimum_clip_duration_s=2, minimum_segment_duration_s=1,
                           minimum_fixed_L0_pixels=80, minimum_component_valid_fraction=.8,
                           minimum_joint_score=.3, maximum_masked_jump_speed_L0_per_s=12,
                           paired_metrics_use_common_valid_frames=True,
                           exclude_segment_quality_flags=True,
                           primary_minimum_source_clip_retained_duration_fraction=.5,
                           meaning='Engineering QC rules, not clinical thresholds or calibrated error bounds'),
        sensitivities=[dict(name='strict_coverage', minimum_component_valid_fraction=.9,
                            minimum_fixed_L0_pixels=120,
                            explanation='Repeat exact outcomes with stricter pose coverage and image scale'),
                       dict(name='static_camera_only',
                            explanation='Exclude additional-input clips marked panning or moving; unresolved metadata retained as unverified')],
        matching_fields=['activity_group', 'environment', 'view', 'pose_method'],
        modes=dict(
            unsupported_context_descriptive=dict(
                description='Additional-input clips with no visible aid versus publisher-reported healthy references, same task/environment/view/model.',
                limitation='Additional-input subject identities are unresolved; source files are aggregation blocks, not independent participants. Reference support is based on curated ordinary-walking context, not measured assistance force.',
                inference='Descriptive only; no participant count or inferential interval for the unknown-identity group'),
            guarded_known_subject_sensitivity=dict(
                description='Known ProGait participants versus publisher-reported healthy references, same task/environment/view/model; guarding context differs.',
                limitation='ProGait has a nearby assistant and visible or possible belt contact. This is an explicitly context-unmatched sensitivity comparison, not an estimate of the effect of amputation.',
                inference='Descriptive; no intervals with fewer than five independently identified participants per group; exclude free-arm outcomes')),
        aggregation=['eligible segment statistics, duration-weighted within each clip',
                     'median over clips/views within conservative physical-trial block',
                     'median over trial blocks per known participant',
                     'median and IQR across participant summaries; unknown identities use source-file blocks with participant n=null'],
        uncertainty=dict(minimum_known_subjects_each_group=5, bootstrap_unit='participant',
                         bootstrap_resamples=2000, random_seed=20261114,
                         no_cluster_interval_for_unknown_identity=True,
                         note='No p-values or causal interpretation; participant intervals, if eligible, cover sampling variability only, not pose/calibration error'),
        outcomes=[
            dict(id='knee_rom95_mean', family='M05', view='side', source='paired:knee:mean_rom95', unit='deg', label='Mean left/right projected knee ROM (P95−P05)'),
            dict(id='knee_rom95_asymmetry', family='M05', view='side', source='paired:knee:absolute_rom95_difference', unit='deg', label='Absolute left/right projected knee ROM difference'),
            dict(id='knee_p95_mean', family='M05', view='side', source='paired:knee:mean_p95', unit='deg', label='Mean left/right projected knee bend P95'),
            dict(id='knee_temporal_sd_mean', family='M05', view='side', source='paired:knee:mean_sd', unit='deg', label='Mean left/right within-segment projected knee SD; not cycle variability'),
            dict(id='trunk_frontal_rom95', family='M02', view='front_rear', source='observation:trunk_image_inclination_deg:rom95', unit='deg', label='Frontal image-plane trunk inclination ROM'),
            dict(id='trunk_side_rom95', family='M03', view='side', source='observation:trunk_image_inclination_deg:rom95', unit='deg', label='Side image-plane trunk inclination ROM'),
            dict(id='hip_line_rom95_proxy', family='M01', view='front_rear', source='observation:left_up_hip_line_deg:rom95', unit='deg', label='Hip-landmark line ROM proxy; not phase-specific pelvic hiking'),
            dict(id='shoulder_line_rom95', family='M15', view='front_rear', source='observation:left_up_shoulder_line_deg:rom95', unit='deg', label='Shoulder-landmark line ROM'),
            dict(id='arm_rom95_mean', family='M16', view='side', source='paired:arm:mean_rom95', unit='L0', label='Mean left/right wrist–shoulder horizontal ROM proxy', free_arm_only=True),
            dict(id='arm_rom95_asymmetry', family='M16', view='side', source='paired:arm:absolute_rom95_difference', unit='L0', label='Absolute left/right wrist–shoulder ROM difference', free_arm_only=True),
        ],
        unavailable_comparisons=['Verified stance/swing, step times/length, double support, clearance, vaulting, phase-specific hiking and clinical gait labels: calibration or contact/reference requirements unmet.',
                                 'Primary running/jogging comparison: no verified healthy running reference with matched views in this curated set.',
                                 'Unmatched treadmill, aided, oblique and transition strata are retained with explicit counts.',
                                 'MoVi/MPI form a separate provisional cohort; they are never merged into the publisher-reported healthy reference.'],
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in [MANIFEST, OUT / 'harmonization_subset.json',
                         OUT / 'additional_context_review.json', OUT / 'progait_subset_context_review.json']},
        rows=rows)
    target.write_text(json.dumps(protocol, indent=2, allow_nan=False))
    (OUT / 'comparison_manifest_frozen.sha256').write_text(digest(target) + '\n')
    print(json.dumps(dict(rows=len(rows), modes=dict(Counter(m for r in rows for m in r['comparison_modes'])))))


if __name__ == '__main__':
    main()

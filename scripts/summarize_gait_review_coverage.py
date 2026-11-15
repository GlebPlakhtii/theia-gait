"""Summarize saved predictions without new inference or threshold selection."""
from pathlib import Path
import json
from motionbert_gait_common import ROOT, now, sha, write
from gait_from_scratch_evaluation import merge_review_intervals


def summarize(folder):
    folder=Path(folder);prediction_path=folder/'predictions.json'
    windows=json.loads(prediction_path.read_text())['windows'];summary={}
    for key in sorted({w['split']+'_'+w['method'] for w in windows}):
        selected=[w for w in windows if w['split']+'_'+w['method']==key]
        def interval(w):return dict(sequence_id=w['source_sha256']+':'+w['method'],start_time_s=w['source_time_start_s'],end_time_s=w['source_time_end_s'])
        evaluable=[w for w in selected if any(e['prediction_state']!='unavailable' for e in w['events'])]
        comparable=[w for w in selected if any(e['disagreement'] is not None for e in w['events'])]
        different=[w for w in comparable if any(e['disagreement'] is True for e in w['events'])]
        merged=merge_review_intervals([interval(w) for w in evaluable])
        observed=merge_review_intervals([interval(w) for w in selected])
        summary[key]=dict(windows=len(selected),windows_with_any_available_target=len(evaluable),
            fully_abstained_windows=len(selected)-len(evaluable),
            fully_abstained_window_fraction=1-len(evaluable)/len(selected) if selected else None,
            evaluable_duration_s=merged['review_duration_s'],observed_duration_s=observed['review_duration_s'],
            evaluable_duration_fraction=merged['review_duration_s']/observed['review_duration_s'] if observed['review_duration_s'] else None,
            windows_with_comparable_teacher_and_model=len(comparable),teacher_disagreement_windows=len(different),
            teacher_disagreement_window_fraction=len(different)/len(comparable) if comparable else None,
            per_target={target:dict(available=sum(e['prediction_state']!='unavailable' for w in selected for e in w['events'] if e['target_id']==target),
                positive=sum(e['prediction_state']=='positive' for w in selected for e in w['events'] if e['target_id']==target)) for target in (selected[0]['events'][i]['target_id'] for i in range(13))} if selected else {},
            interpretation='Window abstention includes unsupported views, contexts and references; it is not solely pose-estimator failure. Durations union original-source intervals within method.')
    write(folder/'review_coverage.json',dict(created_at=now(),predictions_sha256=sha(prediction_path),groups=summary,
        new_inference=False,new_threshold_fitting=False,code_sha256=sha(Path(__file__))))
    report_path=folder/'report_data.json';data=json.loads(report_path.read_text());data['review_coverage']=summary
    if 'review_coverage.json' not in data['artifacts']:data['artifacts'].append('review_coverage.json')
    write(report_path,data)
    status_path=folder/'report_status.json';status=json.loads(status_path.read_text());status['files_sha256']['report_data.json']=sha(report_path)
    status['files_sha256']['review_coverage.json']=sha(folder/'review_coverage.json');write(status_path,status)
    return summary


if __name__=='__main__':
    for folder in [ROOT/'results/gait_from_scratch_v1',ROOT/'results/motionbert_gait_finetune_v1/completed_run']:
        result=summarize(folder);print(folder.name,result['test_sapiens'])

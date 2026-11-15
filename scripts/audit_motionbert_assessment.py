"""Mechanical consistency and checkpoint audit; never train or select a model."""
import json
from pathlib import Path
import numpy as np
import torch
from motionbert_gait_common import ROOT, sha, write, now


def audit(folder):
    folder=Path(folder);read=lambda name:json.loads((folder/name).read_text())
    data=np.load(folder/'dataset/dataset.npz');pred=np.load(folder/'predictions.npz')
    evaluation=read('evaluation/results.json');comparison=read('architecture_comparison.json')
    assert evaluation['status']=='complete'
    assert sha(folder/'dataset/dataset.npz')==evaluation['dataset_sha256']==comparison['dataset_sha256']
    assert comparison['same_participants_windows_labels_and_threshold']
    selection=(data['split']==2)&(data['method']==0)
    for j,row in enumerate(evaluation['reports']['test_sapiens']['labels']):
        keep=selection&(data['labels'][:,j]>=0)&row['enabled']&data['input_any_valid']
        actual=data['labels'][keep,j];positive=pred['probabilities'][keep,j]>=.5
        assert row['n']==len(actual)
        counts={'tp':int(((actual==1)&positive).sum()),'fp':int(((actual==0)&positive).sum()),
            'tn':int(((actual==0)&~positive).sum()),'fn':int(((actual==1)&~positive).sum())}
        assert all(row[k]==v for k,v in counts.items())
        if len(set(actual))<2:assert all(row[k] is None for k in ['precision','recall','f1','average_precision'])
    for j,row in enumerate(evaluation['reports']['test_sapiens']['outcomes']):
        keep=selection&data['measurement_mask'][:,j]&data['input_any_valid'];assert row['n']==int(keep.sum())
        if keep.any():assert abs(row['mae']-float(np.abs(data['values'][keep,j]-pred['values'][keep,j]).mean()))<1e-5
    for window in read('predictions.json')['windows']:
        for event in window['events']:
            assert event['clinical_form_label'] is None and event['treatment_or_prosthetic_adjustment'] is None
            if event['prediction_state']=='positive':assert event['quality_gate_passed'] and event['predicted_evaluability_score']>=.5 and event['predicted_value']>=0
            if window['method']=='publisher':assert event['prediction_state']=='unavailable'
    initial=torch.load(folder/'training/initial.pt',map_location='cpu',weights_only=False)
    warmup=torch.load(folder/'training/warmup.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,warmup['model_state'][k]) for k,v in initial['model_state'].items() if k.startswith('encoder.'))
    del warmup
    best=torch.load(folder/'training/best.pt',map_location='cpu',weights_only=False)
    changed=[k for k,v in initial['model_state'].items() if k.startswith('encoder.') and not torch.equal(v,best['model_state'][k])]
    assert changed and best['encoder_updates']>0
    assert all(torch.isfinite(v).all() for v in best['model_state'].values())
    # Both spatial and temporal blocks must actually change, not just the output head.
    assert any('blocks_st' in k for k in changed) and any('blocks_ts' in k for k in changed)
    files=['dataset/dataset_manifest.json','dataset_audit.json','device_pilot.json','training/resolved_training.json',
        'training/training_result.json','training/initial.pt','training/warmup.pt','training/best.pt','training/last.pt',
        'backbone_update_audit.json','evaluation/results.json','architecture_comparison.json','predictions.npz','latency.json',
        'report_status.json']
    result=dict(status='passed',audited_at=now(),same_dataset_as_scratch=True,heldout_confusion_and_mae_recomputed=True,
        predictions_quality_gates_checked=True,warmup_encoder_unchanged=True,best_encoder_changed_tensors=len(changed),
        spatial_and_temporal_blocks_updated=True,checkpoint_finite=True,clinical_validation=False,
        final_browser_review_required=True,files_sha256={name:sha(folder/name) for name in files},
        audit_code_sha256=sha(Path(__file__)))
    write(folder/'experiment_audit.json',result);print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':audit(ROOT/'results/motionbert_gait_finetune_v1/completed_run')

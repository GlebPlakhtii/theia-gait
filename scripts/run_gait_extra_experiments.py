"""Run the frozen additional experiment queue, then stop for held-out scoring."""
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import time
import traceback
import torch
import numpy as np
from gait_extra_data import OUT,BASE,freeze,refit,snapshot,condition_data,datasets
from gait_assessment_data import load_dataset
from gait_from_scratch_training import audit_inputs,selection_basis
from gait_from_scratch_model import TCNConfig
from gait_extra_train import train_tcn,train_mb
from motionbert_gait_common import ROOT,sha,write,now


def run():
    torch.set_num_threads(2);base=load_dataset(BASE);protocol=freeze(base)
    code={n:sha(ROOT/'scripts'/n) for n in ['gait_extra_data.py','gait_extra_train.py','run_gait_extra_experiments.py']}
    write(OUT/'implementation_manifest.json',dict(created_at=now(),code_sha256=code,protocol_sha256=sha(OUT/'protocol.json')))
    split_data={}
    for name,mapping in protocol['participant_maps'].items():
        d=refit(base,mapping)
        if name=='original':
            assert np.array_equal(d['labels'],base['labels'])
            assert np.array_equal(d['label_mask'],base['label_mask'])
            assert np.allclose(d['standardized'],base['standardized'],atol=1e-6)
            assert d['normalization']==base['normalization']
        snapshot(d,mapping,OUT/'splits'/name);split_data[name]=d
    write(OUT/'preflight.json',dict(status='passed',created_at=now(),original_labels_and_normalization_exactly_reproduced=True,
        folds=4,known_test_people_unique_across_folds=15,all_folds_refit_training_only=True,
        classes_enabled={k:int(d['enabled'].sum()) for k,d in split_data.items()},clinical_validation=False))
    cutoff=datetime.fromisoformat(protocol['training_cutoff_utc']).timestamp();results=[]
    for job in protocol['jobs']:
        if time.time()>=cutoff:
            results.append(dict(**job,status='not_run_time_cutoff'));continue
        if shutil.disk_usage(OUT).free<2_000_000_000:
            results.append(dict(**job,status='not_run_disk_reserve'));continue
        folder=OUT/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=False)
        write(OUT/'progress.json',dict(status='training',updated_at=now(),current=job,
            completed=len([r for r in results if r['status']=='trained_with_validation']),scheduled=len(protocol['jobs'])))
        d=condition_data(split_data[job['split']],job['condition']);train,val=datasets(d,job['condition'])
        enabled=torch.from_numpy(d['enabled']);audit_inputs(train,val,enabled,TCNConfig())
        provenance=dict(base_dataset_sha256=base['manifest']['dataset_sha256'],protocol_sha256=sha(OUT/'protocol.json'),
            split_manifest_sha256=sha(OUT/'splits'/job['split']/'manifest.json'),implementation_code_sha256=code,
            job=job,train_sampling_slots=len(train),unique_training_windows=len({(i.participant_id,i.physical_trial_id) for i in train.identities}),
            validation_windows=len(val),centering_counts=d.get('centering_counts'),no_test_result_access=True)
        # The field below distinguishes sampling slots from independent trial units.
        provenance['training_participant_trial_units']=provenance.pop('unique_training_windows')
        write(folder/'run_manifest.json',provenance)
        try:
            result=(train_tcn if job['architecture']=='tcn' else train_mb)(train,val,job,folder/'training',enabled,
                d['baselines']['positive_class_weight'],provenance,cutoff)
            results.append(dict(**job,**{k:result[k] for k in ['status','best_epoch','best_validation_score','elapsed_s','epochs_entered']}))
            print(json.dumps(results[-1]),flush=True)
        except Exception as e:
            write(folder/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
            results.append(dict(**job,status='failed',error=str(e)));print(json.dumps(results[-1]),flush=True)
        write(OUT/'training_summary.json',dict(updated_at=now(),runs=results))
        del train,val,d;gc.collect();torch.cuda.empty_cache()
    for name,digest in code.items():assert sha(ROOT/'scripts'/name)==digest,'Training code changed during run'
    write(OUT/'training_summary.json',dict(updated_at=now(),runs=results,training_phase_complete=True,
        test_evaluations_during_training=0,all_requested_runs_trained=all(r['status']=='trained_with_validation' for r in results)))
    write(OUT/'progress.json',dict(status='training_complete',updated_at=now(),trained=sum(r['status']=='trained_with_validation' for r in results),scheduled=len(protocol['jobs'])))


if __name__=='__main__':run()

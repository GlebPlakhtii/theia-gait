"""Replay the selected checkpoint; verify cached test outputs and source isolation."""
import json,time
import numpy as np
import torch
from evaluate_gait_selected import ROOT,OUT,EX,BASE,write,digest
from gait_assessment_data import load_dataset
from gait_from_scratch_model import GaitTCN
from evaluate_motionbert_assessment import predict

torch.set_num_threads(2)
r=json.loads((OUT/'results.json').read_text());d=load_dataset(BASE)
d['normalization']=json.loads((EX/'splits/original/normalization.json').read_text())
cp=EX/'runs'/r['model']/'training/best.pt';s=torch.load(cp,map_location='cpu',weights_only=False)
m=GaitTCN().cuda();m.load_state_dict(s['model_state'],strict=True)
ids=np.flatnonzero((d['split']==2)&(d['method']==0)&d['input_any_valid'])
predict(m,d,ids[:4],4);torch.cuda.synchronize();start=time.perf_counter()
p,v,q=predict(m,d,ids,64);torch.cuda.synchronize();elapsed=time.perf_counter()-start
cached=np.load(EX/'runs'/r['model']/'test_predictions.npz');ci={int(i):j for j,i in enumerate(cached['indices'])};ix=[ci[int(i)] for i in ids]
diff={k:float(np.max(np.abs(a[ids]-cached[k][ix]))) for k,a in [('probabilities',p),('values',v),('evaluability',q)]}
assert diff['probabilities']<1e-5 and diff['values']<1e-3 and diff['evaluability']<1e-5,diff
for key in ['subject','trial_block','source_sha256']:
 groups={}
 for w in d['windows']:
  if w[key] is not None:groups.setdefault(w[key],set()).add(w['split'])
 assert all(len(ss)==1 for ss in groups.values()),key
audit=dict(status='passed',checkpoint_sha256=digest(cp),parameters=m.parameter_count,gpu=torch.cuda.get_device_name(),batch_size=64,
 windows=len(ids),elapsed_s=elapsed,windows_per_s=len(ids)/elapsed,excludes='Pose extraction, disk loading and video decoding; warm GPU batch inference only',
 maximum_absolute_difference=diff,all_frozen_dataset_input_hashes_verified=True,person_trial_source_separation=True,
 selected_checkpoint_reexecuted=True,clinical_validation=False)
write(OUT/'inference_verification.json',audit);np.savez_compressed(OUT/'replayed_predictions.npz',indices=ids,probabilities=p[ids],values=v[ids],evaluability=q[ids])
print(json.dumps(audit,indent=2))

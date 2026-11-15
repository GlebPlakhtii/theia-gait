"""Load the frozen common assessment dataset; model inputs exclude labels/IDs."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from motionbert_gait_common import ROOT, METHODS, sha, now, write
from gait_from_scratch_model import TCNConfig
from gait_from_scratch_training import WindowDataset
from prepare_gait_assessment_dataset import identity


def load_dataset(folder, verify_inputs=True):
    folder = Path(folder)
    read = lambda name: json.loads((folder/name).read_text())
    manifest = read('dataset_manifest.json')
    assert manifest['status'] == 'frozen'
    assert sha(folder/'dataset.npz') == manifest['dataset_sha256']
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(folder/name) == digest, name
    if verify_inputs:
        for name, digest in manifest['input_hashes'].items():
            assert sha(ROOT/name) == digest, name
        for name, digest in manifest['code_hashes'].items():
            assert sha(ROOT/name) == digest, name
    with np.load(folder/'dataset.npz', allow_pickle=False) as a:
        data = {key:a[key].copy() for key in a.files}
    data.update(manifest=manifest, windows=read('windows.json'), protocol=read('label_protocol.json'),
                normalization=read('normalization.json'), baselines=read('training_baselines.json'),
                inventory=read('clip_inventory.json'), reference_fits=read('reference_fits.json'))
    n = len(data['x']); mc=TCNConfig()
    assert len(data['windows'])==n and len(data['inventory'])==596
    assert [c['id'] for c in data['protocol']['labels']]==list(mc.classes)
    assert [o['id'] for o in data['protocol']['outcomes']]==list(mc.outcomes)
    assert np.isfinite(data['x']).all()
    assert np.array_equal(data['explicit_validity'],data['x'][...,2]>0)
    data['pose'] = np.concatenate([data['x'], data['explicit_validity'][...,None]],-1).astype(np.float32)
    context=np.zeros((n,len(mc.context_names)),np.float32)
    for i,m in enumerate(data['windows']):
        context[i,mc.views.index(m['view'] if m['view'] in mc.views else 'unknown')]=1
        context[i,len(mc.views)+mc.environments.index(m['environment'] if m['environment'] in mc.environments else 'other')]=1
    data['context']=context
    centers=np.asarray([data['normalization'][m]['center'] for m in METHODS],np.float32)
    scales=np.asarray([data['normalization'][m]['scale'] for m in METHODS],np.float32)
    counts=np.asarray([data['normalization'][m]['count'] for m in METHODS])
    data['regression_mask']=data['measurement_mask'] & (counts[data['method']]>0)
    data['standardized']=np.nan_to_num((data['values']-centers[data['method']])/scales[data['method']],nan=0).astype(np.float32)
    data['enabled']=np.array([label['classification_enabled'] for label in data['protocol']['labels']],bool)
    data['label_mask']=(data['labels']>=0)&data['enabled'][None]&(data['method'][:,None]==0)
    return data


def split_indices(data, split, primary_only=False):
    # Partial quality failures remain eligible for quality supervision. Completely
    # missing input windows never enter either model or pooled representation.
    keep=(data['split']==split)&data['input_any_valid']
    if primary_only: keep &= data['method']==0
    return np.flatnonzero(keep)


def window_dataset(data, ids):
    targets=dict(classification=np.maximum(data['labels'][ids],0).astype(np.float32),
        regression=data['standardized'][ids], evaluability=data['measurement_mask'][ids].astype(np.float32))
    masks=dict(classification=data['label_mask'][ids],regression=data['regression_mask'][ids],
        evaluability=data['quality_scope'][ids])
    return WindowDataset(torch.from_numpy(data['pose'][ids]),torch.from_numpy(data['context'][ids]),
        {k:torch.from_numpy(v) for k,v in targets.items()}, {k:torch.from_numpy(v) for k,v in masks.items()},
        torch.from_numpy(data['method'][ids].astype(np.int64)),[identity(data['windows'][i]) for i in ids])


def audit_dataset(data, folder):
    mapping=json.loads((Path(folder)/'splits.json').read_text())['subjects']
    windows=data['windows']; owners={}; partitions={field:defaultdict(set) for field in ('subject','trial_block','source_sha256')}
    keys=set()
    for i,m in enumerate(windows):
        assert m['key'] not in keys; keys.add(m['key'])
        assert m['split']==mapping.get(m['subject'],'exploratory')
        assert len(m['metric_family_records'])==17
        assert m['clinical_form_label'] is None
        assert m['end_time_s']-m['start_time_s'] <= 1.000001
        for field,parts in partitions.items():
            if m[field] is not None:parts[m[field]].add(m['split'])
        if m['subject'] and m['split']!='exploratory':
            assert owners.setdefault(m['trial_block'],m['subject'])==m['subject']
        if m['method']=='publisher':assert not data['label_mask'][i].any()
        if not m['input_eligible']:assert not data['measurement_mask'][i].any()
        for label in m['labels']:
            fit=data['reference_fits'].get(label['reference_fit'])
            if fit:
                assert len(fit['participants'])>=5 and fit['q75']>fit['q25']
                assert all(mapping[p]=='train' and p.startswith(('ljmu360_','lbmc_')) for p in fit['participants'])
                if m['split']=='train':assert m['subject'] not in fit['participants']
                assert fit['sha256']==label['reference_sha256']
    assert all(len(s)==1 for parts in partitions.values() for s in parts.values())
    for j,label in enumerate(data['protocol']['labels']):
        signs=[{m['subject'] for i,m in enumerate(windows) if m['split']=='train' and m['method']=='sapiens' and m['input_eligible'] and data['labels'][i,j]==s} for s in [0,1]]
        assert label['classification_enabled']==(min(map(len,signs))>=2)
    result=dict(audited_at=now(),status='passed',dataset_sha256=data['manifest']['dataset_sha256'],
        windows=len(windows),clips=596,method_records=609,source_trial_participant_leakage=False,
        reference_fit_count=len(data['reference_fits']),all_references_training_only=True,
        classes_enabled=int(data['enabled'].sum()),all_invalid_windows=int((~data['input_any_valid']).sum()),
        input_eligible_windows=int(data['input_eligible'].sum()),
        split_windows=dict(Counter(m['split'] for m in windows)),
        independent_clinical_labels=False)
    return result

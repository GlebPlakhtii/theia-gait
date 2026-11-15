"""Validate explicit split assignments; never turn previously exposed data into test data."""
import argparse
from common import read_csv, sha256, write_json


def validate(rows, mode='participant'):
    errors, groups, ids = [], {}, set()
    for r in rows:
        clip = r['clip_id']
        if clip in ids:
            errors.append(f'Duplicate clip_id: {clip}')
        ids.add(clip)
        split = r['split']
        if split not in {'development', 'train', 'validation', 'test', 'unassigned', 'excluded'}:
            errors.append(f'{clip}: invalid split {split}')
        if split in {'unassigned','excluded'}:
            continue
        if split == 'test' and r.get('prior_exposure') != 'none_documented':
            errors.append(f'{clip}: test requires documented no prior exposure')
        required = ['participant_id','source_recording_id','session_id','sha256','duplicate_group']
        if any(not r.get(k) or r[k] == 'unknown' for k in required):
            errors.append(f'{clip}: unresolved identity/hash/duplicate group')
        if r.get('duplicate_review_status') != 'reviewed':
            errors.append(f'{clip}: near-duplicate review incomplete')
        keys = ['source_recording_id','sha256','simultaneous_capture_group','duplicate_group']
        if mode == 'participant':
            keys += ['participant_id']
        else:
            keys += ['participant_session']
        for key in keys:
            value = r.get(key) if key != 'participant_session' else r.get('participant_id','') + '/' + r.get('session_id','')
            if not value or value == 'unknown':
                continue
            token = (key, value)
            if token in groups and groups[token] != split:
                errors.append(f'{clip}: {key} crosses {groups[token]}/{split}')
            groups[token] = split
    return errors


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/manifests/videos.csv')
    p.add_argument('--output', default='data/manifests/splits.json')
    p.add_argument('--mode', choices=['participant','session'], default='participant')
    a = p.parse_args()
    rows = read_csv(a.manifest)
    errors = validate(rows,a.mode)
    if errors:
        raise SystemExit('\n'.join(errors))
    write_json(a.output, {'status':'assignments_validated_not_an_evaluation', 'mode':a.mode,
        'manifest_sha256':sha256(a.manifest), 'test_available':any(r['split']=='test' for r in rows),
        'clips':{r['clip_id']:r['split'] for r in rows}})
    print(f'Validated {len(rows)} assignments; no random frame/repetition split')

"""Recompute confusion counts from archived predictions, without rerunning models."""
import json
from pathlib import Path
from common import sha256, write_json


def main():
    root=Path('archive/colleague'); checks=[]
    for filename in ('patient_transfer_v2_results.json','camera_trunk_v2_results.json'):
        p=root/filename; d=json.loads(p.read_text())
        targets=d['targets'] if 'targets' in d else {d['target']:d['result']}
        for target, data in targets.items():
            for method, stored in data['methods'].items():
                counts=dict(tp=0,fp=0,tn=0,fn=0)
                for r in data['records']:
                    if r['cohort'] != 'patient':
                        continue
                    y=bool(r['y']); pred=bool(r['predictions'][method]['warning'])
                    counts['tp' if y and pred else 'fn' if y else 'fp' if pred else 'tn']+=1
                checks.append({'file':filename,'sha256':sha256(p),'target':target,'method':method,
                               'recomputed':counts,'stored':{k:stored[k] for k in counts},
                               'matches':all(counts[k]==stored[k] for k in counts)})
    write_json('audit/historical_count_checks.json',{'scope':'Saved-prediction arithmetic only; not model reproduction','checks':checks})
    print(f'{sum(c["matches"] for c in checks)}/{len(checks)} historical count tables match')
    if not all(c['matches'] for c in checks):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Check complete native-frame coverage and summarize predictions, not accuracy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from validate_pose_outputs import validate


def read_csv(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--queue',default='data/manifests/pose_queue_ready.csv')
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args(); queue=read_csv(a.queue)
    schema=json.loads((a.directory/'joint_schema.json').read_text())
    names={int(i):n for i,n in schema['body_feet_23'].items()}
    report={'scope':'Prediction integrity and coverage only; not pose accuracy, true visibility or clinical validation','clips':[],'errors':[]}
    allrows=[]
    for q in queue:
        cid=q['clip_id']; dest=a.directory/cid
        if not (dest/'complete.json').exists():
            report['errors'].append(f'{cid}: no completed output');continue
        rows=read_csv(dest/'poses.csv');issues=validate(rows,[q]);report['errors']+=issues
        done=json.loads((dest/'complete.json').read_text())
        assert hashlib.sha256((dest/'poses.csv').read_bytes()).hexdigest()==done['pose_csv_sha256']
        raw=np.load(dest/'raw_308.npz'); kp=raw['keypoints'];sc=raw['scores'];frames=raw['source_frame_index'];ts=raw['timestamp_s']
        expected={int(r['source_frame_index']):float(r['timestamp_s']) for r in read_csv(q['frame_map']) if float(q['start_s'])<=float(r['timestamp_s'])<float(q['end_s'])}
        if not a.pilot and list(frames)!=list(expected):report['errors'].append(f'{cid}: incomplete native frame coverage')
        if kp.shape!=(len(frames),308,2) or sc.shape!=(len(frames),308):report['errors'].append(f'{cid}: invalid raw tensor shape')
        if len(rows)!=len(frames)*23:report['errors'].append(f'{cid}: incomplete 23-joint CSV')
        if any(abs(t-expected[int(f)])>1e-8 for f,t in zip(frames,ts)):report['errors'].append(f'{cid}: raw timestamp mismatch')
        byframe={}
        for r in rows:byframe.setdefault(int(r['source_frame_index']),set()).add(r['joint_id'])
        if any(v!=set(names.values()) for v in byframe.values()):report['errors'].append(f'{cid}: named schema mismatch')
        stats={n:{'raw_score_median':float(np.median(sc[:,i])), 'fraction_score_below_0_3':float(np.mean(sc[:,i]<.3))} for i,n in names.items()}
        report['clips'].append({'clip_id':cid,'frames':len(frames),'csv_rows':len(rows),'native_frame_first':int(frames[0]),'native_frame_last':int(frames[-1]),
            'timestamp_first':float(ts[0]),'timestamp_last':float(ts[-1]),'csv_missing_joint_rows':sum(r['x']=='' for r in rows),
            'raw_score_outside_0_1':int(np.count_nonzero((sc<0)|(sc>1))),'joint_score_summary':stats,
            'visibility':'not established by scores; manual occlusion review required'})
        allrows+=rows
    report['total_frames']=sum(c['frames'] for c in report['clips']);report['total_csv_rows']=len(allrows)
    (a.directory/'integrity_report.json').write_text(json.dumps(report,indent=2))
    if not report['errors'] and allrows:
        with (a.directory/'poses_all.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    print(json.dumps({k:v for k,v in report.items() if k!='clips'},indent=2))
    raise SystemExit(bool(report['errors']))

if __name__=='__main__':main()

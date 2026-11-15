"""Evaluate independently reviewed rep events and explicit abstention intervals.

Input JSON: {clips: [{clip_id, participant_id, session_id, duration_s,
annotation_status: 'independent_review_complete', observable_intervals: [[a,b]],
events_s: [t]}]}. Predictions: {clips: [{clip_id, events_s: [t],
active_intervals: [[a,b]]}]}. All intervals are half-open. Times are clip-relative.
Missing clip predictions are errors, never silently dropped. Empty event lists
are valid but zero coverage cannot pass the predeclared coverage gate.
"""
import argparse
import json
import math
from common import sha256, write_json


def intervals(value, duration):
    previous = 0
    for a,b in value:
        if not all(math.isfinite(x) for x in (a,b)) or not 0 <= a < b <= duration or a < previous:
            raise ValueError('Intervals must be finite, sorted, disjoint and within duration')
        previous = b
    return value


def contains(t, spans):
    return any(a <= t < b for a,b in spans)


def overlap(a,b):
    return sum(max(0, min(y,v)-max(x,u)) for x,y in a for u,v in b)


def match_events(truth, pred, tolerance):
    # Earliest-compatible ordered matching maximizes cardinality for fixed tolerance.
    i=j=tp=0
    while i<len(truth) and j<len(pred):
        if abs(truth[i]-pred[j]) <= tolerance:
            tp+=1; i+=1; j+=1
        elif pred[j] < truth[i]-tolerance:
            j+=1
        else:
            i+=1
    return tp


def evaluate(truth, predictions, tolerance=.5, min_coverage=.8):
    if not math.isfinite(tolerance) or tolerance <= 0 or not 0 < min_coverage <= 1:
        raise ValueError('Invalid tolerance or coverage gate')
    gt=truth['clips']; pr=predictions['clips']
    if not gt:
        raise ValueError('No ground truth clips')
    ids=[c['clip_id'] for c in gt]; pids=[c['clip_id'] for c in pr]
    if len(set(ids)) != len(ids) or len(set(pids)) != len(pids) or set(ids) != set(pids):
        raise ValueError('Prediction IDs must match all unique ground-truth clip IDs')
    ps={c['clip_id']:c for c in pr}; rows=[]
    for c in gt:
        if c['annotation_status'] != 'independent_review_complete':
            raise ValueError('Unreviewed/AI draft annotations cannot be evaluation truth')
        d=c['duration_s']; p=ps[c['clip_id']]
        if not math.isfinite(d) or d<=0:
            raise ValueError('Invalid duration')
        obs=intervals(c['observable_intervals'],d); active=intervals(p['active_intervals'],d)
        for events in (c['events_s'],p['events_s']):
            if events != sorted(set(events)) or any(not math.isfinite(t) or not 0 <= t < d for t in events):
                raise ValueError('Events must be finite, unique, sorted and within duration')
        if any(not contains(t,obs) for t in c['events_s']):
            raise ValueError('Ground-truth event in an unobservable interval')
        if any(not contains(t,active) for t in p['events_s']):
            raise ValueError('Prediction event emitted during declared abstention')
        scored=[t for t in p['events_s'] if contains(t,obs)]
        # Match separately across visibility spans: no match across an occlusion cut.
        tp=sum(match_events([t for t in c['events_s'] if a<=t<b],
                            [t for t in scored if a<=t<b],tolerance) for a,b in obs)
        fp=len(scored)-tp; fn=len(c['events_s'])-tp
        visible=sum(b-a for a,b in obs); covered=overlap(obs,active)
        if not visible:
            raise ValueError('No observable duration; exclude with an explicit reason before evaluation')
        rows.append({'clip_id':c['clip_id'],'participant_id':c['participant_id'],'session_id':c['session_id'],
          'tp':tp,'fp':fp,'fn':fn,'reference_events':len(c['events_s']),'predicted_events':len(scored),
          'count_absolute_error':abs(len(scored)-len(c['events_s'])),
          'observable_s':visible,'covered_s':covered,'coverage':covered/visible,
          'unsupported_events':len(p['events_s'])-len(scored),
          'event_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None})
    clusters={}
    for r in rows:
        key=r['participant_id']+'/'+r['session_id']
        clusters.setdefault(key,[]).append(r)
    summaries=[]
    for key,rr in clusters.items():
        tp,fp,fn=(sum(r[k] for r in rr) for k in ('tp','fp','fn'))
        visible=sum(r['observable_s'] for r in rr); covered=sum(r['covered_s'] for r in rr)
        summaries.append({'cluster':key,'clips':len(rr),'tp':tp,'fp':fp,'fn':fn,
            'event_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'coverage':covered/visible,'false_events_per_visible_minute':fp/(visible/60),
            'count_mae':sum(r['count_absolute_error'] for r in rr)/len(rr)})
    f1=[c['event_f1'] for c in summaries if c['event_f1'] is not None]
    return {'tolerance_s':tolerance,'min_coverage':min_coverage,'clips':rows,'clusters':summaries,
        'primary_macro_session_event_f1':sum(f1)/len(f1) if f1 else None,
        'coverage_gate_passed':all(c['coverage']>=min_coverage for c in summaries),
        'independent_participants':len({r['participant_id'] for r in rows}),
        'independent_person_sessions':len(summaries),
        'note':'Descriptive; no frame-level independence or medical correctness claim'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--truth',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    result=evaluate(json.load(open(a.truth)),json.load(open(a.predictions)))
    result.update(truth_sha256=sha256(a.truth),predictions_sha256=sha256(a.predictions))
    write_json(a.output,result)
    print(json.dumps(result,indent=2))

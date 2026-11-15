"""Training-only hierarchical sampling; metadata never enters the model tensor.

No datasets are opened here. Actual use requires the completed-data freeze and
independent participant/source/trial split audit in the experiment plan.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


PRIMARY = 'sapiens2_1b'
AUXILIARY = 'publisher'


@dataclass(frozen=True)
class WindowIdentity:
    participant_id: str
    physical_trial_id: str
    source: str
    method: str
    split: str
    known_participant: bool


def _training_groups(records):
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    trial_owners={}
    for i,record in enumerate(records):
        if record.split!='train':
            raise ValueError('Training sampling/statistics cannot consume validation or test windows')
        if record.known_participant is not True or not record.participant_id or not record.physical_trial_id:
            raise ValueError('Known participant and physical-trial identities are required')
        if record.source not in ('progait','ljmu360','lbmc'):
            raise ValueError('Unresolved/provisional source cohorts remain exploratory-only')
        if record.method not in (PRIMARY,AUXILIARY) or (record.method==AUXILIARY and record.source!='progait'):
            raise ValueError('Unsupported training pose method/source pairing')
        owner=trial_owners.setdefault(record.physical_trial_id,record.participant_id)
        if owner!=record.participant_id:
            raise ValueError('One physical trial cannot belong to different participants')
        groups[record.method][record.participant_id][record.physical_trial_id].append(i)
    return groups


def conditional_balancing_weights(records):
    """Weights sum to one per method: equal person, then trial, then window.

    For target statistics, pass only the training windows eligible for that
    particular outcome. Recompute after that mask so unavailable windows do not
    reduce an otherwise eligible person's intended weight. Fit methods separately.
    """
    groups=_training_groups(records)
    weights=np.zeros(len(records),dtype=np.float64)
    for people in groups.values():
        for trials in people.values():
            for indices in trials.values():
                weights[indices]=1/(len(people)*len(trials)*len(indices))
    return weights


def sample_training_epoch(records, epoch, seed=20261114):
    """Sample with replacement within person/trial; cap publisher at half.

    An epoch has N_primary + min(N_publisher, N_primary) draws. Within each
    method, people and then their physical trials are uniformly selected before
    selecting a window. Different camera views stay within their physical trial.
    """
    if not isinstance(epoch,int) or epoch<0:
        raise ValueError('Epoch must be a nonnegative integer')
    groups=_training_groups(records)
    primary_count=sum(r.method==PRIMARY for r in records)
    auxiliary_count=sum(r.method==AUXILIARY for r in records)
    if primary_count==0:
        raise ValueError('Publisher-only data cannot replace the primary experiment')
    draws={PRIMARY:primary_count,AUXILIARY:min(auxiliary_count,primary_count)}
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch]))
    indices=[]
    for method,count in draws.items():
        if not count:continue
        people=groups[method];person_ids=sorted(people)
        trial_ids={person:sorted(trials) for person,trials in people.items()}
        for _ in range(count):
            person=person_ids[int(rng.integers(len(person_ids)))]
            trials=trial_ids[person];trial=trials[int(rng.integers(len(trials)))]
            windows=people[person][trial]
            indices.append(windows[int(rng.integers(len(windows)))])
    indices=np.asarray(indices,dtype=np.int64);rng.shuffle(indices)
    audit={}
    for method in draws:
        selected=[records[int(i)] for i in indices if records[int(i)].method==method]
        audit[method]=dict(samples=len(selected),
            participant_draw_counts=dict(Counter(r.participant_id for r in selected)),
            physical_trial_draw_counts=dict(Counter(r.physical_trial_id for r in selected)))
    assert audit[AUXILIARY]['samples']<=len(indices)//2
    return indices,dict(seed=seed,epoch=epoch,sample_count=len(indices),methods=audit,
                        replacement=True,metadata_used_only_for_sampling=True)

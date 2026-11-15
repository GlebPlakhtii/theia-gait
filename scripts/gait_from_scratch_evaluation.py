"""Metrics for the observed-window experiment; no dataset access or model fitting.

Unknown targets require explicit masks. Classification scores are uncalibrated,
and their fixed reporting threshold is 0.5. Average precision uses a step-wise
precision-recall integral with tied scores grouped, not trapezoidal interpolation.
"""
from collections import defaultdict

import numpy as np


def _available_pairs(target, prediction, mask):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("Expected equally sized one-dimensional arrays")
    mask = np.ones(target.shape, dtype=bool) if mask is None else np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != target.shape:
        raise ValueError("Availability requires an explicit boolean mask of matching shape")
    if not np.isfinite(target[mask]).all() or not np.isfinite(prediction[mask]).all():
        raise ValueError("An available target or prediction is nonfinite")
    return target[mask], prediction[mask], int((~mask).sum())


def binary_metrics(target, probability, mask=None):
    """Report a supported class at a fixed threshold without tuning on test data.

    Counts remain visible for an absent class, but precision/recall/F1/AP are
    withheld unless both target signs occur. Participant support is a separate
    upstream eligibility gate; this function counts windows, never people.
    """
    y, score, unavailable = _available_pairs(target, probability, mask)
    if not np.isin(y, [0, 1]).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Available binary targets and probabilities must be valid")
    positive = y == 1
    predicted = score >= .5
    tp = int((positive & predicted).sum())
    fp = int((~positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    tn = int((~positive & ~predicted).sum())
    supported = bool(positive.any() and (~positive).any())
    result = dict(
        available_windows=len(y), unavailable_windows=unavailable,
        positive_windows=int(positive.sum()), negative_windows=int((~positive).sum()),
        confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn), threshold=.5,
        score_calibration="uncalibrated", supported=supported,
        unavailable_reason=None if supported else "class_lacks_both_target_signs",
        precision=None, recall=None, f1=None, average_precision=None,
        precision_recall_curve=[],
        integration="non-interpolated average precision; ties grouped",
    )
    if not supported:
        return result
    # A classifier with no positive prediction has undefined precision. Its F1
    # and recall are zero; preserve that distinction instead of inventing 1.0.
    result.update(precision=tp / (tp + fp) if tp + fp else None,
                  recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn))
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    cumulative_tp = np.cumsum(positive[order])
    ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(y) - 1]
    precisions = cumulative_tp[ends] / (ends + 1)
    recalls = cumulative_tp[ends] / positive.sum()
    result["average_precision"] = float(np.sum(np.diff(np.r_[0., recalls]) * precisions))
    result["precision_recall_curve"] = [
        dict(threshold=None, precision=1., recall=0.)
    ] + [dict(threshold=float(sorted_score[i]), precision=float(p), recall=float(r))
         for i, p, r in zip(ends, precisions, recalls)]
    return result


def numeric_metrics(target, prediction, mask=None):
    """Return errors in the caller's declared original measurement units."""
    y, estimate, unavailable = _available_pairs(target, prediction, mask)
    error = estimate - y
    return dict(
        available_windows=len(y), unavailable_windows=unavailable,
        mae=float(np.abs(error).mean()) if len(y) else None,
        median_absolute_error=float(np.median(np.abs(error))) if len(y) else None,
        mean_signed_error=float(error.mean()) if len(y) else None,
        rmse=float(np.sqrt(np.mean(error ** 2))) if len(y) else None,
        unavailable_reason=None if len(y) else "no_available_numeric_targets",
    )


def merge_review_intervals(intervals):
    """Union intervals in each explicit sequence/timebase before counting time.

    Callers supply original-source time when merging overlapping cuts, and use
    different sequence IDs for incompatible methods or distinct recordings.
    Several classes/regions flagging the same interval do not multiply seconds.
    """
    grouped = defaultdict(list)
    for interval in intervals:
        sequence = interval["sequence_id"]
        start, end = float(interval["start_time_s"]), float(interval["end_time_s"])
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("An explicit sequence/timebase identifier is required")
        if not np.isfinite([start, end]).all() or not 0 <= start < end:
            raise ValueError("Intervals require finite, ordered, nonnegative times")
        grouped[sequence].append((start, end))
    merged = []
    for sequence, times in sorted(grouped.items()):
        current = []
        for start, end in sorted(times):
            if current and start <= current[-1][1]:
                current[-1][1] = max(end, current[-1][1])
            else:
                current.append([start, end])
        merged.extend(dict(sequence_id=sequence, start_time_s=start, end_time_s=end,
                           duration_s=end - start) for start, end in current)
    return dict(intervals=merged, review_event_count=len(merged),
                review_duration_s=float(sum(x["duration_s"] for x in merged)))

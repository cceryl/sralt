"""Public-validation-only complementarity and probability reliability reports."""

import math

import torch


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def reliability(probabilities, labels, bins=10):
    p, y = probabilities.float().cpu(), labels.float().cpu()
    if not y.numel():
        return dict(samples=0, brier=None, ece=None, bins=[])
    bucket = (p * bins).long().clamp_max(bins - 1)
    rows, ece = [], 0.
    for i in range(bins):
        mask = bucket == i
        count = int(mask.sum())
        confidence = float(p[mask].mean()) if count else None
        frequency = float(y[mask].mean()) if count else None
        if count:
            ece += count / y.numel() * abs(confidence - frequency)
        rows.append(dict(bin=i, samples=count, mean_probability=confidence, positive_rate=frequency))
    return dict(samples=y.numel(), brier=float((p - y).square().mean()), ece=ece, bins=rows)


def degree_buckets(degrees):
    minimum = degrees.cpu().min(-1).values
    return {'All': torch.ones_like(minimum, dtype=torch.bool),
            'isolated_endpoint': minimum == 0, 'min_degree_1_2': (minimum >= 1) & (minimum <= 2),
            'min_degree_3_9': (minimum >= 3) & (minimum <= 9), 'min_degree_10_plus': minimum >= 10}


def complementarity(probabilities, labels, degrees):
    p, labels = probabilities.cpu(), labels.cpu()
    result = {}
    for name, mask in degree_buckets(degrees).items():
        selected, y = p[mask], labels[mask]
        c, prefix = selected.unbind(-1)
        correct_c, correct_p = (c > .5) == y.bool(), (prefix > .5) == y.bool()
        result[name] = dict(samples=y.numel(), positives=int(y.sum()),
            probability_pearson=finite(torch.corrcoef(selected.T)[0, 1]) if y.numel() > 1 else None,
            disagreement_rate=float(((c > .5) != (prefix > .5)).float().mean()) if y.numel() else None,
            csa_only_correct=int((correct_c & ~correct_p).sum()),
            prefix_only_correct=int((~correct_c & correct_p).sum()),
            both_correct=int((correct_c & correct_p).sum()), both_wrong=int((~correct_c & ~correct_p).sum()),
            csa=reliability(c, y), item_prefix_joint=reliability(prefix, y))
    return result


def calibration_by_degree(probabilities, labels, degrees):
    return {name: reliability(probabilities[mask], labels[mask])
            for name, mask in degree_buckets(degrees).items()}

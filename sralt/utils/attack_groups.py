"""Final-test reports only: shared validation-All-optimal F1; true degree is report-only."""

import math

import torch


REPORT_GROUPS = ('All', 'Head', 'Tail', 'TT', 'isolated_Tail')
PRIMARY_GROUPS = ('All', 'Tail', 'TT')
REPORT_REVISION = 'validation-all-best-f1-auc-selection-fpr05-v2'


def optimal_f1_threshold(probabilities, labels):
    """Maximize unweighted validation-All F1, including whole ties; prefer higher thresholds.

    Decisions use p >= threshold. Choose this using public validation labels only, then freeze it for test.
    It is independent of AUC-only checkpoint selection.
    """
    if not labels.numel():
        raise ValueError('Cannot select an F1 threshold from empty validation All.')
    order = probabilities.argsort(descending=True, stable=True)
    scores, y = probabilities[order], labels[order]
    ends = torch.cat([scores[:-1] != scores[1:], torch.ones(1, dtype=torch.bool, device=scores.device)])
    counts = torch.arange(1, y.numel() + 1, device=y.device)[ends]
    tp = y.cumsum(0)[ends]
    f1 = 2 * tp.double() / (counts + y.sum()).clamp_min(1)
    return float(scores[ends][f1.argmax()])


@torch.no_grad()
def grouped_attack_report(edges, probabilities, labels, data, head, tail, metrics, *, f1_threshold):
    """Five test groups share the frozen public-validation-All F1 threshold."""
    edges, probabilities, labels = edges.cpu(), probabilities.cpu(), labels.cpu()
    is_head = torch.zeros(data.num_users, dtype=torch.bool)
    is_head[head.cpu()] = True
    is_tail = torch.zeros_like(is_head)
    is_tail[tail.cpu()] = True
    degree = torch.bincount(data.shadow_social_edge_index[0].cpu(), minlength=data.num_users)
    left, right = edges
    head_count = is_head[left].long() + is_head[right].long()
    threshold = float(f1_threshold)
    weights = {'All': torch.ones_like(left), 'Head': head_count, 'Tail': 2 - head_count,
               'TT': (head_count == 0).long()}
    isolated = is_tail & (degree == 0)
    weights['isolated_Tail'] = isolated[left].long() + isolated[right].long()
    report = {}
    for group, weight in weights.items():
        indices = torch.arange(labels.numel()).repeat_interleave(weight)
        y, p = labels[indices], probabilities[indices]
        result = dict(samples=y.numel(), positives=int(y.sum()), negatives=int((y == 0).sum()),
                      f1_threshold=threshold, threshold_source='public validation All maximum F1',
                      predicted_positive_rate=float((p >= threshold).float().mean()) if y.numel() else None)
        for metric in metrics:
            name = type(metric).__name__.removeprefix('Binary')
            if name == 'TPRatFPR':
                name = f'TPR@FPR={metric.target_fpr:g}'
            value = None
            requires_both_classes = name in ('AUROC', 'ROC') or name.startswith('TPR@')
            if y.numel() and (not requires_both_classes or y.unique().numel() == 2):
                calculator = metric.clone().cpu()
                calculator.reset()
                # Integer decisions avoid torchmetrics' special handling of 0/1 thresholds.
                predictions = (p >= threshold).long() if name in ('F1Score', 'F1') else p
                calculated = float(calculator(predictions, y))
                value = calculated if math.isfinite(calculated) else None
            result[name] = value
        report[group] = result
    return report

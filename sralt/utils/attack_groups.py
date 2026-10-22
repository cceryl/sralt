"""Final-test reports only: true-degree groups never become model inputs."""

import math

import torch


@torch.no_grad()
def grouped_attack_report(edges, probabilities, labels, data, head, tail, metrics):
    """HH/HT/TT count pairs once; Tail subgroups keep the endpoint weighting."""
    edges, probabilities, labels = edges.cpu(), probabilities.cpu(), labels.cpu()
    is_head = torch.zeros(data.num_users, dtype=torch.bool)
    is_head[head.cpu()] = True
    is_tail = torch.zeros_like(is_head)
    is_tail[tail.cpu()] = True
    degree = torch.bincount(data.shadow_social_edge_index[0].cpu(), minlength=data.num_users)
    left, right = edges
    head_count = is_head[left].long() + is_head[right].long()
    weights = {'All': torch.ones_like(left), 'Head': head_count, 'Tail': 2 - head_count,
               'HH': (head_count == 2).long(), 'HT': (head_count == 1).long(), 'TT': (head_count == 0).long()}
    for name, selected in [('isolated_Tail', is_tail & (degree == 0)),
                            ('non_isolated_Tail', is_tail & (degree > 0))]:
        weights[name] = selected[left].long() + selected[right].long()
    report = {}
    for group, weight in weights.items():
        indices = torch.arange(labels.numel()).repeat_interleave(weight)
        y, p = labels[indices], probabilities[indices]
        result = dict(samples=y.numel(), positives=int(y.sum()), negatives=int((y == 0).sum()),
                      predicted_positive_rate=float((p > .5).float().mean()) if y.numel() else None)
        for metric in metrics:
            name = type(metric).__name__.removeprefix('Binary')
            if name == 'TPRatFPR':
                name = f'TPR@FPR={metric.target_fpr:g}'
            value = None
            requires_both_classes = name in ('AUROC', 'ROC') or name.startswith('TPR@')
            if y.numel() and (not requires_both_classes or y.unique().numel() == 2):
                calculator = metric.clone().cpu()
                calculator.reset()
                calculated = float(calculator(p, y))
                value = calculated if math.isfinite(calculated) else None
            result[name] = value
        report[group] = result
    return report

"""Legacy untrained public-degree ranking with a public-validation F1 threshold."""

import logging

import torch

from .base import AttackModel
from sralt.attack.data import EvalAttackDataset
from sralt.utils import DataLoader, compute_attack_metrics, head_tail_split, model_fingerprint


class PublicDegreeAttack(AttackModel):
    def __init__(self, user_feature_dim: int):
        super().__init__(user_feature_dim)
        self.register_buffer('threshold', torch.zeros((), dtype=torch.float64))

    def score(self, user_edge_index, data):
        # Training public edges only: held-out validation/hidden edges never enter degrees.
        degree = torch.bincount(data.shadow_social_edge_index[0], minlength=data.num_users).double()
        return degree[user_edge_index].log1p().sum(0)

    @torch.no_grad()
    def predict(self, user_edge_index, data):
        # A single monotone mapping gives a shared 0.5 decision threshold for every group.
        # This is a score mapping, not an estimate of calibrated membership probability.
        return (self.score(user_edge_index, data) - self.threshold).sigmoid()

    @torch.no_grad()
    def train_model(self, *, data, tail_ratio, writer):
        validation = EvalAttackDataset(data, split='val')
        scores, order = self.score(validation.edge_index, data).sort(descending=True)
        labels = validation.labels[order]
        ends = torch.cat([scores[:-1] != scores[1:], scores.new_ones(1, dtype=torch.bool)])
        indices = ends.nonzero().flatten()
        tp = labels.cumsum(0)[indices]
        f1 = 2 * tp / (indices + 1 + labels.sum())
        # Include complete tied-score groups. Ties in F1 prefer the larger threshold.
        index = indices[f1.argmax()]
        threshold = ((scores[index] + scores[index + 1]) / 2 if index + 1 < scores.numel()
                     else scores[index] - 1)
        self.threshold.copy_(threshold)
        self.statistical_diagnostics = dict(
            score='log1p(training_public_degree[u]) + log1p(training_public_degree[v])',
            degree_source='shadow_social_edge_index (attack training positives only)',
            threshold_rule='maximize All public-validation F1; ties prefer higher threshold',
            threshold=float(threshold), validation_f1=float(f1.max()),
            validation_samples=len(validation), trained_parameters=0)
        logging.info('PublicDegreeAttack: %s', self.statistical_diagnostics)

    @torch.no_grad()
    def test_model(self, *, data, tail_ratio, metrics, test_batch_size):
        head, tail = head_tail_split(data.num_users, data.target_social_edge_index, tail_ratio)
        loader = DataLoader(EvalAttackDataset(data, split='test'), batch_size=test_batch_size, shuffle=False)
        return compute_attack_metrics(data.device,
            lambda u, v: self.predict(torch.stack([u, v]), data), metrics, loader, head, tail)

    def __repr__(self):
        return (f'PublicDegreeAttack(user_feature_dim={self.user_feature_dim}, '
                f'fingerprint={model_fingerprint(self)})')

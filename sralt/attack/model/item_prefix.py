"""Relation matching on shared item-prefix sets, without raw item-pair inputs."""

import torch
from torch import nn

from .base import AttackModel
from .tail_proxy import TailProxyAttack
from sralt.utils import model_fingerprint


class ItemPrefixAttack(AttackModel):
    """Reuse fixed BCE/AdamW/balanced training and continuous endpoint features."""

    feature_blocks = 4

    def __init__(self, user_feature_dim, prefix_levels=(8,), prefix_matching=True,
                 normalize_features=True, sparse_training=True, sparse_probability=.5,
                 hidden_dim=128, dropout=.2):
        super().__init__(user_feature_dim)
        if user_feature_dim % self.feature_blocks:
            raise ValueError(f'{type(self).__name__} requires {self.feature_blocks} equally sized continuous blocks.')
        self.prefix_levels = tuple(prefix_levels)
        self.prefix_matching = prefix_matching
        self.normalize_features = normalize_features
        self.sparse_training, self.sparse_probability = sparse_training, sparse_probability
        self.hidden_dim, self.dropout = hidden_dim, dropout
        self.continuous = nn.Sequential(nn.Linear(4 * user_feature_dim, hidden_dim), nn.LeakyReLU(.2))
        self.alignment = (nn.Sequential(nn.Linear(2 * len(prefix_levels), hidden_dim // 2), nn.LeakyReLU(.2))
                          if prefix_matching else None)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim + (hidden_dim // 2 if prefix_matching else 0), hidden_dim),
            nn.LeakyReLU(.2), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.loss_fn = nn.BCEWithLogitsLoss()

    # Share the accepted training/endpoint routines without inheriting the old
    # recommendation_evidence method or its raw-item feature switches.
    endpoint_features = TailProxyAttack.endpoint_features
    sample_sparse_mask = TailProxyAttack.sample_sparse_mask
    train_model = TailProxyAttack.train_model
    test_model = TailProxyAttack.test_model
    predict = TailProxyAttack.predict

    def prefix_evidence(self, users, data):
        ids = data.side_features['prefix_ids']
        left, right = ids[users[0]], ids[users[1]]
        match = ((left[..., :, None] == right[..., None, :])
                 & (left[..., :, None] >= 0) & (right[..., None, :] >= 0)).any(-1)
        information = data.side_features['prefix_information'][left.clamp_min(0)]
        shared = (information * match).sum(-1)
        masses = data.side_features['prefix_mass']
        union = masses[users[0]] + masses[users[1]] - shared
        return torch.stack([shared, shared / union.clamp_min(1e-12)], dim=-1).flatten(1)

    def forward(self, user_edge_index, data, sparse_mask=None):
        users, order = user_edge_index.sort(dim=0)
        if sparse_mask is not None:
            sparse_mask = sparse_mask.gather(0, order)
        left, right = self.endpoint_features(users, data, sparse_mask).flatten(-2).unbind(0)
        pair = torch.cat([left, right, (left - right).abs(), left * right], dim=-1)
        representations = [self.continuous(pair)]
        if self.prefix_matching:
            representations.append(self.alignment(self.prefix_evidence(users, data)))
        return self.classifier(torch.cat(representations, dim=-1)).squeeze(-1)

    def __repr__(self):
        settings = {k: getattr(self, k) for k in ('user_feature_dim', 'prefix_levels', 'prefix_matching',
            'normalize_features', 'sparse_training', 'sparse_probability', 'hidden_dim', 'dropout')}
        return f'ItemPrefixAttack({settings}, fingerprint={model_fingerprint(self)})'

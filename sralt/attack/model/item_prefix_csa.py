"""Continuous endpoints, item-prefix matching and layer-aware user-code matching."""

import torch
from torch import nn

from .item_prefix import ItemPrefixAttack
from sralt.utils import model_fingerprint


class ItemPrefixCSAAttack(ItemPrefixAttack):
    def __init__(self, user_feature_dim, prefix_levels=(8,), user_num_quantizers=8,
                 prefix_matching=True, user_matching=True, normalize_features=True,
                 sparse_training=True, sparse_probability=.5, hidden_dim=128, dropout=.2):
        super().__init__(user_feature_dim, prefix_levels, prefix_matching, normalize_features,
                         sparse_training, sparse_probability, hidden_dim, dropout)
        self.user_num_quantizers, self.user_matching = user_num_quantizers, user_matching
        self.user_alignment = (nn.Sequential(nn.Linear(2 * user_num_quantizers, hidden_dim // 2),
            nn.LeakyReLU(.2)) if user_matching else None)
        width = hidden_dim + (int(prefix_matching) + int(user_matching)) * (hidden_dim // 2)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, hidden_dim),
            nn.LeakyReLU(.2), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def user_evidence(self, users, data):
        left, right = data.side_features['user_code_ids'][users].unbind(0)
        match = (left == right).to(data.user_features.dtype)
        information = data.side_features['user_code_information'][left]
        return torch.stack([match, match * information], dim=-1).flatten(1)

    def forward(self, user_edge_index, data, sparse_mask=None):
        users, order = user_edge_index.sort(dim=0)
        if sparse_mask is not None:
            sparse_mask = sparse_mask.gather(0, order)
        left, right = self.endpoint_features(users, data, sparse_mask).flatten(-2).unbind(0)
        pair = torch.cat([left, right, (left - right).abs(), left * right], dim=-1)
        representations = [self.continuous(pair)]
        if self.prefix_matching:
            representations.append(self.alignment(self.prefix_evidence(users, data)))
        if self.user_matching:
            representations.append(self.user_alignment(self.user_evidence(users, data)))
        return self.classifier(torch.cat(representations, dim=-1)).squeeze(-1)

    def __repr__(self):
        settings = {key: getattr(self, key) for key in ('user_feature_dim', 'prefix_levels',
            'user_num_quantizers', 'prefix_matching', 'user_matching', 'normalize_features',
            'sparse_training', 'sparse_probability', 'hidden_dim', 'dropout')}
        return f'ItemPrefixCSAAttack({settings}, fingerprint={model_fingerprint(self)})'

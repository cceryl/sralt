"""Small classifier on queried recommendation overlap, without learned user features."""

import torch.nn as nn

from .base import AttackModel
from .mlp import MLP
from sralt.utils import model_fingerprint


class RecOnlyAttack(MLP):
    # Reuse MLP's sampling, BCE training, public validation and grouped evaluation.
    def __init__(self, user_feature_dim: int, hidden_dim: int = 16):
        AttackModel.__init__(self, user_feature_dim)
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(nn.Linear(1, hidden_dim), nn.LeakyReLU(.2), nn.Linear(hidden_dim, 1))
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, user_edge_index, recommendations):
        if recommendations.is_floating_point():
            raise ValueError('RecOnlyAttack requires Top-K item IDs; use shadow/model=none.')
        left, right = recommendations[user_edge_index[0]], recommendations[user_edge_index[1]]
        count = (left[:, :, None] == right[:, None, :]).any(-1).sum(-1, keepdim=True).float()
        return self.net(count).squeeze(-1)

    def __repr__(self):
        return (f'RecOnlyAttack(user_feature_dim={self.user_feature_dim}, hidden_dim={self.hidden_dim}, '
                f'fingerprint={model_fingerprint(self)})')

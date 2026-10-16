import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttackModel
from .mlp import MLP
from sralt.utils import model_fingerprint


class SGDWithGradientClipping(torch.optim.SGD):
    """SGD with the gradient clipping used by SMIA's baseline attack script."""

    def __init__(self, params, lr: float, momentum: float, weight_decay: float, grad_clip_norm: float):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.grad_clip_norm = grad_clip_norm

    def step(self, closure=None):
        params = [param for group in self.param_groups for param in group['params']]
        torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
        return super().step(closure)


class MINER(MLP):
    """Social MINER / En_MINER from the SMIA authors' public implementation.

    Reuses MLP's training, validation, testing and probability prediction, but
    constructs a distance-based network with unweighted binary cross entropy.
    This is the social baseline, without original MINER's bilateral network.
    Reference: https://github.com/XuHao-bit/SMIA/tree/a7e79411244a574ba0b539510f105094a8956682
    """

    def __init__(
        self, user_feature_dim: int, embed_dim: int, num_recommendations: int,
        bidirectional: bool, dropout: float, leaky_relu_slope: float
    ):
        # Initialize only the distance network, avoiding MLP's much wider net.
        AttackModel.__init__(self, user_feature_dim)
        if user_feature_dim != (num_recommendations + 1) * embed_dim:
            raise ValueError('MINER expects packed user/item embeddings from shadow/model=miner.')

        self.embed_dim = embed_dim
        self.num_recommendations = num_recommendations
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.leaky_relu_slope = leaky_relu_slope
        self.attack_feature_dim = (8 if bidirectional else 4) * num_recommendations

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.net = nn.Sequential(
            nn.Linear(self.attack_feature_dim, self.attack_feature_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(self.attack_feature_dim, self.attack_feature_dim // 2),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(self.attack_feature_dim // 2, 1)
        )

    @staticmethod
    def _distances(items: torch.Tensor, user: torch.Tensor) -> torch.Tensor:
        """Return metric-major L1, L2, cosine, Bray-Curtis features [B, 4K]."""
        user = user.unsqueeze(1)
        difference = items - user
        l1 = difference.abs().sum(dim=-1)
        l2 = torch.linalg.vector_norm(difference, dim=-1)
        cosine = 1 - F.cosine_similarity(items, user, dim=-1, eps=1e-12)
        bray_curtis = l1 / (items + user).abs().sum(dim=-1).clamp_min(1e-12)
        return torch.cat([l1, l2, cosine, bray_curtis], dim=1)

    def get_attack_features(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """Compute one or both directions after canonicalizing undirected pairs.

        Input user rows contain ``[user_embedding, item_1, ..., item_K]``.
        MINER uses ``D(R(u), v)``; En_MINER appends ``D(R(v), u)``.
        """
        user1, user2 = user_edge_index
        user1, user2 = torch.minimum(user1, user2), torch.maximum(user1, user2)
        features1 = user_features[user1].reshape(-1, self.num_recommendations + 1, self.embed_dim)
        features2 = user_features[user2].reshape(-1, self.num_recommendations + 1, self.embed_dim)
        distances = self._distances(features1[:, 1:], features2[:, 0])
        if self.bidirectional:
            reverse_distances = self._distances(features2[:, 1:], features1[:, 0])
            distances = torch.cat([distances, reverse_distances], dim=1)
        return distances

    def forward(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        features = self.get_attack_features(user_edge_index, user_features)
        return self.net(features).squeeze(dim=1)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'user_feature_dim={self.user_feature_dim}, '
            f'embed_dim={self.embed_dim}, '
            f'num_recommendations={self.num_recommendations}, '
            f'bidirectional={self.bidirectional}, '
            f'dropout={self.dropout}, '
            f'leaky_relu_slope={self.leaky_relu_slope}, '
            f'fingerprint={model_fingerprint(self)})'
        )

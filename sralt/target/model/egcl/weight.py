import typing
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def l2_loss(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    """TensorFlow l2_loss uses half the squared norm, without averaging."""
    return sum((parameter.square().sum() for parameter in parameters), start=torch.tensor(0.0)) * 0.5


class AdaptiveWeight(nn.Module):
    """Learn positive-pair weights for EGCL's non-sampling square loss."""

    def __init__(self, embed_dim: int, offset: float):
        super().__init__()
        self.offset = offset
        self.network = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )
        for parameter in self.parameters():
            nn.init.trunc_normal_(parameter, std=0.01, a=-0.02, b=0.02)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        features = torch.cat([left, right, left * right], dim=1)
        return F.softplus(self.network(features).squeeze(-1)) + self.offset

    def _positive_loss(
        self, left: torch.Tensor, right: torch.Tensor, include_constant: bool
    ) -> torch.Tensor:
        scores = (left * right).sum(dim=1)
        weights = self(left, right)
        correction = (weights - 1) * scores.square() - 2 * weights * scores
        if include_constant:
            correction = correction + weights
        return correction.sum()

    def square_loss(
        self, left: torch.Tensor, right: torch.Tensor,
        edge_index: torch.Tensor, batch_size: int, include_constant: bool
    ) -> torch.Tensor:
        """Exact all-pairs square sum via Gram matrices plus positive corrections."""
        loss = ((left.t() @ left) * (right.t() @ right)).sum()
        for edges in edge_index.split(batch_size, dim=1):
            # Index inside checkpoint so edge-by-dimension activations are recomputed.
            loss = loss + typing.cast(torch.Tensor, checkpoint(
                self._edge_loss, left, right, edges, include_constant, use_reentrant=False
            ))
        return loss

    def _edge_loss(
        self, left: torch.Tensor, right: torch.Tensor,
        edges: torch.Tensor, include_constant: bool
    ) -> torch.Tensor:
        return self._positive_loss(left[edges[0]], right[edges[1]], include_constant)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import degree, scatter


class DualContrastiveLearning(nn.Module):
    """Full-node, sum-reduced InfoNCE with column-normalized neighbor views."""

    def __init__(self, temperature: float, intra_reg: float, inter_reg: float, batch_size: int):
        super().__init__()
        self.temperature = temperature
        self.intra_reg = intra_reg
        self.inter_reg = inter_reg
        self.batch_size = batch_size

    def _info_nce_chunk(
        self, anchors: torch.Tensor, candidates: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits = anchors @ candidates.t() / self.temperature
        return F.cross_entropy(logits, labels, reduction='sum')

    def info_nce(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """Chunk anchors only; every anchor still contrasts against every node."""
        view1 = F.normalize(view1, dim=1, eps=1e-6)
        view2 = F.normalize(view2, dim=1, eps=1e-6)
        loss = view1.new_zeros(())
        for start in range(0, view1.size(0), self.batch_size):
            anchors = view1[start:start + self.batch_size]
            labels = torch.arange(start, start + anchors.size(0), device=view1.device)
            # Recompute logits in backward instead of retaining an N-by-N graph.
            loss = loss + checkpoint(
                self._info_nce_chunk, anchors, view2, labels, use_reentrant=False
            )
        return loss

    def neighbor_views(
        self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor,
        behavioral_embeds: torch.Tensor, social_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply R/column_degree and S/column_degree as in the release."""
        num_users = social_embeds.size(0)
        edges = behavioral_edge_index[:, behavioral_edge_index[0] < num_users]
        users, items = edges
        item_degree = degree(items, num_nodes=behavioral_embeds.size(0)).clamp(min=1)
        weights = item_degree[items].reciprocal().unsqueeze(1)
        user_neighbors = scatter(
            behavioral_embeds[items] * weights, users, dim=0, dim_size=num_users, reduce='sum'
        )
        item_neighbors = scatter(
            behavioral_embeds[users] * weights, items - num_users, dim=0,
            dim_size=behavioral_embeds.size(0) - num_users, reduce='sum'
        )
        src, dst = social_edge_index
        social_degree = degree(dst, num_nodes=num_users).clamp(min=1)
        social_neighbors = scatter(
            social_embeds[dst] / social_degree[dst, None], src,
            dim=0, dim_size=num_users, reduce='sum'
        )
        return torch.cat([user_neighbors, item_neighbors]), social_neighbors

    def forward(
        self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor,
        behavioral_embeds: torch.Tensor, social_embeds: torch.Tensor
    ) -> torch.Tensor:
        num_users = social_embeds.size(0)
        behavioral_neighbors, social_neighbors = self.neighbor_views(
            social_edge_index, behavioral_edge_index, behavioral_embeds, social_embeds
        )
        intra_loss = (
            self.info_nce(behavioral_embeds[:num_users], behavioral_neighbors[:num_users]) +
            self.info_nce(social_embeds, social_neighbors) +
            self.info_nce(behavioral_embeds[num_users:], behavioral_neighbors[num_users:])
        )
        inter_loss = self.info_nce(behavioral_embeds[:num_users], social_embeds)
        return self.intra_reg * intra_loss + self.inter_reg * inter_loss

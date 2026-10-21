from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


@dataclass
class MeanPropagation:
    """One immutable, normalized graph shared by all branches in a batch."""

    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    self_weight: torch.Tensor
    adjacency: torch.Tensor

    def aggregate(self, embeds: torch.Tensor) -> torch.Tensor:
        if torch.are_deterministic_algorithms_enabled():
            src, dst = self.edge_index
            return scatter(embeds[src] * self.edge_weight[:, None], dst,
                           dim=0, dim_size=embeds.size(0), reduce='sum')
        return self.adjacency @ embeds


class MeanGraph:
    """Cache CSR indices once; update only masks, degrees and values per batch.

    Input edges are unique and immutable. CSR rows are message destinations.
    Zeroing a removed edge and recomputing degrees is equivalent to deleting it.
    """

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype):
        self.num_nodes = num_nodes
        self.dtype = dtype
        self.source_edge_index = edge_index
        keys = edge_index[1] * num_nodes + edge_index[0]
        order = keys.argsort()
        self.keys = keys[order]
        self.edge_index = edge_index[:, order]
        degree = torch.bincount(self.edge_index[1], minlength=num_nodes)
        self.crow_indices = torch.cat([degree.new_zeros(1), degree.cumsum(0)])
        self._full: MeanPropagation | None = None

    def propagation(self, excluded_edges: torch.Tensor | None = None) -> MeanPropagation:
        if excluded_edges is None and self._full is not None:
            return self._full

        src, dst = self.edge_index
        keep = torch.ones(src.size(0), device=src.device, dtype=self.dtype)
        if excluded_edges is not None:
            u, v = excluded_edges
            keys = torch.cat([v * self.num_nodes + u, u * self.num_nodes + v])
            positions = torch.searchsorted(self.keys, keys)
            in_range = positions < self.keys.numel()
            positions, keys = positions[in_range], keys[in_range]
            positions = positions[self.keys[positions] == keys]
            keep[positions] = 0

        degree = torch.bincount(dst[keep.bool()], minlength=self.num_nodes)
        weights = keep / degree.to(self.dtype).clamp_min(1)[dst]
        # The original DGL implementation computes these weights in float32.
        degree_float = degree.float()
        self_weight = (1 - degree_float / (degree_float + 1e-8)).to(self.dtype)
        adjacency = torch.sparse_csr_tensor(
            self.crow_indices, src, weights, size=(self.num_nodes, self.num_nodes),
            dtype=self.dtype, device=src.device
        )
        result = MeanPropagation(self.edge_index, weights, self_weight, adjacency)
        if excluded_edges is None:
            self._full = result
        return result


class SocialConv(nn.Module):
    """One friend-neighbor mean, without a residual or self loop."""

    def forward(self, graph: MeanPropagation, embeds: torch.Tensor) -> torch.Tensor:
        return graph.aggregate(embeds)


class RatingConv(nn.Module):
    """Two interaction means with the authors' degree-dependent self weights."""

    def forward(self, graph: MeanPropagation, embeds: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            embeds = graph.aggregate(embeds) + graph.self_weight[:, None] * embeds
        return embeds

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


from sralt.utils import build_normalized_adj


class SocialConv(nn.Module):
    """
    Social Convolution Layer that aggregates information from social neighbors.
    """

    def __init__(self):
        super().__init__()

        self._cached_edge_index = torch.Tensor()
        self._cached_sparse_adj = torch.Tensor()

    def forward(self, edge_index: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SocialConv layer.

        :param edge_index: Edge index tensor of shape (2, num_edges) representing social connections.
        :param embeds: Embeddings tensor of shape (num_users, embed_dim).
        """
        num_users = embeds.size(0)

        if torch.are_deterministic_algorithms_enabled():
            src, dst = edge_index
            return scatter(embeds[src], dst, dim_size=num_users, reduce='mean')

        if self._cached_edge_index is not edge_index:
            self._cached_edge_index = edge_index
            self._cached_sparse_adj = build_normalized_adj(
                edge_index=edge_index,
                num_nodes=num_users,
                edge_weight=None,
                add_self_loops=False,
                normalization='target'
            )

        return self._cached_sparse_adj @ embeds


class RatingConv(nn.Module):
    """
    Rating Convolution Layer that aggregates information from behavioral interactions.
    """

    def __init__(self):
        super().__init__()

        self._cached_edge_index = torch.Tensor()
        self._cached_sparse_adj = torch.Tensor()

    def forward(self, edge_index: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RatingConv layer.

        :param edge_index: Edge index tensor of shape (2, num_edges) representing behavioral interactions.
        :param embeds: Embeddings tensor of shape (num_nodes, embed_dim).
        :param self_weight: Self-connection weights tensor of shape (num_nodes,).
        """
        num_nodes = embeds.size(0)

        if torch.are_deterministic_algorithms_enabled():
            src, dst = edge_index

            aggregated = scatter(embeds[src], dst, dim_size=num_nodes, reduce='mean')
            embeds = aggregated + embeds

            aggregated = scatter(embeds[src], dst, dim_size=num_nodes, reduce='mean')
            embeds = aggregated + embeds

            return embeds

        if self._cached_edge_index is not edge_index:
            self._cached_edge_index = edge_index
            self._cached_sparse_adj = build_normalized_adj(
                edge_index=edge_index,
                num_nodes=num_nodes,
                edge_weight=None,
                add_self_loops=False,
                normalization='target'
            )

        aggregated = self._cached_sparse_adj @ embeds
        embeds = aggregated + embeds

        aggregated = self._cached_sparse_adj @ embeds
        embeds = aggregated + embeds

        return embeds

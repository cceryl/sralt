import torch
import torch.nn as nn


def build_sparse_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.sparse.Tensor:
    """
    Build a sparse adjacency matrix for mean aggregation.

    :param edge_index: Tensor of shape (2, num_edges).
    :param num_nodes: Total number of nodes.
    :return: Sparse adjacency matrix of shape (num_nodes, num_nodes).
    """
    src, dst = edge_index
    deg = torch.bincount(dst, minlength=num_nodes).float()
    deg_inv = 1.0 / deg.clamp(min=1)
    weights = deg_inv[dst]

    return torch.sparse_coo_tensor(torch.stack([dst, src]), weights, (num_nodes, num_nodes)).coalesce()


class SocialConv(nn.Module):
    """
    Social Convolution Layer that aggregates information from social neighbors.
    """

    def __init__(self):
        super().__init__()

        self.cached_edge_index = None
        self.cached_sparse_adj = None

    def forward(self, edge_index: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SocialConv layer.

        :param edge_index: Edge index tensor of shape (2, num_edges) representing social connections.
        :param embeds: Embeddings tensor of shape (num_users, embed_dim).
        """
        num_users = embeds.size(0)

        if self.cached_edge_index is None or self.cached_edge_index is not edge_index:
            self.cached_edge_index = edge_index
            self.cached_sparse_adj = build_sparse_adj(edge_index, num_users)

        return self.cached_sparse_adj @ embeds


class RatingConv(nn.Module):
    """
    Rating Convolution Layer that aggregates information from behavioral interactions.
    """

    def __init__(self):
        super().__init__()

        self.cached_edge_index = None
        self.cached_sparse_adj = None

    def forward(self, edge_index: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RatingConv layer.

        :param edge_index: Edge index tensor of shape (2, num_edges) representing behavioral interactions.
        :param embeds: Embeddings tensor of shape (num_nodes, embed_dim).
        :param self_weight: Self-connection weights tensor of shape (num_nodes,).
        """
        num_nodes = embeds.size(0)

        if self.cached_edge_index is None or self.cached_edge_index is not edge_index:
            self.cached_edge_index = edge_index
            self.cached_sparse_adj = build_sparse_adj(edge_index, num_nodes)

        aggregated = self.cached_sparse_adj @ embeds
        embeds = aggregated + embeds

        aggregated = self.cached_sparse_adj @ embeds
        embeds = aggregated + embeds

        return embeds

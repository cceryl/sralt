import math
import typing

import torch
import torch.nn as nn
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.models.lightgcn import LGConv, BPRLoss


class AttentiveAggregation(nn.Module):
    """
    Attentive Embedding Readout module in EGCL.
    """

    def __init__(self, embed_dim: int):
        """
        :param embed_dim: Dimension of the embeddings.
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, embeds: torch.Tensor, layer_embeds: torch.Tensor) -> torch.Tensor:
        """
        :param embeds: Initial embeddings of shape (num_nodes, embed_dim).
        :param layer_embeds: Embeddings from each layer of shape (num_nodes, num_layers, embed_dim).
        :return: Final embeddings after attentive readout of shape (num_nodes, embed_dim).
        """
        all_embeds = torch.cat([embeds.unsqueeze(1), layer_embeds], dim=1)
        embeds = embeds.unsqueeze(1).expand_as(all_embeds)

        weights: torch.Tensor = self.attention(torch.cat([embeds, all_embeds], dim=-1))
        return (weights * all_embeds).sum(dim=1)


class AdaptivePropagation(nn.Module):
    """
    Adaptive Neighbor Aggregation module in EGCL.
    """

    def __init__(self, num_layers: int):
        """
        :param num_layers: Number of propagation layers.
        """
        super().__init__()

        self.num_layers = num_layers
        self.conv = LGConv(normalize=False)

        self._edge_index: torch.Tensor | None = None
        self._num_nodes = -1
        self._weights = torch.tensor([])
        self._degrees = torch.tensor([])

    def _get_weights(self, edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge weights and normalized degree for adaptive propagation.
        :param edge_index: Edge indices of shape (2, num_edges).
        :param num_nodes: Total number of nodes in the graph.
        :return: Edge weights of shape (num_edges,) and normalized degree of shape (num_nodes,).
        """
        if self._edge_index is edge_index and self._num_nodes == num_nodes:
            return self._weights, self._degrees

        self._edge_index = edge_index
        self._num_nodes = num_nodes

        _, self._weights = typing.cast(
            tuple[torch.Tensor, torch.Tensor],
            gcn_norm(edge_index, num_nodes=num_nodes, add_self_loops=False)
        )

        degrees = degree(edge_index[0], num_nodes=num_nodes)
        self._degrees = degrees.clamp(min=1e-7).log().sigmoid()

        return self._weights, self._degrees

    def forward(self, embeds: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param embeds: Initial embeddings of shape (num_nodes, embed_dim).
        :param edge_index: Edge indices of shape (2, num_edges).
        :param degrees: Normalized degree tensor of shape (num_nodes,).
        :return: Embeddings after adaptive propagation of shape (num_nodes, num_layers, embed_dim).
        """
        all_embeds = []
        edge_weights, degrees = self._get_weights(edge_index, embeds.size(0))

        for layer in range(self.num_layers):
            embeds = self.conv(embeds, edge_index, edge_weights)
            weights = (-degrees).exp() * (degrees ** layer) / float(math.factorial(layer))
            all_embeds.append(embeds * weights.unsqueeze(1))

        return torch.stack(all_embeds, dim=1)


class GraphAdaptivePropagation(nn.Module):
    """
    Graph Adaptive Propagation module in EGCL.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int, num_layers: int, decay: float):
        """
        :param num_users: Number of users.
        :param num_items: Number of items.
        :param embed_dim: Dimension of the embeddings.
        :param num_layers: Number of propagation layers.
        :param decay: Decay factor for the BPR loss.
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.social_embeds = nn.Embedding(num_users, embed_dim)
        self.behavioral_embeds = nn.Embedding(num_users + num_items, embed_dim)

        self.social_propagate = AdaptivePropagation(num_layers)
        self.behavioral_propagate = AdaptivePropagation(num_layers)

        self.social_readout = AttentiveAggregation(embed_dim)
        self.behavioral_user_readout = AttentiveAggregation(embed_dim)
        self.behavioral_item_readout = AttentiveAggregation(embed_dim)

        self.loss_fn = BPRLoss(lambda_reg=decay)

    def forward(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param social_edge_index: Undirected edge index of shape (2, num_social_edges) for the user-user social graph.
        :param behavioral_edge_index: Undirected edge index of shape (2, num_behavioral_edges) for the user-item behavioral graph.
        :return: Behavioral embeddings of shape (num_users + num_items, embed_dim) and social embeddings of shape (num_users, embed_dim).
        """
        social_layer_embeds = self.social_propagate(self.social_embeds.weight, social_edge_index)
        behavioral_layer_embeds = self.behavioral_propagate(self.behavioral_embeds.weight, behavioral_edge_index)

        social_user_embeds = self.social_readout(self.social_embeds.weight, social_layer_embeds)
        behavioral_user_embeds = self.behavioral_user_readout(
            self.behavioral_embeds.weight[:self.num_users], behavioral_layer_embeds[:self.num_users]
        )
        behavioral_item_embeds = self.behavioral_item_readout(
            self.behavioral_embeds.weight[self.num_users:], behavioral_layer_embeds[self.num_users:]
        )

        return torch.cat([behavioral_user_embeds, behavioral_item_embeds], dim=0), social_user_embeds

    def compute_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
        Compute the BPR loss for training.
        :param pos_scores: Predicted scores for positive samples of shape (batch_size,).
        :param neg_scores: Predicted scores for negative samples of shape (batch_size,).
        :param nodes: Node indices corresponding to the scores of shape (3 * batch_size,). 
        :return: BPR loss value as a scalar tensor.
        """
        user_nodes = nodes[nodes < self.num_users]

        social_node_embeds = self.social_embeds(user_nodes)
        behavioral_node_embeds = self.behavioral_embeds(nodes)
        node_embeds = torch.cat([social_node_embeds, behavioral_node_embeds], dim=0)

        return self.loss_fn(pos_scores, neg_scores, node_embeds)

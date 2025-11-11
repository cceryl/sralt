from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TargetModel(nn.Module, ABC):
    """
    Base class for target social recommendation models.
    """

    def __init__(self, num_users: int, num_items: int):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items

    @abstractmethod
    def forward(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the target model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape specified by the actual model.
        """
        pass

    @abstractmethod
    def get_embedding(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings of the target model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Embedding tensor of shape (num_nodes, embed_dim).
        """
        pass

    @abstractmethod
    def calculate_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the target model.

        :param pos_scores: Score tensor for positive edges.
        :param neg_scores: Score tensor for negative edges.
        :param nodes: Node tensor of shape (num_nodes_in_batch,) representing nodes in the current batch.
        :return: Loss tensor of shape specified by the actual model.
        """
        pass

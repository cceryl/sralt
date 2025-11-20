import torch
import torch_geometric.nn as gnn

from .base import ShadowModel

from . import register_shadow_model
from sralt.utils import calculate_model_fingerprint


@register_shadow_model('Prototype')
class Prototype(ShadowModel):
    """
    A prototype shadow model using RVQ to augment social connections with behavioral interactions.
    TODO: This is a prototype implementation of an enhanced shadow model.
    """

    def __init__(self, num_users: int, num_items: int):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        """
        super().__init__(num_users, num_items)

    def forward(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the target model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape TODO.
        """
        raise NotImplementedError

    def get_embedding(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings of the target model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Embedding tensor of shape TODO.
        """
        raise NotImplementedError

    def calculate_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
        Compute loss using Bayesian Personalized Ranking (BPR) loss and L2 regularization loss.

        :param pos_scores: Score tensor for positive edges of shape TODO.
        :param neg_scores: Score tensor for negative edges of shape TODO.
        :param nodes: Node tensor of shape (num_nodes_in_batch,) representing nodes in the current batch.
        :return: The total loss tensor.
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_user_features(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the user features learned by the shadow model for attack purposes.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: A tensor of shape (num_users, feature_dim) representing user features.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        String representation and fingerprint of the target model.
        """
        return (
            f'Prototype('
            f'num_users={self.num_users}, '
            f'num_items={self.num_items}, '
            f'fingerprint={calculate_model_fingerprint(self)})'
        )

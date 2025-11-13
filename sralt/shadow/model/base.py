from abc import abstractmethod

import torch

from sralt.target.model import TargetModel


class ShadowModel(TargetModel):
    """
    Base class for shadow social recommendation models.
    """

    def __init__(self, num_users: int, num_items: int):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        """
        super().__init__(num_users, num_items)

    @torch.no_grad()
    @abstractmethod
    def get_user_features(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the user features learned by the shadow model for attack purposes.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: A tensor of shape (num_users, feature_dim) representing user features.
        """
        pass

from abc import abstractmethod

import torch
import torch.nn as nn


class AttackModel(nn.Module):
    """
    Base class for social connection attack models.
    """

    def __init__(self, user_feature_dim: int):
        """
        Initialize the AttackModel.

        :param user_feature_dim: Dimension of user features.
        """
        super().__init__()

        self.user_feature_dim = user_feature_dim

    @abstractmethod
    def forward(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate the user social connection logits.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: The user social connection logits tensor of shape (batch_size,).
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def predict(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Predict the attack outcome.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: The predicted probability tensor of shape (batch_size,).
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation and fingerprint of the attack model.
        """
        pass

import torch
import torch.nn as nn

from . import register_attack_model
from .base import AttackModel
from sralt.utils import calculate_model_fingerprint


@register_attack_model('SMIA')
class SMIAAttackModel(AttackModel):
    """
    SMIA attack model for social connection prediction.
    """

    def __init__(self, user_feature_dim: int, dropout: float = 0.5, leaky_relu_slope: float = 0.2):
        """
        Initialize the SMIAAttackModel.

        :param user_feature_dim: Dimension of user features.
        :param dropout: Dropout rate.
        :param leaky_relu_slope: Slope for LeakyReLU activation.
        """
        super().__init__(user_feature_dim)
        self.dropout = dropout
        self.leaky_relu_slope = leaky_relu_slope

        self.net = nn.Sequential(
            nn.Linear(4 * user_feature_dim, 4 * user_feature_dim, bias=True),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(4 * user_feature_dim, 2 * user_feature_dim, bias=True),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(2 * user_feature_dim, 1, bias=True)
        )

    def forward(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate the user social connection logits.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: The user social connection logits tensor of shape (batch_size,).
        """
        user1, user2 = user_edge_index
        user1_features = user_features[user1]
        user2_features = user_features[user2]

        diff_features = torch.abs(user1_features - user2_features)
        prod_features = user1_features * user2_features

        combined_features = torch.cat([user1_features, user2_features, diff_features, prod_features], dim=1)

        logits: torch.Tensor = self.net(combined_features)
        return logits.squeeze(dim=1)

    @torch.no_grad()
    def predict(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Predict the attack outcome.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: The predicted probability tensor of shape (batch_size,).
        """
        logits = self.forward(user_edge_index, user_features)
        probs = torch.sigmoid(logits)
        return probs

    def __repr__(self) -> str:
        """
        String representation and fingerprint of the attack model.
        """
        return (
            f'SMIAAttackModel('
            f'user_feature_dim={self.user_feature_dim}, '
            f'dropout={self.dropout}, '
            f'leaky_relu_slope={self.leaky_relu_slope}, '
            f'fingerprint={calculate_model_fingerprint(self)})'
        )

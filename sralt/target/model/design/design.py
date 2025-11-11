import torch
import torch.nn.functional as F

from .twin import TwinGCN
from .social import SocialGCN
from .rating import RatingGCN
from sralt.target.model import register_target_model, TargetModel
from sralt.utils import calculate_model_fingerprint


@register_target_model('DESIGN')
class DESIGN(TargetModel):
    """
    Distillation Enhanced Social Graph Network for recommendation.
    """

    def __init__(
        self, num_users: int, num_items: int, embed_dim: int,
        user_fusion_layers: list[int], item_fusion_layers: list[int],
        decay: float = 1e-4, distill_weight: float = 1.0
    ):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embeddings.
        :param user_fusion_layers: Indices of layers to sum when computing the final embedding for users.
        :param item_fusion_layers: Indices of layers to sum when computing the final embedding for items.
        :param decay: L2 regularization coefficient.
        :param distill_weight: Weight for the distillation loss.
        """
        super().__init__(num_users, num_items)

        self.social = SocialGCN(num_users, num_items, embed_dim, user_fusion_layers)
        self.rating = RatingGCN(num_users, num_items, embed_dim, user_fusion_layers, item_fusion_layers)
        self.twin = TwinGCN(num_users, num_items, embed_dim, user_fusion_layers, item_fusion_layers)

        self.decay = decay
        self.distill_weight = distill_weight

    def forward(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DESIGN model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Stack of score tensors from SocialGCN, RatingGCN, and TwinGCN of shape (num_target_edges, 3).
        """
        social_scores: torch.Tensor = self.social(social_edge_index, target_edge_index)
        rating_scores: torch.Tensor = self.rating(behavioral_edge_index, target_edge_index)
        twin_scores: torch.Tensor = self.twin(social_edge_index, behavioral_edge_index, target_edge_index)

        return torch.stack([social_scores, rating_scores, twin_scores], dim=1)

    def get_embedding(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings of TwinGCN model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Embedding tensor of shape (num_nodes, embed_dim).
        """
        return self.twin.get_embedding(social_edge_index, behavioral_edge_index)

    def calculate_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
        Compute loss using Bayesian Personalized Ranking (BPR) loss, L2 regularization, and distillation loss.
        Note that pos_scores[i] and neg_scores[i] correspond to the same user.

        :param pos_scores: Score tensor of shape (num_samples, 3).
        :param neg_scores: Score tensor of shape (num_samples, 3).
        :param nodes: Node indices tensor of shape (3 * num_samples,).
        :return: Computed loss tensor.
        """
        # BPR loss
        diff_scores = pos_scores - neg_scores
        rank_loss = -F.logsigmoid(diff_scores).mean()

        # L2 regularization
        batch_size = pos_scores.size(0)
        reg_loss = (
            self.social.embeds.weight[nodes].norm(2).pow(2) +
            self.rating.embeds.weight[nodes].norm(2).pow(2) +
            self.twin.embeds.weight[nodes].norm(2).pow(2)
        ) / batch_size

        # Distillation loss
        log_probs = F.log_softmax(diff_scores, dim=0)
        distill_loss = torch.scalar_tensor(0.0, device=nodes.device)

        for teacher, student in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]:
            teacher_log_probs = log_probs[:, teacher].detach()
            student_log_probs = log_probs[:, student]
            distill_loss += F.kl_div(student_log_probs, teacher_log_probs, reduction='batchmean', log_target=True)

        return rank_loss + self.decay * reg_loss + self.distill_weight * distill_loss

    def __repr__(self) -> str:
        return (
            f'DESIGN(num_users={self.num_users}, '
            f'num_items={self.num_items}, '
            f'embed_dim={self.twin.embeds.embedding_dim}, '
            f'user_fusion_layers={self.twin.user_fusion_layers}, '
            f'item_fusion_layers={self.twin.item_fusion_layers}, '
            f'decay={self.decay}, '
            f'distill_weight={self.distill_weight}, '
            f'fingerprint={calculate_model_fingerprint(self)})'
        )

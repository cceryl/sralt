import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class DualContrastiveLearning(nn.Module):
    """
    Augmentation-free Dual Contrastive Learning module in EGCL.
    """

    def __init__(self, temperature: float, intra_reg: float, inter_reg: float):
        """
        :param temperature: Temperature parameter for InfoNCE loss.
        :param intra_reg: Intra-view regularization coefficient.
        :param inter_reg: Inter-view regularization coefficient.
        """
        super().__init__()

        self.temperature = temperature
        self.intra_reg = intra_reg
        self.inter_reg = inter_reg

    def info_nce(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        Compute the InfoNCE loss between two views.
        :param view1: Tensor of shape (batch_size, embed_dim) for the first view.
        :param view2: Tensor of shape (batch_size, embed_dim) for the second view.
        :return: Scalar tensor representing the InfoNCE loss.
        """
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        logits = (view1 @ view2.t()) / self.temperature
        labels = torch.arange(view1.size(0), device=view1.device)

        return F.cross_entropy(logits, labels)

    def forward(
        self,
        social_edge_index: torch.Tensor,
        behavioral_edge_index: torch.Tensor,
        behavioral_embeds: torch.Tensor,
        social_embeds: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor
    ) -> torch.Tensor:
        """
        :param social_edge_index: Undirected edge index of shape (2, num_social_edges) for the user-user social graph.
        :param behavioral_edge_index: Undirected edge index of shape (2, num_behavioral_edges) for the user-item behavioral graph.
        :param behavioral_embeds: Behavioral embeddings of shape (num_users + num_items, embed_dim).
        :param social_embeds: Social embeddings of shape (num_users, embed_dim).
        :param users: Tensor of shape (num_users,) representing the user indices.
        :param items: Tensor of shape (num_items,) representing the item indices.
        :return: Scalar tensor representing the total loss.
        """
        num_users = social_embeds.size(0)
        num_items = behavioral_embeds.size(0) - num_users

        src, tgt = social_edge_index
        social_neighbor_embeds = scatter(social_embeds[src], tgt, dim_size=num_users, reduce='mean')

        src, tgt = behavioral_edge_index
        behavioral_neighbor_embeds = scatter(behavioral_embeds[src], tgt, dim_size=num_users+num_items, reduce='mean')

        inter_loss = self.info_nce(social_embeds[users], behavioral_embeds[users])
        intra_loss = (
            self.info_nce(social_embeds[users], social_neighbor_embeds[users]) +
            self.info_nce(behavioral_embeds[users], behavioral_neighbor_embeds[users]) +
            self.info_nce(behavioral_embeds[items], behavioral_neighbor_embeds[items])
        ) / 3.0

        return self.inter_reg * inter_loss + self.intra_reg * intra_loss

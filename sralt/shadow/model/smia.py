import torch
import torch_geometric.nn as gnn

from .base import ShadowModel

from . import register_shadow_model
from sralt.utils import calculate_model_fingerprint, build_normalized_adj


@register_shadow_model('SMIA')
class SMIAShadow(ShadowModel):
    """
    Shadow model implementing the SMIA dual-branch shadow model.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int, num_layers: int, decay: float):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embedding vectors.
        :param num_layers: Number of LightGCN layers.
        :param decay: Weight decay factor for regularization.
        """
        super().__init__(num_users, num_items)

        self.num_nodes = num_users + num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.decay = decay

        self.social_branch = gnn.models.LightGCN(self.num_nodes, embed_dim, num_layers, normalize=False)
        self.behavioral_branch = gnn.models.LightGCN(self.num_nodes, embed_dim, num_layers, normalize=False)

        self.cached_social_edge_index = torch.Tensor()
        self.cached_social_sparse_adj = torch.Tensor()
        self.cached_behavioral_edge_index = torch.Tensor()
        self.cached_behavioral_sparse_adj = torch.Tensor()

    def get_sparse_adj(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute and cache the normalized sparse adjacency matrices for both social and behavioral graphs.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: A tuple containing the sparse adjacency matrices for social and behavioral graphs.s
        """
        if self.cached_social_edge_index is not social_edge_index:
            self.cached_social_edge_index = social_edge_index
            self.cached_social_sparse_adj = build_normalized_adj(
                edge_index=social_edge_index,
                num_nodes=self.num_nodes,
                edge_weight=None,
                add_self_loops=False,
                normalization='symmetric'
            )

        if self.cached_behavioral_edge_index is not behavioral_edge_index:
            self.cached_behavioral_edge_index = behavioral_edge_index
            self.cached_behavioral_sparse_adj = build_normalized_adj(
                edge_index=behavioral_edge_index,
                num_nodes=self.num_nodes,
                edge_weight=None,
                add_self_loops=False,
                normalization='symmetric'
            )

        return self.cached_social_sparse_adj, self.cached_behavioral_sparse_adj

    def forward(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the target model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape (2, num_target_edges), where the rows correspond to social and behavioral scores respectively.
        """
        social_adj, behavioral_adj = self.get_sparse_adj(social_edge_index, behavioral_edge_index)

        social_scores = self.social_branch(social_adj, target_edge_index)
        behavioral_scores = self.behavioral_branch(behavioral_adj, target_edge_index)

        return torch.stack([social_scores, behavioral_scores])

    def get_embedding(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings of the target model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Embedding tensor of shape (num_nodes, 2 * embed_dim) where the halves correspond to social and behavioral embeddings respectively.
        """
        social_adj, behavioral_adj = self.get_sparse_adj(social_edge_index, behavioral_edge_index)

        social_embeds = self.social_branch.get_embedding(social_adj)
        behavioral_embeds = self.behavioral_branch.get_embedding(behavioral_adj)

        return torch.cat([social_embeds, behavioral_embeds], dim=1)

    def calculate_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
        Compute loss using Bayesian Personalized Ranking (BPR) loss and L2 regularization loss.

        :param pos_scores: Score tensor for positive edges of shape (2, num_positive_edges).
        :param neg_scores: Score tensor for negative edges of shape (2, num_negative_edges).
        :param nodes: Node tensor of shape (num_nodes_in_batch,) representing nodes in the current batch.
        :return: The total loss tensor.
        """
        return (
            self.social_branch.recommendation_loss(pos_scores[0], neg_scores[0], nodes, lambda_reg=self.decay) +
            self.behavioral_branch.recommendation_loss(pos_scores[1], neg_scores[1], nodes, lambda_reg=self.decay)
        )

    @torch.no_grad()
    def get_user_features(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the user features learned by the shadow model for attack purposes.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: A tensor of shape (num_users, feature_dim) representing user features.
        """
        social_adj, behavioral_adj = self.get_sparse_adj(social_edge_index, behavioral_edge_index)

        social_embeds = self.social_branch.get_embedding(social_adj)
        behavioral_embeds = self.behavioral_branch.get_embedding(behavioral_adj)
        social_user_embeds = social_embeds[:self.num_users]
        behavioral_user_embeds = behavioral_embeds[:self.num_users]

        user_item_features = []
        user_item_edge_index = behavioral_edge_index[:, behavioral_edge_index[0] < self.num_users]

        for u in range(self.num_users):
            item_indices = user_item_edge_index[1, user_item_edge_index[0] == u]
            item_embeds = (social_embeds[item_indices] + behavioral_embeds[item_indices]) / 2
            items_mean = item_embeds.mean(dim=0)
            items_var = item_embeds.var(dim=0)
            user_item_features.append(torch.cat([items_mean, items_var]))

        user_item_features = torch.stack(user_item_features)
        user_features = torch.cat([social_user_embeds, behavioral_user_embeds, user_item_features], dim=1)
        return user_features

    def __repr__(self) -> str:
        """
        String representation and fingerprint of the model.
        """
        return (
            f'SMIAShadowModel('
            f'num_users={self.num_users}, '
            f'num_items={self.num_items}, '
            f'embed_dim={self.embed_dim}, '
            f'num_layers={self.num_layers}, '
            f'decay={self.decay}, '
            f'fingerprint={calculate_model_fingerprint(self)})'
        )

import torch
import torch.nn as nn

from .layers import SocialConv, RatingConv, MeanPropagation


class TwinGCN(nn.Module):
    """
    Twin Graph Convolutional Network combining social and rating information for recommendation.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int, user_fusion_layers: list[int], item_fusion_layers: list[int]):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embeddings.
        :param user_fusion_layers: Indices of layers to sum when computing the final embedding for users.
        :param item_fusion_layers: Indices of layers to sum when computing the final embedding for items.
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.user_fusion_layers = user_fusion_layers
        self.item_fusion_layers = item_fusion_layers

        self.embeds = nn.Embedding(num_users + num_items, embed_dim)
        nn.init.normal_(self.embeds.weight, std=0.01)
        self.social = SocialConv()
        self.rating = RatingConv()

    def get_embedding(self, social_graph: MeanPropagation, rating_graph: MeanPropagation) -> torch.Tensor:
        """
        Get user and item embeddings after social and rating convolutions.

        :param social_graph: Prepared friend-neighbor mean propagation.
        :param rating_graph: Prepared interaction propagation, shared with RatingGCN.
        :return: Embedding tensor of shape (num_nodes, embed_dim).
        """
        social_users = self.embeds.weight[:self.num_users]
        rating_embeds = self.embeds.weight
        all_user_embeds = [social_users]
        for _ in range(max(self.user_fusion_layers)):
            social_users = self.social(social_graph, social_users)
            rating_embeds = self.rating(rating_graph, rating_embeds)
            all_user_embeds.append(0.5 * social_users + 0.5 * rating_embeds[:self.num_users])

        all_user_embeds = [all_user_embeds[i] for i in self.user_fusion_layers]
        user_embeds = torch.stack(all_user_embeds).sum(dim=0)
        item_embeds = self.embeds.weight[self.num_users:]
        final_embeds = torch.cat([user_embeds, item_embeds], dim=0)

        return final_embeds

    def forward(self, social_graph: MeanPropagation, rating_graph: MeanPropagation, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TwinGCN model.

        :param social_graph: Prepared friend-neighbor mean propagation.
        :param rating_graph: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape (num_target_edges,).
        """
        embeds = self.get_embedding(social_graph, rating_graph)

        src, dst = target_edge_index
        scores = (embeds[src] * embeds[dst]).sum(dim=1)

        return scores

import torch
import torch.nn as nn

from .layers import RatingConv


class RatingGCN(nn.Module):
    """
    Rating Graph Convolutional Network for recommendation.
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
        self.layer = RatingConv()

    def get_embedding(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get user and item embeddings after rating convolution.

        :param edge_index: Undirected edge index tensor of shape (2, num_edges) representing behavioral interactions.
        :return: Embedding tensor of shape (num_nodes, embed_dim).
        """
        embeds = self.embeds.weight
        all_embeds = [embeds]

        for _ in range(max(self.user_fusion_layers + self.item_fusion_layers)):
            embeds = self.layer(edge_index, embeds)
            all_embeds.append(embeds)

        all_user_embeds = [all_embeds[i][:self.num_users] for i in self.user_fusion_layers]
        all_item_embeds = [all_embeds[i][self.num_users:] for i in self.item_fusion_layers]
        user_embeds = torch.stack(all_user_embeds).mean(dim=0)
        item_embeds = torch.stack(all_item_embeds).mean(dim=0)
        final_embeds = torch.cat([user_embeds, item_embeds], dim=0)

        return final_embeds

    def forward(self, edge_index: torch.Tensor, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RatingGCN model.

        :param edge_index: Undirected edge index tensor of shape (2, num_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape (num_target_edges,).
        """
        embeds = self.get_embedding(edge_index)

        src, dst = target_edge_index
        scores = (embeds[src] * embeds[dst]).sum(dim=1)

        return scores

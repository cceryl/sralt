import torch
import torch.nn as nn

from .layers import SocialConv, MeanPropagation


class SocialGCN(nn.Module):
    """
    Social Graph Convolutional Network for recommendation.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int, user_fusion_layers: list[int]):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embeddings.
        :param user_fusion_layers: Indices of layers to sum when computing the final embedding for users.
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.user_fusion_layers = user_fusion_layers

        self.embeds = nn.Embedding(num_users + num_items, embed_dim)
        nn.init.normal_(self.embeds.weight, std=0.01)
        self.layer = SocialConv()

    def get_embedding(self, graph: MeanPropagation) -> torch.Tensor:
        """
        Get user and item embeddings after social convolution.

        :param graph: Prepared friend-neighbor mean propagation.
        :return: Embedding tensor of shape (num_nodes, embed_dim).
        """
        user_embeds = self.embeds.weight[:self.num_users]
        all_user_embeds = [user_embeds]

        for _ in range(max(self.user_fusion_layers)):
            user_embeds = self.layer(graph, user_embeds)
            all_user_embeds.append(user_embeds)

        user_embeds = torch.stack([all_user_embeds[i] for i in self.user_fusion_layers]).sum(dim=0)
        item_embeds = self.embeds.weight[self.num_users:]
        final_embeds = torch.cat([user_embeds, item_embeds], dim=0)

        return final_embeds

    def forward(self, graph: MeanPropagation, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SocialGCN model.

        :param graph: Prepared friend-neighbor mean propagation.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape (num_target_edges,).
        """
        embeds = self.get_embedding(graph)

        src, dst = target_edge_index
        scores = (embeds[src] * embeds[dst]).sum(dim=1)

        return scores

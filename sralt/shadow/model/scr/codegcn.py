import torch
import torch.nn as nn
from torch_geometric.nn.models.lightgcn import LGConv, BPRLoss


class CodeGCN(nn.Module):
    """
    CodeGCN uses user codes and code embeddings to reconstruct user embeddings before feeding them into a LightGCN.
    """

    def __init__(
        self,
        num_codes: int,
        num_users: int,
        num_items: int,
        embed_dim: int,
        num_layers: int,
        lambda_reg: float = 1e-4
    ):
        """
        :param num_codes: Number of codes.
        :param num_items: Number of items.
        :param num_users: Number of users.
        :param embed_dim: Dimension of embeddings.
        :param num_layers: Number of LightGCN convolution layers.
        :param lambda_reg: Regularization coefficient for BPR loss.
        """
        super().__init__()

        self.num_codes = num_codes
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.lambda_reg = lambda_reg

        self.code_embeds = nn.Embedding(num_codes, embed_dim)
        self.item_embeds = nn.Embedding(num_items, embed_dim)
        self.layers = nn.ModuleList([LGConv(normalize=True) for _ in range(num_layers)])
        self.loss_fn = BPRLoss(lambda_reg=lambda_reg)

        self._cached_edge_index = torch.Tensor()
        self._cached_sparse_adj = torch.Tensor()

    def forward(self, edge_index: torch.Tensor, user_codes: torch.Tensor, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get user and item embeddings.

        :param edge_index: Edge index for the graph.
        :param user_codes: Tensor of shape (num_users, num_codes_per_user) containing code indices for each user.
        :param target_edge_index: Edge index for target interactions.
        :return: Scores for the target interactions.
        """
        embeds = self.get_embedding(edge_index, user_codes)

        src = embeds[target_edge_index[0]]
        dst = embeds[target_edge_index[1]]

        return (src * dst).sum(dim=-1)

    def get_embedding(self, edge_index: torch.Tensor, user_codes: torch.Tensor) -> torch.Tensor:
        """
        Get the reconstructed embeddings after GCN propagation.

        :param edge_index: Edge index for the graph.
        :param user_codes: Tensor of shape (num_users, num_codes_per_user) containing code indices for each user.
        :return: Reconstructed embeddings of shape (num_users + num_items, embed_dim).
        """
        user_embeds = torch.sum(self.code_embeds(user_codes), dim=1)
        item_embeds = self.item_embeds.weight
        embeds = torch.cat([user_embeds, item_embeds], dim=0)

        weight = 1.0 / (self.num_layers + 1)
        result = embeds * weight

        for layer in self.layers:
            embeds = layer(embeds, edge_index)
            result += embeds * weight

        return result

    def recommendation_loss(
        self,
        user_codes: torch.Tensor,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Bayesian Personalized Ranking (BPR) loss.

        :param user_codes: Tensor of shape (num_users, num_codes_per_user) containing code indices for each user.
        :param pos_scores: Scores for positive interactions.
        :param neg_scores: Scores for negative interactions.
        :param nodes: Nodes involved in the interactions.
        :return: BPR loss value.
        """
        user_nodes = nodes[nodes < self.num_users]
        item_nodes = nodes[nodes >= self.num_users] - self.num_users
        code_nodes = user_codes[user_nodes].view(-1)

        code_embeds = self.code_embeds(code_nodes)
        item_embeds = self.item_embeds(item_nodes)
        reg_embeds = torch.cat([code_embeds, item_embeds], dim=0)

        return self.loss_fn(pos_scores, neg_scores, reg_embeds)

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
from torch_geometric.nn.models.lightgcn import BPRLoss


class GCNLayer(nn.Module):
    """
    A single GCN layer for propagating embeddings on the social and interaction graphs.
    """

    def __init__(self, embed_dim: int):
        """
        :param embed_dim: Embedding dimension for input and output embeddings.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.layer = GCNConv(in_channels=embed_dim, out_channels=embed_dim)

    def forward(
        self, *, embeds: torch.Tensor, num_users: int,
        social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor,
        social_edge_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Propagate node embeddings on social and behavioral graphs then fuse them.

        :param embeds: Embeddings for users and items of shape (num_users + num_items, embed_dim).
        :param num_users: Total number of users.
        :param social_edge_index: Edge index for the social graph of shape (2, num_social_edges).
        :param behavioral_edge_index: Edge index for the user-item behavioral graph of shape (2, num_behavioral_edges).
        :param social_edge_mask: Optional mask for social edges of shape (num_social_edges,), where True indicates a valid edge.
        """
        user_embeds = embeds[:num_users]
        social_edge_weight = social_edge_mask.float() if social_edge_mask is not None else None
        social_out = self.layer(user_embeds, social_edge_index, edge_weight=social_edge_weight)
        behavioral_out = self.layer(embeds, behavioral_edge_index)

        fused_user_embeds = (social_out + behavioral_out[:num_users]) * 0.5
        fused_item_embeds = behavioral_out[num_users:]

        return torch.cat([fused_user_embeds, fused_item_embeds], dim=0)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(embed_dim={self.embed_dim})'


class Denoiser(nn.Module):
    """
    Denoiser module for GDMSR, responsible for learning a mask over social edges to identify and mitigate noisy connections.
    """

    def __init__(
        self, num_users: int, num_items: int, embed_dim: int, num_layers: int,
        alpha: float, beta: float, gamma: float, epsilon: int, r: float, decay: float
    ):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Embedding dimension for users and items.
        :param num_layers: Number of GCN layers.
        :param alpha: Weight for social relation prediction loss.
        :param beta: Weight for previous confidence scores in EMA update.
        :param gamma: Exponent in adaptive denoising ratio calculation.
        :param epsilon: Threshold for user degree in adaptive denoising ratio calculation.
        :param r: Scale factor in adaptive denoising ratio calculation.
        :param decay: Decay weight in BPR loss.
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.r = r

        self.embeds = nn.Embedding(num_users + num_items, embed_dim)
        self.sep_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.eos_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.gcn = nn.ModuleList([GCNLayer(embed_dim) for _ in range(num_layers)])
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, dim_feedforward=embed_dim, nhead=1, batch_first=True
        )
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        self.bpr_loss = BPRLoss(lambda_reg=decay)
        self.link_pred_loss = nn.BCELoss()

    def get_embedding(
        self, *, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor, social_edge_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the propagated embeddings after passing through GCN layers.

        :param embeds: Initial embeddings for users and items of shape (num_users + num_items, embed_dim).
        :param num_users: Total number of users.
        :param social_edge_index: Edge index for the social graph of shape (2, num_social_edges).
        :param behavioral_edge_index: Edge index for the user-item behavioral graph of shape (2, num_behavioral_edges).
        :param social_edge_mask: Mask for social edges of shape (num_social_edges,), where True indicates a valid edge.
        :return: Propagated embeddings of shape (num_users + num_items, embed_dim).
        """
        embeds = self.embeds.weight

        for layer in self.gcn:
            embeds = layer(
                embeds=embeds,
                num_users=self.num_users,
                social_edge_index=social_edge_index,
                behavioral_edge_index=behavioral_edge_index,
                social_edge_mask=social_edge_mask
            )

        return embeds

    def _recommendation_forward(
        self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor,
        social_edge_mask: torch.Tensor, target_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the recommendation scores for the given target edges.

        :param embeds: Initial embeddings for users and items of shape (num_users + num_items, embed_dim).
        :param num_users: Total number of users.
        :param social_edge_index: Edge index for the social graph of shape (2, num_social_edges).
        :param behavioral_edge_index: Edge index for the user-item behavioral graph of shape (2, num_behavioral_edges).
        :param social_edge_mask: Mask for social edges of shape (num_social_edges,), where True indicates a valid edge.
        :param target_edge_index: Edge index for user-item pairs to compute recommendation scores, of shape (2, num_target_edges).
        :return: Recommendation scores for the target edges of shape (num_target_edges,).
        """
        embeds = self.get_embedding(
            social_edge_index=social_edge_index,
            behavioral_edge_index=behavioral_edge_index,
            social_edge_mask=social_edge_mask
        )

        src, dst = target_edge_index
        scores = (embeds[src] * embeds[dst]).sum(dim=1)

        return scores

    def _relation_confidence_forward(
        self, item_history: torch.Tensor, item_history_mask: torch.Tensor, users: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the relation confidence scores for user pairs based on their interaction histories.

        :param item_history: Tensor of shape (num_users, sequence_length) containing all item IDs in user interaction histories.
        :param item_history_mask: Tensor of shape (num_users, sequence_length) where True indicates padding positions in the item history.
        :param users: Tensor of shape (2, batch_size) containing user pairs (u, v) for relation modeling.
        :return: Tensor of shape (batch_size,) containing relation confidence scores.
        """
        device = users.device
        batch_size = users.size(1)

        u, v = users
        batch_sep = self.sep_token.expand(batch_size, -1, -1)
        batch_eos = self.eos_token.expand(batch_size, -1, -1)
        special_token_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)

        combined_sequence = torch.cat(
            [self.embeds(item_history[u]), batch_sep, self.embeds(item_history[v]), batch_eos],
            dim=1
        )
        combined_mask = torch.cat(
            [item_history_mask[u], special_token_mask, item_history_mask[v], special_token_mask],
            dim=1
        )

        encoded_sequence = self.transformer(
            src=combined_sequence,
            src_key_padding_mask=combined_mask
        )
        eos_summary = encoded_sequence[:, -1, :]
        confidence_scores: torch.Tensor = self.scorer(eos_summary)

        return confidence_scores.squeeze(-1)

    def forward(
        self, *,
        social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor, social_edge_mask: torch.Tensor,
        item_history: torch.Tensor, item_history_mask: torch.Tensor,
        users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor, pos_peers: torch.Tensor, neg_peers: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the recommendation scores and relation confidence scores, and combine them into a final loss.

        :param social_edge_index: Edge index for the social graph of shape (2, num_social_edges).
        :param behavioral_edge_index: Edge index for the user-item behavioral graph of shape (2, num_behavioral_edges).
        :param social_edge_mask: Mask for social edges of shape (num_social_edges,), where True indicates a valid edge.
        :param item_history: Tensor of shape (num_users, sequence_length) containing all item IDs in user interaction histories.
        :param item_history_mask: Tensor of shape (num_users, sequence_length) where True indicates padding positions in the item history.
        :param users: Tensor of shape (batch_size,) containing user IDs of the batch.
        :param pos_items: Tensor of shape (batch_size,) containing positive item IDs corresponding to the users.
        :param neg_items: Tensor of shape (batch_size,) containing negative item IDs corresponding to the users.
        :param pos_peers: Tensor of shape (batch_size,) containing positive peer user IDs corresponding to the users.
        :param neg_peers: Tensor of shape (batch_size,) containing negative peer user IDs corresponding to the users.
        :return: The final combined loss value for the batch.
        """
        pos_rec_scores = self._recommendation_forward(
            social_edge_index=social_edge_index,
            behavioral_edge_index=behavioral_edge_index,
            social_edge_mask=social_edge_mask,
            target_edge_index=torch.stack([users, pos_items], dim=0)
        )
        neg_rec_scores = self._recommendation_forward(
            social_edge_index=social_edge_index,
            behavioral_edge_index=behavioral_edge_index,
            social_edge_mask=social_edge_mask,
            target_edge_index=torch.stack([users, neg_items], dim=0)
        )
        pos_relation_scores = self._relation_confidence_forward(
            item_history=item_history,
            item_history_mask=item_history_mask,
            users=torch.stack([users, pos_peers], dim=0)
        )
        neg_relation_scores = self._relation_confidence_forward(
            item_history=item_history,
            item_history_mask=item_history_mask,
            users=torch.stack([users, neg_peers], dim=0)
        )

        relation_scores = torch.cat([pos_relation_scores, neg_relation_scores], dim=0)
        relation_labels = torch.cat([
            torch.ones_like(pos_relation_scores), torch.zeros_like(neg_relation_scores)], dim=0
        )

        node_ids = torch.cat([users, pos_items, neg_items])
        recommendation_loss = self.bpr_loss(pos_rec_scores, neg_rec_scores, self.embeds(node_ids))
        relation_loss = self.link_pred_loss(relation_scores, relation_labels.float())
        loss = self.alpha * relation_loss + (1.0 - self.alpha) * recommendation_loss

        return loss

    @torch.no_grad()
    def denoise_social_graph(
        self,
        item_history: torch.Tensor, item_history_mask: torch.Tensor,
        social_edge_index: torch.Tensor, confidence_scores: torch.Tensor,
        batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a mask for social edges based on the confidence scores. Note that the mask may not be undirected.

        :param item_history: Tensor of shape (num_users, sequence_length) containing all item IDs in user interaction histories.
        :param item_history_mask: Tensor of shape (num_users, sequence_length) where True indicates padding positions in the item history.
        :param social_edge_index: Undirected edge index for the social graph of shape (2, num_social_edges).
        :param confidence_scores: Tensor of shape (num_social_edges,) containing previous confidence scores, for EMA update.
        :param batch_size: Batch size for processing users in the social graph.
        :return: New confidence scores and mask for social edges, both of shape (num_social_edges,).
        """
        device = social_edge_index.device

        new_confidence_scores = torch.zeros_like(confidence_scores)
        for start in range(0, social_edge_index.size(1), batch_size):
            end = min(start + batch_size, social_edge_index.size(1))
            batch_users = social_edge_index[:, start:end]
            batch_confidence_scores = self._relation_confidence_forward(
                item_history=item_history,
                item_history_mask=item_history_mask,
                users=batch_users
            )
            new_confidence_scores[start:end] = batch_confidence_scores

        smoothed_scores = self.beta * confidence_scores + (1.0 - self.beta) * new_confidence_scores
        degrees = degree(social_edge_index[0], num_nodes=self.num_users)
        mask = torch.ones(social_edge_index.size(1), dtype=torch.bool, device=device)

        for u in range(self.num_users):
            if degrees[u] <= self.epsilon:
                continue

            drop_num = ((degrees[u].log2().floor() ** self.gamma) * self.r).clamp(min=0, max=degrees[u].item())
            edge_indices = (social_edge_index[0] == u).nonzero(as_tuple=True)[0]

            _, indices = torch.topk(smoothed_scores[edge_indices], k=int(drop_num), largest=False)
            mask[edge_indices[indices]] = False

        return smoothed_scores, mask

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'num_users={self.num_users}, '
            f'num_items={self.num_items}, '
            f'embed_dim={self.embed_dim}, '
            f'num_layers={self.num_layers}, '
            f'alpha={self.alpha}, '
            f'beta={self.beta}, '
            f'gamma={self.gamma}, '
            f'epsilon={self.epsilon}, '
            f'r={self.r}, '
            f'decay={self.decay})'
        )

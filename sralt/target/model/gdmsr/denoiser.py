import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.utils import degree


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
        self.social = nn.Linear(embed_dim, embed_dim)
        self.user_to_item = nn.Linear(embed_dim, embed_dim)
        self.item_to_user = nn.Linear(embed_dim, embed_dim)
        self.conv = LGConv(normalize=False)
        for layer in (self.social, self.user_to_item, self.item_to_user):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _propagate(
        self, embeds: torch.Tensor, edge_index: torch.Tensor,
        layer: nn.Linear, edge_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """DGL GraphConv's relation-specific out/in normalization, bias and ReLU."""
        src, dst = edge_index
        out_degree = degree(src, num_nodes=embeds.size(0)).clamp(min=1)
        in_degree = degree(dst, num_nodes=embeds.size(0)).clamp(min=1)
        weights = out_degree[src].rsqrt() * in_degree[dst].rsqrt()
        if edge_mask is not None:
            # Mask messages; degrees still refer to the original relation graph.
            weights = weights * edge_mask
        aggregated = self.conv(embeds, edge_index, weights)
        return layer(aggregated).relu()

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
        social_out = self._propagate(user_embeds, social_edge_index, self.social, social_edge_mask)
        user_edges = behavioral_edge_index[:, behavioral_edge_index[0] < num_users]
        item_edges = behavioral_edge_index[:, behavioral_edge_index[0] >= num_users]
        item_out = self._propagate(embeds, user_edges, self.user_to_item)
        user_out = self._propagate(embeds, item_edges, self.item_to_user)

        fused_user_embeds = (social_out + user_out[:num_users]) * 0.5
        fused_item_embeds = item_out[num_users:]

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
        nn.init.normal_(self.embeds.weight, std=0.1)
        nn.init.normal_(self.sep_token, std=0.1)
        nn.init.normal_(self.eos_token, std=0.1)

        self.gcn = nn.ModuleList([GCNLayer(embed_dim) for _ in range(num_layers)])
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=1, batch_first=True
        )
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        self.link_pred_loss = nn.BCELoss()

    def get_embedding(
        self, *, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor, social_edge_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the propagated embeddings after passing through GCN layers.

        :param social_edge_index: Edge index for the social graph of shape (2, num_social_edges).
        :param behavioral_edge_index: Edge index for the user-item behavioral graph of shape (2, num_behavioral_edges).
        :param social_edge_mask: Mask for social edges of shape (num_social_edges,), where True indicates a valid edge.
        :return: Embeddings of shape (num_users + num_items, (num_layers + 1) * embed_dim).
        """
        embeds = self.embeds.weight
        all_embeds = [embeds]

        for layer in self.gcn:
            embeds = layer(
                embeds=embeds,
                num_users=self.num_users,
                social_edge_index=social_edge_index,
                behavioral_edge_index=behavioral_edge_index,
                social_edge_mask=social_edge_mask
            )
            all_embeds.append(embeds)

        return torch.cat(all_embeds, dim=1)

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
        embeds = self.get_embedding(
            social_edge_index=social_edge_index,
            behavioral_edge_index=behavioral_edge_index,
            social_edge_mask=social_edge_mask
        )
        pos_rec_scores = (embeds[users] * embeds[pos_items]).sum(dim=1)
        neg_rec_scores = (embeds[users] * embeds[neg_items]).sum(dim=1)
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

        recommendation_loss = F.softplus(neg_rec_scores - pos_rec_scores).mean()
        regularization = (embeds[users].square().sum() + embeds[pos_items].square().sum()) / users.size(0)
        relation_loss = self.link_pred_loss(relation_scores, relation_labels.float())
        loss = self.alpha * relation_loss + (1.0 - self.alpha) * recommendation_loss + self.decay * regularization

        return loss

    @torch.no_grad()
    def denoise_social_graph(
        self,
        item_history: torch.Tensor, item_history_mask: torch.Tensor,
        social_edge_index: torch.Tensor, confidence_scores: torch.Tensor,
        batch_size: int, social_edge_mask: torch.Tensor | None = None,
        initialize: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a mask for social edges based on the confidence scores. Note that the mask may not be undirected.

        :param item_history: Tensor of shape (num_users, sequence_length) containing all item IDs in user interaction histories.
        :param item_history_mask: Tensor of shape (num_users, sequence_length) where True indicates padding positions in the item history.
        :param social_edge_index: Undirected edge index for the social graph of shape (2, num_social_edges).
        :param confidence_scores: Tensor of shape (num_social_edges,) containing previous confidence scores, for EMA update.
        :param batch_size: Batch size for processing users in the social graph.
        :param social_edge_mask: Previous mask; removed edges remain removed.
        :param initialize: Initialize confidence without smoothing or removing edges.
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

        smoothed_scores = new_confidence_scores if initialize else (
            self.beta * confidence_scores + (1.0 - self.beta) * new_confidence_scores
        )
        degrees = degree(social_edge_index[0], num_nodes=self.num_users)
        mask = (torch.ones(social_edge_index.size(1), dtype=torch.bool, device=device)
                if social_edge_mask is None else social_edge_mask.clone())
        if initialize:
            return smoothed_scores, mask

        drop_counts = (degrees.clamp(min=1).log2().floor().pow(self.gamma) * self.r).long()
        drop_counts = torch.minimum(drop_counts.clamp(min=0), degrees.long())
        drop_counts[degrees < self.epsilon] = 0
        # Group by source while keeping ascending confidence within each group.
        order = smoothed_scores.argsort(stable=True)
        order = order[social_edge_index[0, order].argsort(stable=True)]
        sources = social_edge_index[0, order]
        starts = degrees.long().cumsum(0) - degrees.long()
        ranks = torch.arange(order.numel(), device=device) - starts[sources]
        mask[order[ranks < drop_counts[sources]]] = False

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

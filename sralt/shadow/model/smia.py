import types
import typing
import logging
from typing import Any, Callable

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import LightGCN
from torch_geometric.utils import to_undirected
from torch_geometric.metrics.link_pred import LinkPredMetric

from . import register_shadow_model
from .base import ShadowModel
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.utils import timer, build_normalized_adj, calculate_model_fingerprint, compute_recommendation_metrics
from sralt.utils import DataLoader, EarlyStopping


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

        self.social_branch = LightGCN(self.num_nodes, embed_dim, num_layers)
        self.behavioral_branch = LightGCN(self.num_nodes, embed_dim, num_layers)

        self._cached_social_edge_index = torch.Tensor()
        self._cached_social_sparse_adj = torch.Tensor()
        self._cached_behavioral_edge_index = torch.Tensor()
        self._cached_behavioral_sparse_adj = torch.Tensor()

    def build_adj(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute and cache the sparse adjacency matrices for both social and behavioral graphs.
        When deterministic algorithms are enabled, returns the original edge indices to avoid non-determinism in sparse matrix operations.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: A tuple of social and behavioral sparse adjacency matrices.
        """
        if torch.are_deterministic_algorithms_enabled():
            # Sparse matrix multiplication is non-deterministic on GPU.
            # LightGCN will use deterministic scatter when given edge index directly.
            return social_edge_index, behavioral_edge_index

        if self._cached_social_edge_index is not social_edge_index:
            self._cached_social_edge_index = social_edge_index
            self._cached_social_sparse_adj = build_normalized_adj(
                edge_index=social_edge_index,
                num_nodes=self.num_nodes,
                edge_weight=None,
                add_self_loops=False,
                normalization='none'
            )

        if self._cached_behavioral_edge_index is not behavioral_edge_index:
            self._cached_behavioral_edge_index = behavioral_edge_index
            self._cached_behavioral_sparse_adj = build_normalized_adj(
                edge_index=behavioral_edge_index,
                num_nodes=self.num_nodes,
                edge_weight=None,
                add_self_loops=False,
                normalization='none'
            )

        return self._cached_social_sparse_adj, self._cached_behavioral_sparse_adj

    def forward(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the target model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape (2, num_target_edges), where the rows correspond to social and behavioral scores respectively.
        """
        social_adj, behavioral_adj = self.build_adj(social_edge_index, behavioral_edge_index)

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
        social_adj, behavioral_adj = self.build_adj(social_edge_index, behavioral_edge_index)

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
        social_pos_scores, behavioral_pos_scores = pos_scores
        social_neg_scores, behavioral_neg_scores = neg_scores

        social_loss = self.social_branch.recommendation_loss(
            social_pos_scores, social_neg_scores, nodes, lambda_reg=self.decay
        )
        behavioral_loss = self.behavioral_branch.recommendation_loss(
            behavioral_pos_scores, behavioral_neg_scores, nodes, lambda_reg=self.decay
        )

        return social_loss + behavioral_loss

    @timer('Training model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_model(
        self, *, data: ShadowData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int,
        num_epochs: int, optimizer_cls: type[optim.Optimizer], optimizer_kwargs: dict[str, Any],
        metrics: list[LinkPredMetric], metric_aggregator: types.CodeType, early_stopping_kwargs: dict[str, Any],
        **kwargs
    ):
        """
        Train the SMIA shadow model.

        :param data: The target dataset.
        :param writer: Optional TensorBoard writer for logging.
        :param train_batch_size: Batch size for training.
        :param val_batch_size: Batch size for validation.
        :param val_num_negatives: Number of negative samples per positive sample during validation.
        :param num_epochs: Maximum number of training epochs.
        :param optimizer_cls: Optimizer class to use.
        :param optimizer_kwargs: Keyword arguments for the optimizer.
        :param metrics: List of link prediction metrics to evaluate.
        :param metric_aggregator: Source code of a Callable[[float, list[float]], float] that aggregates loss and metrics.
        :param early_stopping_kwargs: Keyword arguments for the EarlyStopping helper.
        """
        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        train_dataset = TrainShadowDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalShadowDataset(data, split='val', num_negatives=val_num_negatives)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        optimizer = optimizer_cls(self.parameters(), **optimizer_kwargs)
        early_stopping = EarlyStopping(self, **early_stopping_kwargs)
        metric_aggregator_fn: Callable[[float, list[float]], float] = eval(metric_aggregator)

        model_name = self.__class__.__name__
        metrics_names = [
            f'{typing.cast(str, metric.__class__.__name__).removeprefix("LinkPred")}@{metric.k}'
            for metric in metrics
        ]

        for epoch in range(1, num_epochs + 1):
            self.train()

            total_loss = 0.0
            total_samples = 0

            for users, pos_items, neg_items in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
                users: torch.Tensor
                optimizer.zero_grad()

                pos_edges = torch.stack([users, pos_items], dim=0)
                neg_edges = torch.stack([users, neg_items], dim=0)
                node_ids = torch.cat([users, pos_items, neg_items])

                pos_scores: torch.Tensor = self(data.social_edge_index, undirected_behavioral_edge_index, pos_edges)
                neg_scores: torch.Tensor = self(data.social_edge_index, undirected_behavioral_edge_index, neg_edges)

                loss = self.calculate_loss(pos_scores, neg_scores, node_ids)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * users.size(0)
                total_samples += users.size(0)

            average_loss = total_loss / total_samples if total_samples > 0 else float('nan')
            if writer is not None:
                writer.add_scalar(f'{model_name}/Loss/train', average_loss, epoch)

            self.eval()
            metrics_results = compute_recommendation_metrics(
                embeds=self.get_embedding(data.social_edge_index, undirected_behavioral_edge_index),
                metrics=metrics,
                eval_loader=val_loader
            )
            metrics_str = ', '.join([f'{name}={value:.4f}' for name, value in zip(metrics_names, metrics_results)])
            logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metrics_str}')

            if writer is not None:
                for name, value in zip(metrics_names, metrics_results):
                    writer.add_scalar(f'{model_name}/{name}/val', value, epoch)

            aggregated_metric = metric_aggregator_fn(average_loss, metrics_results)
            if early_stopping.step(aggregated_metric):
                break

        early_stopping.load_best_state()

    @torch.no_grad()
    def test_model(
        self, *, data: ShadowData, metrics: list[LinkPredMetric],
        test_batch_size: int, test_num_negatives: int, **kwargs
    ) -> list[float]:
        """
        Test the SMIA shadow model.

        :param data: The shadow dataset.
        :param metrics: A list of link prediction metrics to compute.
        :param writer: Optional TensorBoard writer for logging.
        :param test_batch_size: Batch size for testing.
        :param test_num_negatives: Number of negative samples per positive sample during testing.
        :return: A list of metric values.
        """
        self.eval()

        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        test_dataset = EvalShadowDataset(data, split='test', num_negatives=test_num_negatives)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        metrics_results = compute_recommendation_metrics(
            embeds=self.get_embedding(data.social_edge_index, undirected_behavioral_edge_index),
            metrics=metrics,
            eval_loader=test_loader
        )

        return metrics_results

    @torch.no_grad()
    def get_user_features(self, data: ShadowData) -> torch.Tensor:
        """
        Get the user features learned by the shadow model for attack purposes.

        :param data: The shadow dataset.
        :return: A tensor of shape (num_users, feature_dim) representing user features.
        """
        self.eval()

        social_edge_index = data.social_edge_index
        behavioral_edge_index = to_undirected(data.behavioral_edge_index)
        social_adj, behavioral_adj = self.build_adj(social_edge_index, behavioral_edge_index)

        social_embeds = self.social_branch.get_embedding(social_adj)
        behavioral_embeds = self.behavioral_branch.get_embedding(behavioral_adj)
        social_user_embeds = social_embeds[:self.num_users]
        behavioral_user_embeds = behavioral_embeds[:self.num_users]

        user_item_features = []
        user_item_edge_index = data.behavioral_edge_index
        fused_embeds = (social_embeds + behavioral_embeds) / 2

        for u in range(self.num_users):
            item_indices = user_item_edge_index[1, user_item_edge_index[0] == u]
            items_mean = fused_embeds[item_indices].mean(dim=0)
            items_var = fused_embeds[item_indices].var(dim=0)
            user_item_features.append(torch.cat([items_mean, items_var]))

        user_item_features = torch.stack(user_item_features)
        user_features = torch.cat([social_user_embeds, behavioral_user_embeds, user_item_features], dim=1)
        return user_features

    def __repr__(self) -> str:
        """
        String representation and fingerprint of the model.
        """
        return (
            f'{self.__class__.__name__}('
            f'num_users={self.num_users}, '
            f'num_items={self.num_items}, '
            f'embed_dim={self.embed_dim}, '
            f'num_layers={self.num_layers}, '
            f'decay={self.decay}, '
            f'fingerprint={calculate_model_fingerprint(self)})'
        )

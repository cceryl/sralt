import types
import typing
import logging
from typing import Any, Callable, Literal

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn.models.lightgcn import LightGCN, BPRLoss
from torch_geometric.utils import to_undirected
from torch_geometric.metrics.link_pred import LinkPredMetric

from sralt.shadow.model import register_shadow_model, ShadowModel
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.utils import timer, build_normalized_adj, calculate_model_fingerprint, compute_recommendation_metrics
from sralt.utils import DataLoader, EarlyStopping


@register_shadow_model('CollaborativeFilter')
class CollaborativeFilter(ShadowModel):
    """
    Shadow model that ignores the social graph and use CF in the social branch.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        num_layers: int,
        decay: float,
        ablation_feature: Literal['base', 'without-social', 'without-behavioral'] = 'base'
    ):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embedding vectors.
        :param num_layers: Number of LightGCN layers.
        :param decay: Weight decay factor for regularization.
        :param ablation_feature: Use only one branch for user features if 'without-social' or 'without-behavioral'.
        """
        super().__init__(num_users, num_items)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.decay = decay
        self.ablation_feature = ablation_feature

        self.social_embeds = torch.nn.Embedding(num_users + num_items, embed_dim)
        self.behavioral_branch = LightGCN(num_users + num_items, embed_dim, num_layers)

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
                num_nodes=self.num_users + self.num_items,
                edge_weight=None,
                add_self_loops=False,
                normalization='none'
            )

        if self._cached_behavioral_edge_index is not behavioral_edge_index:
            self._cached_behavioral_edge_index = behavioral_edge_index
            self._cached_behavioral_sparse_adj = build_normalized_adj(
                edge_index=behavioral_edge_index,
                num_nodes=self.num_users + self.num_items,
                edge_weight=None,
                add_self_loops=False,
                normalization='none'
            )

        return self._cached_social_sparse_adj, self._cached_behavioral_sparse_adj

    def forward(
        self,
        stage: Literal['social', 'behavioral'],
        social_edge_index: torch.Tensor,
        behavioral_edge_index: torch.Tensor,
        target_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass depending on the current training stage.

        :param stage: Current training stage, either 'social' or 'behavioral'.
        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape (num_target_edges,) for the target interactions.
        """
        _, behavioral_adj = self.build_adj(social_edge_index, behavioral_edge_index)

        match stage:
            case 'social':
                users, items = target_edge_index
                user_embeds: torch.Tensor = self.social_embeds(users)
                item_embeds: torch.Tensor = self.social_embeds(items)
                return (user_embeds * item_embeds).sum(dim=1)
            case 'behavioral': return self.behavioral_branch(behavioral_adj, target_edge_index)
            case _: raise ValueError(f'Invalid stage "{stage}". Must be "behavioral" or "social".')

    def get_embedding(
        self,
        stage: Literal['social', 'behavioral'],
        social_edge_index: torch.Tensor,
        behavioral_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the user and item embeddings of the current stage.

        :param stage: Current training stage, either 'social' or 'behavioral'.
        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Embedding of shape (num_users + num_items [+ num_codes], embed_dim), with num_codes included in social stage.
        """
        _, behavioral_adj = self.build_adj(social_edge_index, behavioral_edge_index)

        match stage:
            case 'social': return self.social_embeds.weight
            case 'behavioral': return self.behavioral_branch.get_embedding(behavioral_adj)
            case _: raise ValueError(f'Invalid stage "{stage}". Must be "social" or "behavioral".')

    def recommendation_loss(
        self,
        stage: Literal['social', 'behavioral'],
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the recommendation loss for the SimilarityAugmentation model.

        :param stage: Current training stage, either 'social' or 'behavioral'.
        :param pos_scores: Scores for positive interactions.
        :param neg_scores: Scores for negative interactions.
        :param nodes: Nodes involved in the interactions.
        :return: Recommendation loss value.
        """
        match stage:
            case 'social':
                loss_fn = BPRLoss(lambda_reg=self.decay)
                nodes_embeds = self.social_embeds(nodes)
                return loss_fn(pos_scores, neg_scores, nodes_embeds)
            case 'behavioral': return self.behavioral_branch.recommendation_loss(pos_scores, neg_scores, nodes, self.decay)
            case _: raise ValueError(f'Invalid stage "{stage}". Must be "social" or "behavioral".')

    @timer('Training stage "{stage}" of model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_stage(
        self,
        *,
        stage: Literal['social', 'behavioral'],
        social_edge_index: torch.Tensor,
        behavioral_edge_index: torch.Tensor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        optimizer: optim.Optimizer,
        metrics: list[LinkPredMetric],
        metric_aggregator: Callable[[float, list[float]], float],
        early_stopping: EarlyStopping,
        writer: SummaryWriter | None,
    ):
        """
        Train a specific stage of the SimilarityAugmentation shadow model.

        :param stage: Current training stage, either 'social' or 'behavioral'.
        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        :param num_epochs: Maximum number of training epochs.
        :param optimizer: Optimizer for training.
        :param metrics: List of link prediction metrics to evaluate.
        :param metric_aggregator: Callable that aggregates loss and metrics for early stopping.
        :param early_stopping: EarlyStopping helper instance.
        :param writer: Optional TensorBoard writer for logging.
        """
        branch_name = f'{self.__class__.__name__}/{stage.capitalize()}Stage'
        metrics_names = [
            f'{typing.cast(str, metric.__class__.__name__).removeprefix("LinkPred")}@{metric.k}'
            for metric in metrics
        ]
        logging.info(f'Starting {stage} branch training...')

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

                pos_scores = self(stage, social_edge_index, behavioral_edge_index, pos_edges)
                neg_scores = self(stage, social_edge_index, behavioral_edge_index, neg_edges)

                loss: torch.Tensor = self.recommendation_loss(stage, pos_scores, neg_scores, node_ids)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * users.size(0)
                total_samples += users.size(0)

            average_loss = total_loss / total_samples if total_samples > 0 else float('nan')
            if writer is not None:
                writer.add_scalar(f'{branch_name}/Loss/train', average_loss, epoch)

            self.eval()
            metrics_results = compute_recommendation_metrics(
                embeds=self.get_embedding(stage, social_edge_index, behavioral_edge_index),
                metrics=metrics,
                eval_loader=val_loader
            )
            metrics_str = ', '.join([f'{name}={value:.4f}' for name, value in zip(metrics_names, metrics_results)])
            logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metrics_str}')

            if writer is not None:
                for name, value in zip(metrics_names, metrics_results):
                    writer.add_scalar(f'{branch_name}/{name}/val', value, epoch)

            aggregated_metric = metric_aggregator(average_loss, metrics_results)
            if early_stopping.step(aggregated_metric):
                break

        early_stopping.load_best_state()

    @timer('Training model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_model(
        self, *, data: ShadowData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int,
        num_epochs: int, optimizer_cls: type[optim.Optimizer], optimizer_kwargs: dict[str, Any],
        metrics: list[LinkPredMetric], metric_aggregator: types.CodeType, early_stopping_kwargs: dict[str, Any],
        **kwargs
    ):
        """
        Train the SimilarityAugmentation shadow model.

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
        metric_aggregator_fn: Callable[[float, list[float]], float] = eval(metric_aggregator)

        train_dataset = TrainShadowDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalShadowDataset(data, split='val', num_negatives=val_num_negatives)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        train_stage_kwargs = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'num_epochs': num_epochs,
            'metrics': metrics,
            'metric_aggregator': metric_aggregator_fn,
            'writer': writer
        }

        self.train_stage(
            stage='behavioral',
            social_edge_index=data.social_edge_index,
            behavioral_edge_index=undirected_behavioral_edge_index,
            optimizer=optimizer_cls(self.behavioral_branch.parameters(), **optimizer_kwargs),
            early_stopping=EarlyStopping(self.behavioral_branch, **early_stopping_kwargs),
            **train_stage_kwargs
        )

        if self.ablation_feature == 'without-social':
            return

        self.train_stage(
            stage='social',
            social_edge_index=data.social_edge_index,
            behavioral_edge_index=undirected_behavioral_edge_index,
            optimizer=optimizer_cls(self.social_embeds.parameters(), **optimizer_kwargs),
            early_stopping=EarlyStopping(self.social_embeds, **early_stopping_kwargs),
            **train_stage_kwargs
        )

    @torch.no_grad()
    def test_model(
        self, *, data: ShadowData, metrics: list[LinkPredMetric],
        test_batch_size: int, test_num_negatives: int, stage: Literal['social', 'behavioral'],
        **kwargs
    ) -> list[float]:
        """
        Test the SimilarityAugmentation shadow model.

        :param data: The shadow dataset.
        :param metrics: A list of link prediction metrics to compute.
        :param writer: Optional TensorBoard writer for logging.
        :param test_batch_size: Batch size for testing.
        :param test_num_negatives: Number of negative samples per positive sample during testing.
        :param stage: The stage to test, either 'social' or 'behavioral'.
        :return: A list of metric values.
        """
        self.eval()

        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        test_dataset = EvalShadowDataset(data, split='test', num_negatives=test_num_negatives)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        metrics_results = compute_recommendation_metrics(
            embeds=self.get_embedding(stage, data.social_edge_index, undirected_behavioral_edge_index),
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

        behavioral_edge_index = to_undirected(data.behavioral_edge_index)
        _, behavioral_adj = self.build_adj(data.social_edge_index, behavioral_edge_index)

        social_embeds = self.social_embeds.weight
        behavioral_embeds = self.behavioral_branch.get_embedding(behavioral_adj)
        social_user_embeds = social_embeds[:self.num_users]
        behavioral_user_embeds = behavioral_embeds[:self.num_users]

        user_item_features = []
        user_item_edge_index = data.behavioral_edge_index
        match self.ablation_feature:
            case 'base':
                fused_embeds = (social_embeds + behavioral_embeds) / 2
            case 'without-social':
                fused_embeds = behavioral_embeds
            case 'without-behavioral':
                fused_embeds = social_embeds
            case _:
                raise ValueError(f'Invalid ablation_feature "{self.ablation_feature}".')

        for u in range(self.num_users):
            item_indices = user_item_edge_index[1, user_item_edge_index[0] == u]
            items_mean = fused_embeds[item_indices].mean(dim=0)
            items_var = fused_embeds[item_indices].var(dim=0)
            user_item_features.append(torch.cat([items_mean, items_var]))

        user_item_features = torch.stack(user_item_features)

        match self.ablation_feature:
            case 'base':
                return torch.cat([social_user_embeds, behavioral_user_embeds, user_item_features], dim=1)
            case 'without-social':
                return torch.cat([behavioral_user_embeds, user_item_features], dim=1)
            case 'without-behavioral':
                return torch.cat([social_user_embeds, user_item_features], dim=1)
            case _:
                raise ValueError(f'Invalid ablation_feature "{self.ablation_feature}".')

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'num_users={self.num_users}, '
            f'num_items={self.num_items}, '
            f'embed_dim={self.embed_dim}, '
            f'num_layers={self.num_layers}, '
            f'decay={self.decay}, '
            f'ablation_feature={self.ablation_feature}, '
            f'fingerprint={calculate_model_fingerprint(self)})'
        )

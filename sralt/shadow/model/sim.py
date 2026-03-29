import types
import typing
import logging
from typing import Any, Callable, Literal

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import LightGCN
from torch_geometric.utils import coalesce, to_undirected
from torch_geometric.metrics.link_pred import LinkPredMetric

from . import register_shadow_model
from .base import ShadowModel
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.utils import timer, build_normalized_adj, model_fingerprint, compute_recommendation_metrics
from sralt.utils import DataLoader, EarlyStopping


@register_shadow_model('SimilarityAugmentation')
class SimilarityAugmentationShadow(ShadowModel):
    """
    Shadow model implementing the Similarity Augmentation dual-branch shadow model.
    This model augments the social graph by connecting top-k similar users based on their behavioral interactions.    
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        num_layers: int,
        decay: float,
        top_k: int,
        sim: Literal['cosine', 'dot', 'random'],
        undirected_augment: bool = False,
        ablation_feature: Literal['base', 'without-social', 'without-behavioral'] = 'base'
    ):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embedding vectors.
        :param num_layers: Number of LightGCN layers.
        :param decay: Weight decay factor for regularization.
        :param top_k: Number of top similar users to connect in the social graph.
        :param sim: Similarity criterion for finding similar users.
        :param undirected_augment: Whether to add augmented edges in both directions (undirected) or only from source to target (directed).
        :param ablation_feature: Use only one branch for user features if 'without-social' or 'without-behavioral'.
        """
        super().__init__(num_users, num_items)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.decay = decay
        self.top_k = top_k
        self.sim = sim
        self.undirected_augment = undirected_augment
        self.ablation_feature = ablation_feature

        self.social_branch = LightGCN(num_users + num_items, embed_dim, num_layers)
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
        social_adj, behavioral_adj = self.build_adj(social_edge_index, behavioral_edge_index)

        match stage:
            case 'social': return self.social_branch(social_adj, target_edge_index)
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
        social_adj, behavioral_adj = self.build_adj(social_edge_index, behavioral_edge_index)

        match stage:
            case 'social': return self.social_branch.get_embedding(social_adj)
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
            case 'social': return self.social_branch.recommendation_loss(pos_scores, neg_scores, nodes, self.decay)
            case 'behavioral': return self.behavioral_branch.recommendation_loss(pos_scores, neg_scores, nodes, self.decay)
            case _: raise ValueError(f'Invalid stage "{stage}". Must be "social" or "behavioral".')

    def augment_social_graph(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Augments the social graph based on user similarity derived from behavioral interactions.
        Connect each user to their top-k most similar users according to the specified similarity criterion.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Augmented edge index tensor.
        """
        if self.top_k == 0:
            return social_edge_index

        device = social_edge_index.device
        behavioral_embeds = self.behavioral_branch.get_embedding(behavioral_edge_index)[:self.num_users]

        match self.sim:
            case 'cosine':
                normalized_embeds = F.normalize(behavioral_embeds, p=2, dim=1)
                similarity = normalized_embeds @ normalized_embeds.t()
            case 'dot':
                similarity = behavioral_embeds @ behavioral_embeds.t()
            case 'random':
                similarity = torch.rand((self.num_users, self.num_users), device=device)
            case _:
                raise ValueError(f'Invalid similarity criterion "{self.sim}".')

        similarity.fill_diagonal_(float('-inf'))
        top_k_similar: torch.Tensor = torch.topk(similarity, self.top_k, dim=1).indices

        source = torch.arange(self.num_users, device=device).unsqueeze(1).expand(-1, self.top_k).flatten()
        target = top_k_similar.flatten()
        augmented_edges = torch.stack([source, target], dim=0)
        augmented_edge_index = torch.cat([social_edge_index, augmented_edges], dim=1)

        if self.undirected_augment:
            return to_undirected(augmented_edge_index)
        else:
            return coalesce(augmented_edge_index)

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
        metric_names = [
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
            metric_results = compute_recommendation_metrics(
                embeds=self.get_embedding(stage, social_edge_index, behavioral_edge_index),
                metrics=metrics,
                eval_loader=val_loader
            )
            metric_str = ', '.join([f'{name}={value:.4f}' for name, value in zip(metric_names, metric_results)])
            logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metric_str}')

            if writer is not None:
                for name, value in zip(metric_names, metric_results):
                    writer.add_scalar(f'{branch_name}/{name}/val', value, epoch)

            aggregated_metric = metric_aggregator(average_loss, metric_results)
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

        augmented_social_edge_index = self.augment_social_graph(data.social_edge_index, data.behavioral_edge_index)

        self.train_stage(
            stage='social',
            social_edge_index=augmented_social_edge_index,
            behavioral_edge_index=undirected_behavioral_edge_index,
            optimizer=optimizer_cls(self.social_branch.parameters(), **optimizer_kwargs),
            early_stopping=EarlyStopping(self.social_branch, **early_stopping_kwargs),
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

        augmented_social_edge_index = self.augment_social_graph(data.social_edge_index, data.behavioral_edge_index)
        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        test_dataset = EvalShadowDataset(data, split='test', num_negatives=test_num_negatives)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        metric_results = compute_recommendation_metrics(
            embeds=self.get_embedding(stage, augmented_social_edge_index, undirected_behavioral_edge_index),
            metrics=metrics,
            eval_loader=test_loader
        )

        return metric_results

    @torch.no_grad()
    def get_user_features(self, data: ShadowData) -> torch.Tensor:
        """
        Get the user features learned by the shadow model for attack purposes.

        :param data: The shadow dataset.
        :return: A tensor of shape (num_users, feature_dim) representing user features.
        """
        self.eval()

        augmented_social_edge_index = self.augment_social_graph(data.social_edge_index, data.behavioral_edge_index)
        behavioral_edge_index = to_undirected(data.behavioral_edge_index)
        social_adj, behavioral_adj = self.build_adj(augmented_social_edge_index, behavioral_edge_index)

        social_embeds = self.social_branch.get_embedding(social_adj)[:self.num_users + self.num_items]
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
            f'top_k={self.top_k}, '
            f'sim="{self.sim}", '
            f'undirected_augment={self.undirected_augment}, '
            f'ablation_feature={self.ablation_feature}, '
            f'fingerprint={model_fingerprint(self)})'
        )

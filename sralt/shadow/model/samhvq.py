import types
import typing
import logging
from typing import Any, Callable, Literal

import torch
import torch.optim as optim
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import LightGCN
from torch_geometric.utils import to_undirected, coalesce
from torch_geometric.metrics.link_pred import LinkPredMetric

from sralt.shadow.model import register_shadow_model, ShadowModel
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.utils import timer, build_normalized_adj, calculate_model_fingerprint, compute_recommendation_metrics
from sralt.utils import DataLoader, EarlyStopping


@register_shadow_model('SemanticAugmentationMHVQ')
class SemanticAugmentationMHVQ(ShadowModel):
    """
    Shadow model implementing the Semantic Augmentation (SA) attack.
    This is the variant using Multi-Head Vector Quantization (MHVQ) for semantic code generation.

    Stage Behavioral: Train behavioral branch with MHVQ on behavioral graph to obtain semantic user codes.
    Stage Social: Train social branch on social graph augmented with semantic user codes.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        num_layers: int,
        decay: float,
        mhvq_num_quantizers: int,
        mhvq_codebook_size: int,
        commitment_weight: float,
        without_quantization: bool = False,
        ablation_code: Literal['base', 'one-to-one', 'random'] = 'base',
        ablation_feature: Literal['base', 'without-social', 'without-behavioral'] = 'base'
    ):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embedding vectors.
        :param num_layers: Number of LightGCN layers.
        :param decay: Weight decay factor for regularization.
        :param mhvq_num_quantizers: Number of quantizers in MHVQ.
        :param mhvq_codebook_size: Size of the codebook in MHVQ.
        :param commitment_weight: Weight for the MHVQ commitment loss.
        :param without_quantization: If True, use the behavioral embeddings directly without quantization in behavioral stage.
        :param ablation_code: Ablation for user codes, with 'one-to-one' to use direct mapping and 'random' for random codes.
        :param ablation_feature: Use only one branch for user features if 'without-social' or 'without-behavioral'.
        """
        super().__init__(num_users, num_items)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.decay = decay
        self.mhvq_num_quantizers = mhvq_num_quantizers
        self.mhvq_codebook_size = mhvq_codebook_size
        self.commitment_weight = commitment_weight
        self.without_quantization = without_quantization
        self.ablation_code = ablation_code
        self.ablation_feature = ablation_feature

        # Users and codes are one-to-one if codes are removed in ablation
        self.num_codes = mhvq_num_quantizers * mhvq_codebook_size if ablation_code != 'one-to-one' else num_users
        self.social_branch = LightGCN(num_users + num_items + self.num_codes, embed_dim, num_layers)
        self.behavioral_branch = LightGCN(num_users + num_items, embed_dim, num_layers)

        self.mhvq = VectorQuantize(
            dim=embed_dim,
            codebook_dim=embed_dim // 4,  # maybe let this be a param?
            heads=mhvq_num_quantizers,
            separate_codebook_per_head=True,
            codebook_size=mhvq_codebook_size
        )

        self.user_codes: torch.Tensor
        self.register_buffer('user_codes', torch.zeros((num_users, mhvq_num_quantizers), dtype=torch.long))

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass depending on the current training stage.

        :param stage: Current training stage, either 'social' or 'behavioral'.
        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape (num_target_edges,) for the target interactions and commitment loss.
        """
        social_adj, behavioral_adj = self.build_adj(social_edge_index, behavioral_edge_index)

        match stage:
            case 'social':
                return self.social_branch(social_adj, target_edge_index), torch.tensor(0.0)

            case 'behavioral':
                embeds = self.behavioral_branch.get_embedding(behavioral_adj)
                quantized, self.user_codes, commitment_loss = self.mhvq(embeds[:self.num_users])
                quantized = torch.cat([quantized, embeds[self.num_users:]], dim=0)
                commitment_loss: torch.Tensor

                offset = torch.arange(self.mhvq_num_quantizers, device=self.user_codes.device) * self.mhvq_codebook_size
                self.user_codes += offset.unsqueeze(0)

                embeds = quantized if not self.without_quantization else embeds
                src = embeds[target_edge_index[0]]
                dst = embeds[target_edge_index[1]]

                return (src * dst).sum(dim=-1), commitment_loss.sum()

            case _:
                raise ValueError(f'Invalid stage "{stage}". Must be "behavioral" or "social".')

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
        Compute the recommendation loss for the SA model.

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

    def augment_social_graph(self, social_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Augments the social graph with bipartite edges between users and their semantic codes.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :return: Augmented undirected edge index tensor including user-code edges.
        """
        device = social_edge_index.device

        num_codes_per_user = self.mhvq_num_quantizers if self.ablation_code != 'one-to-one' else 1
        user_indices = torch.arange(self.num_users, device=device).repeat_interleave(num_codes_per_user)
        code_indices = (self.user_codes + self.num_users + self.num_items).flatten()

        user_code_edge_index = torch.stack([user_indices, code_indices], dim=0)
        augmented_edge_index = torch.cat([social_edge_index, user_code_edge_index], dim=1)
        return coalesce(to_undirected(augmented_edge_index))

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
        Train a specific stage of the SA shadow model.

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

                pos_scores, commitment_loss_pos = self(stage, social_edge_index, behavioral_edge_index, pos_edges)
                neg_scores, commitment_loss_neg = self(stage, social_edge_index, behavioral_edge_index, neg_edges)
                recommendation_loss = self.recommendation_loss(stage, pos_scores, neg_scores, node_ids)

                loss: torch.Tensor = recommendation_loss + commitment_loss_pos + commitment_loss_neg
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
        Train the SA shadow model.

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

        match self.ablation_code:
            case 'one-to-one':
                self.user_codes = torch.arange(self.num_users, device=self.user_codes.device).unsqueeze(-1)
            case 'random':
                self.user_codes = torch.randint(
                    low=0,
                    high=self.mhvq_num_quantizers * self.mhvq_codebook_size,
                    size=(self.num_users, self.mhvq_num_quantizers),
                    device=self.user_codes.device
                )

        augmented_social_edge_index = self.augment_social_graph(data.social_edge_index)

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
        Test the SA shadow model.

        :param data: The shadow dataset.
        :param metrics: A list of link prediction metrics to compute.
        :param writer: Optional TensorBoard writer for logging.
        :param test_batch_size: Batch size for testing.
        :param test_num_negatives: Number of negative samples per positive sample during testing.
        :param stage: The stage to test, either 'social' or 'behavioral'.
        :return: A list of metric values.
        """
        self.eval()

        augmented_social_edge_index = self.augment_social_graph(data.social_edge_index)
        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        test_dataset = EvalShadowDataset(data, split='test', num_negatives=test_num_negatives)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        metrics_results = compute_recommendation_metrics(
            embeds=self.get_embedding(stage, augmented_social_edge_index, undirected_behavioral_edge_index),
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

        augmented_social_edge_index = self.augment_social_graph(data.social_edge_index)
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
            f'mhvq_num_quantizers={self.mhvq_num_quantizers}, '
            f'mhvq_codebook_size={self.mhvq_codebook_size}, '
            f'commitment_weight={self.commitment_weight}, '
            f'without_quantization={self.without_quantization}, '
            f'ablation_code={self.ablation_code}, '
            f'ablation_feature={self.ablation_feature}, '
            f'fingerprint={calculate_model_fingerprint(self)})'
        )

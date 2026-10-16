import typing
import logging
from typing import Literal

import torch
from tqdm import tqdm
from omegaconf import DictConfig
from torch.optim import Optimizer
from hydra.utils import instantiate
from torch_geometric.nn import LightGCN
from vector_quantize_pytorch import ResidualVQ
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import coalesce, to_undirected
from torch_geometric.metrics.link_pred import LinkPredMetric

from .base import ShadowModel
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.utils import timer, build_normalized_adj, model_fingerprint, compute_recommendation_metrics
from sralt.utils import DataLoader, EarlyStopping


class CSA(ShadowModel):
    """
    Code Semantic Augmentation (CSA) shadow model.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        num_layers: int,
        decay: float,
        num_quantizers: int,
        codebook_size: int,
        commitment_weight: float,
        quantization: bool,
        feature: Literal['both', 'social', 'behavioral']
    ):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embedding vectors.
        :param num_layers: Number of LightGCN layers.
        :param decay: Weight decay factor for regularization.
        :param num_quantizers: Number of quantizers in RVQ.
        :param codebook_size: Size of the codebook in RVQ.
        :param commitment_weight: Weight for the RVQ commitment loss.
        :param quantization: Whether to use quantized embeddings when computing scores in the behavioral branch.
        :param feature: Which branch features to use for attack, either 'both', 'social', or 'behavioral'.
        """
        super().__init__(num_users, num_items)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.decay = decay
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.quantization = quantization
        self.feature: Literal['both', 'social', 'behavioral'] = feature

        self.num_codes = num_quantizers * codebook_size
        self.social_branch = LightGCN(num_users + num_items + self.num_codes, embed_dim, num_layers)
        self.behavioral_branch = LightGCN(num_users + num_items, embed_dim, num_layers)

        self.rvq = ResidualVQ(
            dim=embed_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight
        )

    def _build_adj(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the sparse adjacency matrices for both augmented social graph and behavioral graph.
        When deterministic algorithms are enabled, returns the original edge indices to avoid non-determinism in sparse matrix operations.
        This function requires the edge indices to remain immutable to maintain object consistency.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social and semantic connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: A tuple of social and behavioral sparse adjacency matrices or edge indices depending on the deterministic algorithm setting.
        """
        if torch.are_deterministic_algorithms_enabled():
            return social_edge_index, behavioral_edge_index

        if not hasattr(self, '_adj_cache'):
            self._adj_cache: dict[str, dict[str, torch.Tensor]] = {}

        results = []
        for key, edge_index in [('social', social_edge_index), ('behavioral', behavioral_edge_index)]:
            num_nodes = self.num_users + self.num_items + self.num_codes if key == 'social' else self.num_users + self.num_items

            if self._adj_cache.get(key, {}).get('edge_index') is not edge_index:
                adj = build_normalized_adj(edge_index, num_nodes, normalization='none')
                self._adj_cache[key] = {'edge_index': edge_index, 'adj': adj}

            results.append(self._adj_cache[key]['adj'])

        return tuple(results)

    def forward(
        self,
        branch: Literal['social', 'behavioral'],
        social_edge_index: torch.Tensor,
        behavioral_edge_index: torch.Tensor,
        target_edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass depending on the current training branch.

        :param branch: Current training branch, either 'social' or 'behavioral'.
        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social and semantic connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape (num_target_edges,) for the target interactions and scalar commitment loss.
        """
        social_adj, behavioral_adj = self._build_adj(social_edge_index, behavioral_edge_index)

        match branch:
            case 'social':
                return self.social_branch(social_adj, target_edge_index), torch.tensor(0.0)

            case 'behavioral':
                embeds = self.behavioral_branch.get_embedding(behavioral_adj)
                quantized, _, commitment_loss = self.rvq(embeds[:self.num_users])
                quantized = torch.cat([quantized, embeds[self.num_users:]], dim=0)
                commitment_loss: torch.Tensor

                embeds = quantized if self.quantization else embeds
                src = embeds[target_edge_index[0]]
                dst = embeds[target_edge_index[1]]

                return (src * dst).sum(dim=-1), commitment_loss.sum()

            case _:
                raise ValueError(f'Invalid branch "{branch}". Must be "social" or "behavioral".')

    def _normalize_embeds(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Normalize the embeddings by the mean norm of user and item embeddings respectively.

        :param embeds: The input embeddings of shape (num_users + num_items, embed_dim).
        :return: Normalized embeddings of the same shape.
        """
        user_embeds, item_embeds = torch.split(embeds, [self.num_users, self.num_items], dim=0)

        user_embeds_norm: torch.Tensor = user_embeds.norm(dim=1)
        item_embeds_norm: torch.Tensor = item_embeds.norm(dim=1)

        user_embeds = user_embeds / user_embeds_norm.mean().clamp(min=1e-8)
        item_embeds = item_embeds / item_embeds_norm.mean().clamp(min=1e-8)

        return torch.cat([user_embeds, item_embeds], dim=0)

    def get_embedding(
        self,
        branch: Literal['social', 'behavioral', 'both'],
        social_edge_index: torch.Tensor,
        behavioral_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the user and item embeddings of the specified branch.

        :param branch: Target branch to get embeddings for, either 'social', 'behavioral', or 'both' for fused embeddings.
        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social and semantic connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Embedding of shape (num_users + num_items, embed_dim) for single branch, or concatenated embeddings of shape
                 (num_users + num_items, 2 * embed_dim) if branch is 'both'.
        """
        social_adj, behavioral_adj = self._build_adj(social_edge_index, behavioral_edge_index)

        match branch:
            case 'social': return self.social_branch.get_embedding(social_adj)[:self.num_users + self.num_items]
            case 'behavioral': return self.behavioral_branch.get_embedding(behavioral_adj)

            case 'both':
                social_embeds = self.social_branch.get_embedding(social_adj)[:self.num_users + self.num_items]
                behavioral_embeds = self.behavioral_branch.get_embedding(behavioral_adj)

                fused_embeds = torch.cat([
                    self._normalize_embeds(social_embeds),
                    self._normalize_embeds(behavioral_embeds)
                ], dim=1)

                return fused_embeds

            case _: raise ValueError(f'Invalid branch "{branch}". Must be "social", "behavioral", or "both".')

    def recommendation_loss(
        self,
        branch: Literal['social', 'behavioral'],
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the recommendation loss for the current branch.

        :param branch: Current training branch, either 'social' or 'behavioral'.
        :param pos_scores: Scores for positive interactions.
        :param neg_scores: Scores for negative interactions.
        :param nodes: Nodes involved in the interactions.
        :return: Recommendation loss value.
        """
        match branch:
            case 'social': return self.social_branch.recommendation_loss(pos_scores, neg_scores, nodes, self.decay)
            case 'behavioral': return self.behavioral_branch.recommendation_loss(pos_scores, neg_scores, nodes, self.decay)
            case _: raise ValueError(f'Invalid branch "{branch}". Must be "social" or "behavioral".')

    def _augment_social_graph(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Augments the social graph with bipartite edges between users and their semantic codes.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Augmented undirected edge index tensor including user-code edges.
        """
        device = social_edge_index.device

        behavioral_embeds = self.get_embedding('behavioral', social_edge_index, behavioral_edge_index)
        _, user_codes, _ = self.rvq(behavioral_embeds[:self.num_users])

        offset = torch.arange(self.num_quantizers, device=device) * self.codebook_size
        user_codes = user_codes + offset.unsqueeze(0)

        user_indices = torch.arange(self.num_users, device=device).repeat_interleave(self.num_quantizers)
        code_indices = (user_codes + self.num_users + self.num_items).flatten()

        user_code_edge_index = torch.stack([user_indices, code_indices], dim=0)
        augmented_edge_index = torch.cat([social_edge_index, user_code_edge_index], dim=1)
        return coalesce(to_undirected(augmented_edge_index))

    @timer('Training branch "{branch}" of model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def _train_branch(
        self, *, branch: Literal['social', 'behavioral'],
        social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor,
        train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, optimizer: Optimizer,
        metrics: list[LinkPredMetric], early_stopping: EarlyStopping, writer: SummaryWriter | None
    ):
        """
        Train a specific branch of the model.

        :param branch: Current training branch, either 'social' or 'behavioral'.
        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social (and semantic) connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        :param num_epochs: Maximum number of training epochs.
        :param optimizer: Optimizer for training.
        :param metrics: List of link prediction metrics to compute during validation.
        :param early_stopping: EarlyStopping helper for managing early stopping.
        :param writer: Optional TensorBoard writer for logging.
        """
        branch_name = f'{self.__class__.__name__}/{branch.capitalize()}Branch'
        metric_names = [
            f'{typing.cast(str, metric.__class__.__name__).removeprefix("LinkPred")}@{metric.k}'
            for metric in metrics
        ]
        logging.info(f'Starting {branch} branch training...')

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

                scores, commitment_loss = self(
                    branch, social_edge_index, behavioral_edge_index, torch.cat([pos_edges, neg_edges], dim=1)
                )
                pos_scores, neg_scores = scores.chunk(2)
                recommendation_loss = self.recommendation_loss(branch, pos_scores, neg_scores, node_ids)

                loss: torch.Tensor = recommendation_loss + commitment_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * users.size(0)
                total_samples += users.size(0)

            average_loss = total_loss / total_samples
            if writer is not None:
                writer.add_scalar(f'{branch_name}/Loss/train', average_loss, epoch)

            self.eval()
            metric_results = compute_recommendation_metrics(
                embeds=self.get_embedding(branch, social_edge_index, behavioral_edge_index),
                metrics=metrics,
                eval_loader=val_loader
            )
            metric_str = ', '.join([f'{name}={value:.4f}' for name, value in zip(metric_names, metric_results)])
            logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metric_str}')

            if writer is not None:
                for name, value in zip(metric_names, metric_results):
                    writer.add_scalar(f'{branch_name}/{name}/val', value, epoch)

            if early_stopping.step(sum(metric_results)):
                break

        early_stopping.load_best_state()

    @timer('Training model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_model(
        self, *, data: ShadowData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int, num_epochs: int,
        optimizer: DictConfig, metrics: DictConfig, early_stopping: DictConfig,
        **kwargs
    ):
        """
        Train the CSA shadow model.

        :param data: The shadow dataset.
        :param writer: Optional TensorBoard writer for logging.
        :param train_batch_size: Batch size for training.
        :param val_batch_size: Batch size for validation.
        :param val_num_negatives: Number of negative samples per positive sample during validation.
        :param num_epochs: Maximum number of training epochs.
        :param optimizer: Instantiable optimizer config.
        :param metrics: List of link prediction metric instantiation configs.
        :param early_stopping: Instantiable EarlyStopping helper config.
        """
        if kwargs:
            logging.warning(f'Ignoring unexpected additional training config parameters: {kwargs}')

        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)

        train_dataset = TrainShadowDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalShadowDataset(data, split='val', num_negatives=val_num_negatives)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        common_kwargs = {
            'behavioral_edge_index': undirected_behavioral_edge_index,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'num_epochs': num_epochs,
            'metrics': instantiate(metrics),
            'writer': writer
        }

        self._train_branch(
            branch='behavioral',
            social_edge_index=data.social_edge_index,
            optimizer=instantiate(optimizer, params=self.behavioral_branch.parameters()),
            early_stopping=instantiate(early_stopping, model=self),
            **common_kwargs
        )

        if self.feature == 'behavioral':
            return

        self._train_branch(
            branch='social',
            social_edge_index=self._augment_social_graph(data.social_edge_index, undirected_behavioral_edge_index),
            optimizer=instantiate(optimizer, params=self.social_branch.parameters()),
            early_stopping=instantiate(early_stopping, model=self.social_branch),
            **common_kwargs
        )

    @torch.no_grad()
    def test_model(
        self, *, data: ShadowData, metrics: list[LinkPredMetric],
        test_batch_size: int, test_num_negatives: int, **kwargs
    ) -> list[float]:
        """
        Test the CSA shadow model.

        :param data: The shadow dataset.
        :param metrics: A list of link prediction metrics to compute.
        :param writer: Optional TensorBoard writer for logging.
        :param test_batch_size: Batch size for testing.
        :param test_num_negatives: Number of negative samples per positive sample during testing.
        :return: A list of metric values.
        """
        if kwargs:
            logging.warning(f'Ignoring unexpected additional testing config parameters: {kwargs}')

        self.eval()

        behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        social_edge_index = self._augment_social_graph(data.social_edge_index, behavioral_edge_index)

        test_dataset = EvalShadowDataset(data, split='test', num_negatives=test_num_negatives)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        metric_results = compute_recommendation_metrics(
            embeds=self.get_embedding(self.feature, social_edge_index, behavioral_edge_index),
            metrics=metrics,
            eval_loader=test_loader
        )

        return metric_results

    @torch.no_grad()
    def get_user_features(self, data: ShadowData) -> torch.Tensor:
        """
        Get the user features learned by the shadow model for attack purposes.
        Feature dimension is:
        - 6 * embed_dim if using both branches
        - 3 * embed_dim if using only one branch

        :param data: The shadow dataset.
        :return: A tensor of shape (num_users, feature_dim) representing user features.
        """
        self.eval()

        behavioral_edge_index = to_undirected(data.behavioral_edge_index)
        social_edge_index = self._augment_social_graph(data.social_edge_index, behavioral_edge_index)
        embeds = self.get_embedding(self.feature, social_edge_index, behavioral_edge_index)

        user_item_features = []
        user_item_edge_index = data.behavioral_edge_index

        for u in range(self.num_users):
            item_indices = user_item_edge_index[1, user_item_edge_index[0] == u]
            item_embeds = embeds[item_indices]
            user_item_features.append(torch.cat([item_embeds.mean(dim=0), item_embeds.var(dim=0)]))

        user_item_features = torch.stack(user_item_features)
        return torch.cat([embeds[:self.num_users], user_item_features], dim=1)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'num_users={self.num_users}, '
            f'num_items={self.num_items}, '
            f'embed_dim={self.embed_dim}, '
            f'num_layers={self.num_layers}, '
            f'decay={self.decay}, '
            f'num_quantizers={self.num_quantizers}, '
            f'codebook_size={self.codebook_size}, '
            f'commitment_weight={self.commitment_weight}, '
            f'quantization={self.quantization}, '
            f'feature={self.feature}, '
            f'fingerprint={model_fingerprint(self)})'
        )

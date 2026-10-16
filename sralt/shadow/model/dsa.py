import typing
import logging
from typing import Literal

import torch
from tqdm import tqdm
from omegaconf import DictConfig
from torch.optim import Optimizer
from hydra.utils import instantiate
from torch.nn import functional as F
from torch_geometric.nn import LightGCN
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import coalesce, to_undirected
from torch_geometric.metrics.link_pred import LinkPredMetric

from .base import ShadowModel
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.utils import timer, build_normalized_adj, model_fingerprint, compute_recommendation_metrics
from sralt.utils import DataLoader, EarlyStopping


class DSA(ShadowModel):
    """
    Direct Semantic Augmentation (DSA) shadow model.
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
        undirected: bool,
        feature: Literal['both', 'social', 'behavioral']
    ):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embedding vectors.
        :param num_layers: Number of LightGCN layers.
        :param decay: Weight decay factor for regularization.
        :param top_k: Number of top similar users to connect in the social graph.
        :param sim: Similarity criterion for finding similar users.
        :param undirected: Whether to add augmented edges in both directions (undirected) or only from source to target (directed).
        :param feature: Which branch features to use for attack, either 'both', 'social', or 'behavioral'.
        """
        super().__init__(num_users, num_items)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.decay = decay
        self.top_k = top_k
        self.sim = sim
        self.undirected = undirected
        self.feature: Literal['both', 'social', 'behavioral'] = feature

        self.social_branch = LightGCN(num_users + num_items, embed_dim, num_layers)
        self.behavioral_branch = LightGCN(num_users + num_items, embed_dim, num_layers)

        self.random_neighbors: torch.Tensor
        self.random_graph_initialized: torch.Tensor
        if sim == 'random':
            self.register_buffer('random_neighbors', torch.zeros((num_users, top_k), dtype=torch.long))
            self.register_buffer('random_graph_initialized', torch.tensor(False))

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
            if self._adj_cache.get(key, {}).get('edge_index') is not edge_index:
                adj = build_normalized_adj(edge_index, self.num_users + self.num_items, normalization='none')
                self._adj_cache[key] = {'edge_index': edge_index, 'adj': adj}

            results.append(self._adj_cache[key]['adj'])

        return tuple(results)

    def forward(
        self,
        branch: Literal['social', 'behavioral'],
        social_edge_index: torch.Tensor,
        behavioral_edge_index: torch.Tensor,
        target_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass depending on the current training branch.

        :param branch: Current training branch, either 'social' or 'behavioral'.
        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social and semantic connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape (num_target_edges,) for the target interactions.
        """
        social_adj, behavioral_adj = self._build_adj(social_edge_index, behavioral_edge_index)

        match branch:
            case 'social': return self.social_branch(social_adj, target_edge_index)
            case 'behavioral': return self.behavioral_branch(behavioral_adj, target_edge_index)
            case _: raise ValueError(f'Invalid branch "{branch}". Must be "behavioral" or "social".')

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
            case 'social': return self.social_branch.get_embedding(social_adj)
            case 'behavioral': return self.behavioral_branch.get_embedding(behavioral_adj)

            case 'both':
                social_embeds = self.social_branch.get_embedding(social_adj)
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
        Augments the social graph based on user similarity derived from behavioral interactions.
        Connect each user to their top-k most similar users according to the specified similarity criterion.
        Random neighbors are sampled once and saved in the model state for reuse throughout the experiment.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Augmented edge index tensor.
        """
        if self.top_k == 0:
            return social_edge_index

        device = social_edge_index.device
        if self.sim == 'random' and self.random_graph_initialized:
            top_k_similar = self.random_neighbors
        else:
            behavioral_embeds = self.get_embedding('behavioral', social_edge_index, behavioral_edge_index)[:self.num_users]

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
            top_k_similar = torch.topk(similarity, self.top_k, dim=1).indices

            if self.sim == 'random':
                self.random_neighbors.copy_(top_k_similar)
                self.random_graph_initialized.fill_(True)

        source = torch.arange(self.num_users, device=device).unsqueeze(1).expand(-1, self.top_k).flatten()
        target = top_k_similar.flatten()
        augmented_edges = torch.stack([source, target], dim=0)
        augmented_edge_index = torch.cat([social_edge_index, augmented_edges], dim=1)

        if self.undirected:
            augmented_edge_index = to_undirected(augmented_edge_index)

        return coalesce(augmented_edge_index)

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

                pos_scores = self(branch, social_edge_index, behavioral_edge_index, pos_edges)
                neg_scores = self(branch, social_edge_index, behavioral_edge_index, neg_edges)

                loss: torch.Tensor = self.recommendation_loss(branch, pos_scores, neg_scores, node_ids)
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
        Train the DSA shadow model.

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
            early_stopping=instantiate(early_stopping, model=self.behavioral_branch),
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
        Test the DSA shadow model.

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
            f'top_k={self.top_k}, '
            f'sim="{self.sim}", '
            f'undirected={self.undirected}, '
            f'feature="{self.feature}", '
            f'fingerprint={model_fingerprint(self)})'
        )

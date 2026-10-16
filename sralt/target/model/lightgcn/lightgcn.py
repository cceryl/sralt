import logging
import typing
from typing import Callable

import torch
from tqdm import tqdm
from omegaconf import DictConfig
from torch.optim import Optimizer
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import LightGCN as PyGLightGCN
from torch_geometric.utils import to_undirected
from torch_geometric.metrics.link_pred import LinkPredMetric

from sralt.target.model import TargetModel
from sralt.target.data import TargetData, TrainTargetDataset, EvalTargetDataset
from sralt.utils import timer, build_normalized_adj, model_fingerprint
from sralt.utils import compute_recommendation_metrics, recommend_items, DataLoader, EarlyStopping


class LightGCN(TargetModel):
    """
    Non-social recommender using only user-item interactions.
    Paper: 'LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation'.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int, num_layers: int, decay: float):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embeddings.
        :param num_layers: Number of propagation layers; layers 0 through num_layers are averaged.
        :param decay: PyG recommendation_loss L2 coefficient on initial batch embeddings.
        """
        super().__init__(num_users, num_items)

        self.num_nodes = num_users + num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.decay = decay

        self.gcn = PyGLightGCN(self.num_nodes, embed_dim, num_layers)
        self._cached_edge_index: torch.Tensor | None = None
        self._cached_adj: torch.Tensor | None = None

    def _apply(self, fn: Callable, recurse: bool = True):
        self._cached_edge_index = None
        self._cached_adj = None
        return super()._apply(fn, recurse=recurse)

    def _build_adj(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Cache an unnormalized CSR graph; PyG applies symmetric normalization during propagation.

        The undirected input edge index must remain immutable. Deterministic mode uses edge indices.
        """
        if torch.are_deterministic_algorithms_enabled():
            return edge_index

        if self._cached_edge_index is not edge_index:
            self._cached_adj = build_normalized_adj(edge_index, self.num_nodes, normalization='none')
            self._cached_edge_index = edge_index

        return typing.cast(torch.Tensor, self._cached_adj)

    def get_embedding(self, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param behavioral_edge_index: Undirected user-item edges with global node IDs.
        :return: Embeddings of shape (num_users + num_items, embed_dim).
        """
        return self.gcn.get_embedding(self._build_adj(behavioral_edge_index))

    def forward(self, behavioral_edge_index: torch.Tensor, target_edge_index: torch.Tensor) -> torch.Tensor:
        """Score target edges of shape (2, num_edges) using global user/item IDs."""
        return self.gcn(self._build_adj(behavioral_edge_index), target_edge_index)

    def calculate_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
        Compute mean BPR loss plus L2 regularization on initial embeddings.

        :param pos_scores: Positive scores of shape (batch_size,).
        :param neg_scores: Negative scores of shape (batch_size,).
        :param nodes: Concatenated users, positive items and negative items, retaining repeated IDs.
        :return: Scalar loss, with regularization normalized by the number of triplets.
        """
        return self.gcn.recommendation_loss(pos_scores, neg_scores, nodes, lambda_reg=self.decay)

    def _train_model(
        self, *, data: TargetData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int, num_epochs: int,
        optimizer: Optimizer, metrics: list[LinkPredMetric], early_stopping: EarlyStopping
    ):
        """Train on behavioral training edges and restore the best validation state."""
        behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        train_loader = DataLoader(TrainTargetDataset(data), batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(
            EvalTargetDataset(data, split='val', num_negatives=val_num_negatives),
            batch_size=val_batch_size, shuffle=False
        )

        model_name = self.__class__.__name__
        metric_names = [
            f'{typing.cast(str, metric.__class__.__name__).removeprefix("LinkPred")}@{metric.k}'
            for metric in metrics
        ]

        for epoch in range(1, num_epochs + 1):
            self.train()
            total_loss = 0.0
            total_samples = 0

            for users, pos_items, neg_items in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
                optimizer.zero_grad()
                pos_edges = torch.stack([users, pos_items], dim=0)
                neg_edges = torch.stack([users, neg_items], dim=0)
                nodes = torch.cat([users, pos_items, neg_items])

                # Share one graph propagation between the positive and negative scores.
                scores = self(behavioral_edge_index, torch.cat([pos_edges, neg_edges], dim=1))
                pos_scores, neg_scores = scores.chunk(2)
                loss = self.calculate_loss(pos_scores, neg_scores, nodes)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * users.size(0)
                total_samples += users.size(0)

            average_loss = total_loss / total_samples
            if writer is not None:
                writer.add_scalar(f'{model_name}/Loss/train', average_loss, epoch)

            self.eval()
            with torch.no_grad():
                metric_results = compute_recommendation_metrics(
                    embeds=self.get_embedding(behavioral_edge_index), metrics=metrics, eval_loader=val_loader
                )
            metric_str = ', '.join(f'{name}={value:.4f}' for name, value in zip(metric_names, metric_results))
            logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metric_str}')

            if writer is not None:
                for name, value in zip(metric_names, metric_results):
                    writer.add_scalar(f'{model_name}/{name}/val', value, epoch)

            if early_stopping.step(sum(metric_results)):
                break

        early_stopping.load_best_state()

    @timer('Training model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_model(
        self, *, data: TargetData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int, num_epochs: int,
        optimizer: DictConfig, metrics: DictConfig, early_stopping: DictConfig, **kwargs
    ):
        """Train with Hydra optimizer, validation metric and early stopping configs."""
        if kwargs:
            logging.warning(f'Ignoring unexpected additional training config parameters: {kwargs}')

        self._train_model(
            data=data, writer=writer,
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
            val_num_negatives=val_num_negatives, num_epochs=num_epochs,
            optimizer=instantiate(optimizer, params=self.parameters()),
            metrics=instantiate(metrics), early_stopping=instantiate(early_stopping, model=self)
        )

    @torch.no_grad()
    def test_model(
        self, *, data: TargetData, metrics: list[LinkPredMetric],
        test_batch_size: int, test_num_negatives: int, **kwargs
    ) -> list[float]:
        """Evaluate sampled candidates using only the behavioral training graph for propagation."""
        if kwargs:
            logging.warning(f'Ignoring unexpected additional testing config parameters: {kwargs}')

        self.eval()
        behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        test_loader = DataLoader(
            EvalTargetDataset(data, split='test', num_negatives=test_num_negatives),
            batch_size=test_batch_size, shuffle=False
        )
        return compute_recommendation_metrics(
            embeds=self.get_embedding(behavioral_edge_index), metrics=metrics, eval_loader=test_loader
        )

    @torch.no_grad()
    def recommend(self, *, data: TargetData, num_recommendations: int, batch_size: int) -> torch.Tensor:
        """Query with fixed parameters and full interaction history; return global user-item IDs."""
        self.eval()
        behavioral_edge_index = data.behavioral_edge_index
        return recommend_items(
            embeds=self.get_embedding(to_undirected(behavioral_edge_index)),
            num_users=self.num_users, num_recommendations=num_recommendations,
            mask_edge_index=behavioral_edge_index, batch_size=batch_size
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'num_users={self.num_users}, '
            f'num_items={self.num_items}, '
            f'embed_dim={self.embed_dim}, '
            f'num_layers={self.num_layers}, '
            f'decay={self.decay}, '
            f'fingerprint={model_fingerprint(self)})'
        )

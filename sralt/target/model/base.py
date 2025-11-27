import typing
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected
from torch_geometric.metrics.link_pred import LinkPredMetric

from sralt.target.data import TargetData
from sralt.utils import EarlyStopping, DataLoader
from sralt.utils import timer


class TargetModel(nn.Module, ABC):
    """
    Base class for target social recommendation models.
    """

    def __init__(self, num_users: int, num_items: int):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items

    @abstractmethod
    def forward(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Score tensor of shape specified by the actual model.
        """
        pass

    @abstractmethod
    def get_embedding(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings of the model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Embedding tensor of shape (num_nodes, embed_dim).
        """
        pass

    @abstractmethod
    def calculate_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss for the model.

        :param pos_scores: Score tensor for positive edges.
        :param neg_scores: Score tensor for negative edges.
        :param nodes: Node tensor of shape (num_nodes_in_batch,) representing nodes in the current batch.
        :return: Loss tensor of shape specified by the actual model.
        """
        pass

    @torch.no_grad()
    def compute_metrics_at_k(self, data: TargetData, metrics: list[LinkPredMetric], eval_loader: DataLoader) -> list[float]:
        """
        Compute multiple metrics (e.g., Recall, NDCG) at given k(s).

        :param data: The data on which to compute the metrics.
        :param metrics: A list of metric calculators.
        :param eval_loader: DataLoader for evaluation data.
        :return: The computed metrics as a list of floats.
        """
        self.eval()

        split = getattr(eval_loader.dataset, 'split')
        if split == 'val':
            behavioral_edge_index = data.behavioral_train_edge_index
        elif split == 'test':
            behavioral_edge_index = torch.cat([data.behavioral_train_edge_index, data.behavioral_val_edge_index], dim=1)
        else:
            raise ValueError(f'Unknown split: {split}')

        # Extract user and item embeddings
        embeds = self.get_embedding(data.social_edge_index, to_undirected(behavioral_edge_index))
        user_embeds, item_embeds = torch.split(embeds, [data.num_users, data.num_items], dim=0)

        # Prepare metrics calculators
        metrics = [metric.to(data.device) for metric in metrics]
        max_k = max([metric.k for metric in metrics])
        for metric in metrics:
            metric.reset()

        for users, pos_items, neg_items in eval_loader:
            assert max_k <= neg_items.size(1) + 1, 'Number of candidate items must be at least k.'

            # Adjust indices to 0-based
            users: torch.Tensor = users - data.user_range.start
            pos_items: torch.Tensor = pos_items - data.item_range.start
            neg_items: torch.Tensor = neg_items - data.item_range.start

            # Calculate scores for candidate items
            candidate_items = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
            user_embeds_batch = user_embeds[users].unsqueeze(1)
            item_embeds_batch = item_embeds[candidate_items]
            scores = (user_embeds_batch * item_embeds_batch).sum(dim=-1)

            # Get top-k item indices among candidate items for each user
            _, topk_local = torch.topk(scores, k=max_k, sorted=True)
            topk_global = torch.gather(candidate_items, 1, topk_local)

            # Update all calculators
            users_range = torch.arange(users.size(0), device=data.device)
            edge_label_index = torch.stack([users_range, pos_items])

            for metric in metrics:
                metric.update(topk_global[:, :metric.k], edge_label_index)

        return [metric.compute().item() for metric in metrics]

    @timer('Training model "{args[0].__class__.__name__}" took {elapsed:.2f} seconds.')
    def train_model(
        self,
        *,
        data: TargetData,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        optimizer_cls: type[optim.Optimizer],
        optimizer_kwargs: dict[str, Any],
        metrics: list[LinkPredMetric],
        metric_aggregator: Callable[[float, list[float]], float],
        early_stopping_kwargs: dict[str, Any],
        writer: SummaryWriter | None = None
    ) -> dict[str, float]:
        """
        Generic training loop shared by target and shadow recommendation models.

        :param data: The target or shadow dataset.
        :param train_loader: Loader that yields training triplets (users, pos_items, neg_items).
        :param val_loader: Loader that yields validating triplets (users, pos_items, neg_items).
        :param test_loader: Loader that yields testing triplets (users, pos_items, neg_items).
        :param num_epochs: Maximum number of epochs to train.
        :param optimizer_cls: Optimizer class to instantiate.
        :param optimizer_kwargs: Keyword arguments for optimizer construction.
        :param metrics: Collection of ranking metrics evaluated every epoch and returned on test.
        :param metric_aggregator: Callable that aggregates loss/metrics for early stopping.
        :param early_stopping_kwargs: Keyword arguments for the EarlyStopping helper.
        :param writer: Optional TensorBoard writer used for logging scalars.
        :return: Dictionary containing final test metrics.
        """
        optimizer = optimizer_cls(self.parameters(), **optimizer_kwargs)
        early_stopping = EarlyStopping(self, **early_stopping_kwargs)

        undirected_behavioral_train_edge_index = to_undirected(data.behavioral_train_edge_index)
        model_name = self.__class__.__name__
        metrics_names = [
            typing.cast(str, metric.__class__.__name__).removeprefix('LinkPred') + f'@{metric.k}' for metric in metrics
        ]

        for epoch in range(1, num_epochs + 1):
            self.train()

            total_loss = 0.0
            total_samples = 0

            for users, pos_items, neg_items in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
                optimizer.zero_grad()

                pos_edges = torch.stack([users, pos_items], dim=0)
                neg_edges = torch.stack([users, neg_items], dim=0)
                node_ids = torch.cat([users, pos_items, neg_items])

                pos_scores: torch.Tensor = self(
                    data.social_edge_index, undirected_behavioral_train_edge_index, pos_edges
                )
                neg_scores: torch.Tensor = self(
                    data.social_edge_index, undirected_behavioral_train_edge_index, neg_edges
                )

                loss = self.calculate_loss(pos_scores, neg_scores, node_ids)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * users.size(0)
                total_samples += users.size(0)

            average_loss = total_loss / total_samples if total_samples > 0 else float('nan')
            if writer is not None:
                writer.add_scalar(f'{model_name}/Loss/train', average_loss, epoch)

            metrics_results = self.compute_metrics_at_k(data, metrics, val_loader)
            metrics_str = ', '.join([f'{name}={value:.4f}' for name, value in zip(metrics_names, metrics_results)])
            logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metrics_str}')

            if writer is not None:
                for name, value in zip(metrics_names, metrics_results):
                    writer.add_scalar(f'{model_name}/{name}/val', value, epoch)

            aggregate_metric = metric_aggregator(average_loss, metrics_results)
            if early_stopping.step(aggregate_metric):
                break

        early_stopping.load_best_state()

        test_metrics = self.compute_metrics_at_k(data, metrics, test_loader)
        return {name: value for name, value in zip(metrics_names, test_metrics)}

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation and fingerprint of the model.
        """
        pass

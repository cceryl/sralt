import typing
import logging
from abc import abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Metric
from torch.utils.tensorboard import SummaryWriter

from sralt.utils import EarlyStopping, DataLoader
from sralt.utils import timer


class AttackModel(nn.Module):
    """
    Base class for social connection attack models.
    """

    def __init__(self, user_feature_dim: int):
        """
        Initialize the AttackModel.

        :param user_feature_dim: Dimension of user features.
        """
        super().__init__()

        self.user_feature_dim = user_feature_dim

    @abstractmethod
    def forward(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate the user social connection logits.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: The user social connection logits tensor of shape (batch_size,).
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def predict(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Predict the attack outcome.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: The predicted probability tensor of shape (batch_size,).
        """
        pass

    @torch.no_grad()
    def compute_attack_metrics(self, user_features: torch.Tensor, eval_loader: DataLoader, metrics: list[Metric]) -> list[float]:
        """
        Compute binary classification metrics.

        :param user_features: The user features tensor of shape (num_users, user_feature_dim).
        :param eval_loader: DataLoader for evaluation data.
        :param metrics: A list of metric calculators.
        :return: A list of computed metric values.
        """
        self.eval()

        metrics = [metric.to(user_features.device) for metric in metrics]
        for metric in metrics:
            metric.reset()

        for user1, user2, labels in eval_loader:
            probs = self.predict(torch.stack([user1, user2]), user_features)

            for metric in metrics:
                metric.update(probs, labels)

        return [typing.cast(torch.Tensor, metric.compute()).item() for metric in metrics]

    @timer('Training model {args[0].__class__.__name__} took {elapsed:.2f} seconds.')
    def train_model(
        self,
        *,
        user_features: torch.Tensor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        loss_fn: nn.Module,
        optimizer_cls: type[optim.Optimizer],
        optimizer_kwargs: dict[str, Any],
        metrics: list[Metric],
        metric_aggregator: Callable[[float, list[float]], float],
        early_stopping_kwargs: dict[str, Any],
        writer: SummaryWriter | None = None
    ) -> dict[str, float]:
        """
        Generic training loop shared by attack models.

        :param user_features: The user features tensor of shape (num_users, user_feature_dim).
        :param train_loader: Loader yielding training triplets (user1, user2, labels).
        :param val_loader: Loader yielding validating triplets (user1, user2, labels).
        :param test_loader: Loader yielding testing triplets (user1, user2, labels).
        :param num_epochs: Maximum number of epochs.
        :param loss_fn: Loss function used for optimization.
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

        metrics_names = [metric.__class__.__name__.removeprefix('Binary') for metric in metrics]
        model_name = self.__class__.__name__

        for epoch in range(1, num_epochs + 1):
            self.train()

            total_loss = 0.0
            total_samples = 0

            for user1, user2, labels in train_loader:
                optimizer.zero_grad()

                logits = self(torch.stack([user1, user2]), user_features)

                labels: torch.Tensor
                loss: torch.Tensor = loss_fn(logits, labels.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

            average_loss = total_loss / total_samples if total_samples > 0 else float('nan')
            if writer is not None:
                writer.add_scalar(f'{model_name}/Loss/train', average_loss, epoch)

            metrics_results = self.compute_attack_metrics(user_features, val_loader, metrics)
            metrics_str = ', '.join(f'{name}={value:.4f}' for name, value in zip(metrics_names, metrics_results))
            logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metrics_str}')

            if writer is not None:
                for name, value in zip(metrics_names, metrics_results):
                    writer.add_scalar(f'{model_name}/{name}/val', value, epoch)

            aggregate_metric = metric_aggregator(average_loss, metrics_results)
            if early_stopping.step(aggregate_metric):
                break

        early_stopping.load_best_state()

        test_metrics = self.compute_attack_metrics(user_features, test_loader, metrics)
        return {name: value for name, value in zip(metrics_names, test_metrics)}

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation and fingerprint of the attack model.
        """
        pass

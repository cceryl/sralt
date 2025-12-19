import types
import logging
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Metric
from torch.utils.tensorboard import SummaryWriter

from . import register_attack_model
from .base import AttackModel
from sralt.attack.data import AttackData, TrainAttackDataset, EvalAttackDataset
from sralt.utils import timer, calculate_model_fingerprint, compute_attack_metrics
from sralt.utils import DataLoader, EarlyStopping


@register_attack_model('SMIA')
class SMIAAttack(AttackModel):
    """
    SMIA attack model for social connection prediction.
    """

    def __init__(self, user_feature_dim: int, dropout: float = 0.5, leaky_relu_slope: float = 0.2, pos_weight: float = 10.0):
        """
        Initialize the SMIA attack model.

        :param user_feature_dim: Dimension of user features.
        :param dropout: Dropout rate.
        :param leaky_relu_slope: Slope for LeakyReLU activation.
        :param pos_weight: Positive class weight for BCE loss.
        """
        super().__init__(user_feature_dim)
        self.dropout = dropout
        self.leaky_relu_slope = leaky_relu_slope
        self.pos_weight = pos_weight

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.net = nn.Sequential(
            nn.Linear(4 * user_feature_dim, 4 * user_feature_dim, bias=True),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(4 * user_feature_dim, 2 * user_feature_dim, bias=True),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(2 * user_feature_dim, 1, bias=True)
        )

    def forward(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate the user social connection logits.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: The user social connection logits tensor of shape (batch_size,).
        """
        user1, user2 = user_edge_index
        user1_features = user_features[user1]
        user2_features = user_features[user2]

        diff_features = torch.abs(user1_features - user2_features)
        prod_features = user1_features * user2_features

        combined_features = torch.cat([user1_features, user2_features, diff_features, prod_features], dim=1)

        logits: torch.Tensor = self.net(combined_features)
        return logits.squeeze(dim=1)

    @torch.no_grad()
    def predict(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Predict the attack outcome.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: The predicted probability tensor of shape (batch_size,).
        """
        logits = self.forward(user_edge_index, user_features)
        probs = torch.sigmoid(logits)
        return probs

    @timer('Training model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_model(
        self, *, data: AttackData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int,
        num_epochs: int, optimizer_cls: type[optim.Optimizer], optimizer_kwargs: dict[str, Any],
        metrics: list[Metric], metric_aggregator: types.CodeType, early_stopping_kwargs: dict[str, Any],
        **kwargs
    ):
        """
        Train the attack model.

        :param data: The attack dataset.
        :param writer: Optional TensorBoard writer for logging.
        :train_batch_size: Batch size for training.
        :val_batch_size: Batch size for validation.
        :num_epochs: Number of training epochs.
        :optimizer_cls: Optimizer class.
        :optimizer_kwargs: Keyword arguments for the optimizer.
        :metrics: A list of metric calculators.
        :metric_aggregator: A function to aggregate metrics for early stopping.
        :early_stopping_kwargs: Keyword arguments for early stopping.
        """
        train_dataset = TrainAttackDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalAttackDataset(data, split='val')
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        optimizer = optimizer_cls(self.parameters(), **optimizer_kwargs)
        early_stopping = EarlyStopping(self, **early_stopping_kwargs)
        metric_aggregator_fn: Callable[[float, list[float]], float] = eval(metric_aggregator)

        model_name = self.__class__.__name__
        metrics_names = [metric.__class__.__name__.removeprefix('Binary') for metric in metrics]

        for epoch in range(1, num_epochs + 1):
            self.train()

            total_loss = 0.0
            total_samples = 0

            for user1, user2, labels in train_loader:
                optimizer.zero_grad()

                logits = self(torch.stack([user1, user2]), data.user_features)

                labels: torch.Tensor
                loss: torch.Tensor = self.loss_fn(logits, labels.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

            average_loss = total_loss / total_samples if total_samples > 0 else float('nan')
            if writer is not None:
                writer.add_scalar(f'{model_name}/Loss/train', average_loss, epoch)

            self.eval()
            metrics_results = compute_attack_metrics(
                device=data.device,
                predictor=lambda u1, u2: self.predict(torch.stack([u1, u2]), data.user_features),
                metrics=metrics,
                eval_loader=val_loader
            )
            metrics_str = ', '.join(f'{name}={value:.4f}' for name, value in zip(metrics_names, metrics_results))
            logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metrics_str}')

            if writer is not None:
                for name, value in zip(metrics_names, metrics_results):
                    writer.add_scalar(f'{model_name}/{name}/val', value, epoch)

            aggregate_metric = metric_aggregator_fn(average_loss, metrics_results)
            if early_stopping.step(aggregate_metric):
                break

        early_stopping.load_best_state()

    @torch.no_grad()
    def test_model(self, *, data: AttackData, metrics: list[Metric], test_batch_size: int, **kwargs) -> list[float]:
        """
        Test the attack model.

        :param data: The attack dataset.
        :param metrics: A list of metric calculators.
        :param test_batch_size: Batch size for testing.
        :return: A list of computed metric values.
        """
        self.eval()

        test_dataset = EvalAttackDataset(data, split='test')
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        metrics_results = compute_attack_metrics(
            device=data.device,
            predictor=lambda u1, u2: self.predict(torch.stack([u1, u2]), data.user_features),
            metrics=metrics,
            eval_loader=test_loader
        )

        return metrics_results

    def __repr__(self) -> str:
        """
        String representation and fingerprint of the attack model.
        """
        return (
            f'{self.__class__.__name__}('
            f'user_feature_dim={self.user_feature_dim}, '
            f'dropout={self.dropout}, '
            f'leaky_relu_slope={self.leaky_relu_slope}, '
            f'pos_weight={self.pos_weight}, '
            f'fingerprint={calculate_model_fingerprint(self)})'
        )

import logging

import torch
import torch.nn as nn
from torchmetrics import Metric
from omegaconf import DictConfig
from torch.optim import Optimizer
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter

from .base import AttackModel
from sralt.attack.data import AttackData, TrainAttackDataset, EvalAttackDataset
from sralt.utils import timer, head_tail_split, model_fingerprint, compute_attack_metrics
from sralt.utils import DataLoader, EarlyStopping


class MLP(AttackModel):
    """
    MLP-based classifier for social connection attack.
    """

    def __init__(self, user_feature_dim: int, dropout: float, leaky_relu_slope: float, pos_weight: float):
        """
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
            nn.Linear(4 * user_feature_dim, 4 * user_feature_dim),
            nn.BatchNorm1d(4 * user_feature_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(4 * user_feature_dim, 2 * user_feature_dim),
            nn.BatchNorm1d(2 * user_feature_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(2 * user_feature_dim, 1)
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

    def _train_model(
        self, *, data: AttackData, tail_ratio: float, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, negative_ratio: float, num_epochs: int,
        optimizer: Optimizer, metrics: list[Metric], early_stopping: EarlyStopping
    ):
        """
        Train the attack model.

        :param data: The attack dataset.
        :param tail_ratio: Proportion of users classified as long-tail users based on social connection counts.
        :param writer: Optional TensorBoard writer for logging.
        :param train_batch_size: Batch size for training.
        :param val_batch_size: Batch size for validation.
        :param negative_ratio: The ratio of negative samples to positive samples in each epoch in training.
        :param num_epochs: Number of training epochs.
        :param optimizer: Optimizer for training.
        :param metrics: List of binary classification metrics to compute during validation.
        :param early_stopping: EarlyStopping helper for managing early stopping.
        """
        train_dataset = TrainAttackDataset(data, negative_ratio=negative_ratio)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalAttackDataset(data, split='val')
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        model_name = self.__class__.__name__
        metric_names = [metric.__class__.__name__.removeprefix('Binary') for metric in metrics]
        head_indices, tail_indices = head_tail_split(data.num_users, data.target_social_edge_index, tail_ratio)

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
            all_results, head_results, tail_results = compute_attack_metrics(
                device=data.device,
                predictor=lambda u1, u2: self.predict(torch.stack([u1, u2]), data.user_features),
                metrics=metrics,
                eval_loader=val_loader,
                head_indices=head_indices,
                tail_indices=tail_indices
            )

            def metrics_to_str(results: list[float]) -> str:
                return ', '.join(f'{name}={value:.4f}' for name, value in zip(metric_names, results))

            logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metrics_to_str(all_results)}')
            logging.info(f'    Head users: {metrics_to_str(head_results)}')
            logging.info(f'    Tail users: {metrics_to_str(tail_results)}')

            if writer is not None:
                for group, results in [('All', all_results), ('Head', head_results), ('Tail', tail_results)]:
                    for name, value in zip(metric_names, results):
                        writer.add_scalar(f'{model_name}/{name}/{group}', value, epoch)

            if early_stopping.step(sum(all_results)):
                break

        early_stopping.load_best_state()

    @timer('Training model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_model(
        self, *, data: AttackData, tail_ratio: float, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, negative_ratio: float, num_epochs: int,
        optimizer: DictConfig, metrics: DictConfig, early_stopping: DictConfig,
        **kwargs
    ):
        """
        Train the attack model.

        :param data: The attack dataset.
        :param tail_ratio: Proportion of users classified as long-tail users based on social connection counts.
        :param writer: Optional TensorBoard writer for logging.
        :param train_batch_size: Batch size for training.
        :param val_batch_size: Batch size for validation.
        :param negative_ratio: The ratio of negative samples to positive samples in each epoch in training.
        :param num_epochs: Number of training epochs.
        :param optimizer: Instantiable optimizer config.
        :param metrics: List of binary classification metric instantiation configs.
        :param early_stopping: Instantiable EarlyStopping helper config.
        """
        if kwargs:
            logging.warning(f'Ignoring unexpected additional training config parameters: {kwargs}')

        self._train_model(
            data=data,
            tail_ratio=tail_ratio,
            writer=writer,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            negative_ratio=negative_ratio,
            num_epochs=num_epochs,
            optimizer=instantiate(optimizer, params=self.parameters()),
            metrics=instantiate(metrics),
            early_stopping=instantiate(early_stopping, model=self)
        )

    @torch.no_grad()
    def test_model(
        self, *, data: AttackData, tail_ratio: float, metrics: list[Metric], test_batch_size: int, **kwargs
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Test the attack model.

        :param data: The attack dataset.
        :param tail_ratio: The ratio to be considered as tail users.
        :param metrics: A list of metric calculators.
        :param test_batch_size: Batch size for testing.
        :return: A tuple of lists of computed metric values for all, head, and tail groups.
        """
        if kwargs:
            logging.warning(f'Ignoring unexpected additional testing config parameters: {kwargs}')

        self.eval()

        test_dataset = EvalAttackDataset(data, split='test')
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        head_indices, tail_indices = head_tail_split(
            data.num_users, data.target_social_edge_index, tail_ratio=tail_ratio
        )

        results = compute_attack_metrics(
            device=data.device,
            predictor=lambda u1, u2: self.predict(torch.stack([u1, u2]), data.user_features),
            metrics=metrics,
            eval_loader=test_loader,
            head_indices=head_indices,
            tail_indices=tail_indices
        )

        return results

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
            f'fingerprint={model_fingerprint(self)})'
        )

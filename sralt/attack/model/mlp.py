import logging

import torch
import torch.nn as nn
from torchmetrics import Metric
from omegaconf import DictConfig
from torch.optim import Optimizer
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter

from .base import AttackModel
from sralt.attack.data import AttackData, TrainAttackDataset, EvalAttackDataset, BalancedAttackBatchSampler
from sralt.utils import timer, head_tail_split, model_fingerprint, compute_attack_metrics
from sralt.utils import DataLoader, EarlyStopping


class MLP(AttackModel):
    """
    MLP-based classifier for social connection attack.
    """

    def __init__(self, user_feature_dim: int, dropout: float, leaky_relu_slope: float, pos_weight: float,
                 norm: str = 'none'):
        """
        :param user_feature_dim: Dimension of user features.
        :param dropout: Dropout rate.
        :param leaky_relu_slope: Slope for LeakyReLU activation.
        :param pos_weight: Positive class weight for BCE loss.
        :param norm: Optional hidden-layer normalization ('none' or 'layer').
        """
        super().__init__(user_feature_dim)
        self.dropout = dropout
        self.leaky_relu_slope = leaky_relu_slope
        self.pos_weight = pos_weight
        if norm not in {'none', 'layer'}:
            raise ValueError(f'Unknown MLP normalization: {norm}')
        self.norm = norm

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        layers = []
        for input_dim, hidden_dim in [(4 * user_feature_dim, 4 * user_feature_dim),
                                      (4 * user_feature_dim, 2 * user_feature_dim)]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if norm == 'layer':
                layers.append(nn.LayerNorm(hidden_dim))
            layers.extend([nn.LeakyReLU(leaky_relu_slope), nn.Dropout(dropout)])
        layers.append(nn.Linear(2 * user_feature_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate the user social connection logits.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: The user social connection logits tensor of shape (batch_size,).
        """
        user1, user2 = user_edge_index
        user1, user2 = torch.minimum(user1, user2), torch.maximum(user1, user2)
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
        optimizer: Optimizer, metrics: list[Metric], early_stopping: EarlyStopping,
        balanced_batches: bool = False, logit_diagnostics: bool = False
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
        :param balanced_batches: Preserve 1:1 classes within each complete/tail batch.
        :param logit_diagnostics: Record validation score drift without extra predictions.
        """
        train_dataset = TrainAttackDataset(data, negative_ratio=negative_ratio)
        if balanced_batches:
            train_loader = DataLoader(train_dataset, batch_sampler=BalancedAttackBatchSampler(train_dataset, train_batch_size))
        else:
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalAttackDataset(data, split='val')
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        model_name = self.__class__.__name__
        metric_names = [metric.__class__.__name__.removeprefix('Binary') for metric in metrics]
        head_indices, tail_indices = head_tail_split(data.num_users, data.target_social_edge_index, tail_ratio)
        if logit_diagnostics:
            self.training_diagnostics = []
        previous_decisions = None

        for epoch in range(1, num_epochs + 1):
            self.train()

            total_loss = 0.0
            total_samples = 0
            batch_positive_rates = []
            batch_sizes = []

            for user1, user2, labels in train_loader:
                optimizer.zero_grad()

                logits = self(torch.stack([user1, user2]), data.user_features)

                labels: torch.Tensor
                loss: torch.Tensor = self.loss_fn(logits, labels.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                if logit_diagnostics:
                    batch_positive_rates.append(labels.float().mean().item())
                    batch_sizes.append(labels.numel())

            average_loss = total_loss / total_samples
            if writer is not None:
                writer.add_scalar(f'{model_name}/Loss/train', average_loss, epoch)

            self.eval()
            validation_logits = []

            def predict_validation(user1, user2):
                pairs = torch.stack([user1, user2])
                if not logit_diagnostics:
                    return self.predict(pairs, data.user_features)
                logits = self(pairs, data.user_features)
                validation_logits.append(logits.detach())
                return logits.sigmoid()

            all_results, head_results, tail_results = compute_attack_metrics(
                device=data.device,
                predictor=predict_validation,
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

            if logit_diagnostics:
                logits = torch.cat(validation_logits)
                labels = val_dataset.labels.bool()
                decisions = logits > 0  # Same fixed sigmoid threshold .5.
                diagnostic = dict(
                    epoch=epoch, loss=average_loss,
                    validation=dict(zip(metric_names, all_results)),
                    logit_positive_mean=logits[labels].mean().item(),
                    logit_negative_mean=logits[~labels].mean().item(),
                    logit_positive_std=logits[labels].std(unbiased=False).item(),
                    logit_negative_std=logits[~labels].std(unbiased=False).item(),
                    predicted_positive_rate=decisions.float().mean().item(),
                    logit_near_zero_rate=(logits.abs() < .1).float().mean().item(),
                    validation_prediction_flip_rate=(
                        (decisions != previous_decisions).float().mean().item()
                        if previous_decisions is not None else None),
                    batch_positive_rate_min=min(batch_positive_rates),
                    batch_positive_rate_max=max(batch_positive_rates),
                    last_batch_positive_rate=batch_positive_rates[-1],
                    last_batch_size=batch_sizes[-1], num_updates=len(batch_sizes),
                )
                previous_decisions = decisions
                self.training_diagnostics.append(diagnostic)
                logging.info('MLP score diagnostics: %s', diagnostic)
                if writer is not None:
                    for name, value in diagnostic.items():
                        if isinstance(value, float):
                            writer.add_scalar(f'{model_name}/Diagnostics/{name}', value, epoch)

            if writer is not None:
                for group, results in [('All', all_results), ('Head', head_results), ('Tail', tail_results)]:
                    for name, value in zip(metric_names, results):
                        writer.add_scalar(f'{model_name}/{name}/{group}', value, epoch)

            if early_stopping.step(sum(all_results)):
                break

        early_stopping.load_best_state()
        # Small metadata travels with the shared model cache, including on hits.
        self.training_summary = {'best_epoch': early_stopping.best_epoch,
                                 'stop_epoch': early_stopping.epoch,
                                 'selection_score': early_stopping.best_metric}

    @timer('Training model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_model(
        self, *, data: AttackData, tail_ratio: float, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, negative_ratio: float, num_epochs: int,
        optimizer: DictConfig, metrics: DictConfig, early_stopping: DictConfig,
        balanced_batches: bool = False, logit_diagnostics: bool = False,
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
        :param balanced_batches: Optional equal-class training batches; default unchanged.
        :param logit_diagnostics: Optional training/validation score diagnostics.
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
            early_stopping=instantiate(early_stopping, model=self),
            balanced_batches=balanced_batches,
            logit_diagnostics=logit_diagnostics
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
        normalization = f'norm={self.norm}, ' if getattr(self, 'norm', 'none') != 'none' else ''
        return (
            f'{self.__class__.__name__}('
            f'user_feature_dim={self.user_feature_dim}, '
            f'dropout={self.dropout}, '
            f'leaky_relu_slope={self.leaky_relu_slope}, '
            f'pos_weight={self.pos_weight}, {normalization}'
            f'fingerprint={model_fingerprint(self)})'
        )

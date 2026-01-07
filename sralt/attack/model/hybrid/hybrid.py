import types
import typing
import logging
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Metric
from torch_geometric.utils import degree
from torch.utils.tensorboard import SummaryWriter

from .moe import MoE
from .fusion import get_fusion, BaseFusion
from sralt.attack.model import register_attack_model, AttackModel
from sralt.attack.data import AttackData, TrainAttackDataset, EvalAttackDataset
from sralt.utils import timer, head_tail_split, calculate_model_fingerprint, compute_attack_metrics
from sralt.utils import DataLoader, EarlyStopping


@register_attack_model('Hybrid')
class HybridAttack(AttackModel):
    """
    Hybrid Attack Model for Social Connection Prediction.

    Features:
    1. Advanced pairwise feature fusion.
    2. MoE architecture with head/tail user specialization.
    3. Dynamic loss weighting based on user degree and features.
    """

    def __init__(
        self,
        user_feature_dim: int,
        # Feature fusion
        fusion: list[str] = ['sum', 'diff'],
        low_rank: int = 16,
        attention_heads: int = 1,
        # MoE
        num_experts: int = 2,
        dropout: float = 0.5,
        noise_multiplier: float = 1e-2,
        # Loss weighting
        pos_weight: float = 10.0,
        degree_weight_factor: float = 5.0,
        feature_weight_factor: float = 2.0
    ):
        """
        :param user_feature_dim: Dimension of user features.
        :param fusion: Feature fusion methods to use.
        :param low_rank: Rank for low-rank fusion (if used).
        :param attention_heads: Number of attention heads (if used).
        :param num_experts: Number of experts in MoE.
        :param dropout: Dropout rate.
        :param noise_multiplier: Noise multiplier for gating network.
        :param pos_weight: Positive class weight for BCE loss.
        :param degree_weight_factor: Weight factor for degree-based loss weighting.
        :param feature_weight_factor: Weight factor for feature-based loss weighting.
        """
        super().__init__(user_feature_dim)

        # Feature fusion settings
        self.fusion = fusion
        self.low_rank = low_rank
        self.attention_heads = attention_heads

        self.fusers = nn.ModuleList([
            get_fusion(fusion_name)(
                feature_dim=user_feature_dim,
                rank=low_rank,
                num_heads=attention_heads
            ) for fusion_name in fusion
        ])
        self.fused_feature_dim = sum(
            typing.cast(BaseFusion, fuser).fused_feature_dim(user_feature_dim)
            for fuser in self.fusers
        ) + 2 * user_feature_dim

        # MoE settings
        self.num_experts = num_experts
        self.dropout = dropout
        self.noise_multiplier = noise_multiplier

        self.moe = MoE(
            num_experts=num_experts,
            input_dim=self.fused_feature_dim,
            hidden_dim=self.fused_feature_dim * 2,
            dropout=dropout,
            noise_multiplier=noise_multiplier
        )

        # Loss weighting settings
        self.pos_weight = pos_weight
        self.degree_weight_factor = degree_weight_factor
        self.feature_weight_factor = feature_weight_factor
        self.base_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction='none')

        # Cache
        self._shadow_user_degrees = torch.Tensor()
        self._shadow_user_probs = torch.Tensor()
        self._max_degree = torch.tensor(0.0)
        self._min_degree = torch.tensor(0.0)

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

        fused_features = [fuser(user1_features, user2_features) for fuser in self.fusers]
        fused_features = torch.cat(fused_features + [user1_features, user2_features], dim=1)

        sample_density = torch.stack([
            self._shadow_user_probs[user1],
            self._shadow_user_probs[user2]
        ], dim=1)

        logits: torch.Tensor = self.moe(fused_features, sample_density)
        return logits.squeeze(dim=1)

    def compute_dynamic_weights(
        self,
        user_edge_index: torch.Tensor,
        user_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamic loss weights based on user degrees and features.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: Weight tensor of shape (batch_size,).
        """
        user1, user2 = user_edge_index

        # Degree-based weights: give more weight to tail users
        user1_degrees = self._shadow_user_degrees[user1]
        user2_degrees = self._shadow_user_degrees[user2]
        avg_degree = (user1_degrees + user2_degrees) / 2.0
        normalized_degree = (avg_degree - self._min_degree) / (self._max_degree - self._min_degree)

        degree_weights = 1.0 + self.degree_weight_factor * (1.0 - normalized_degree)

        # Feature-based weights: similar features might indicate easier attack
        user1_features = user_features[user1]
        user2_features = user_features[user2]

        feat_similarity = F.cosine_similarity(user1_features, user2_features, dim=1)
        feature_weights = 1.0 + self.feature_weight_factor * (1.0 - feat_similarity)

        combined_weights = degree_weights * feature_weights
        return combined_weights

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
        self, *, data: AttackData, tail_ratio: float, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int,
        num_epochs: int, optimizer_cls: type[optim.Optimizer], optimizer_kwargs: dict[str, Any],
        metrics: list[Metric], metric_aggregator: types.CodeType, early_stopping_kwargs: dict[str, Any],
        **kwargs
    ):
        """
        Train the attack model.

        :param data: The attack dataset.
        :param tail_ratio: The ratio to be considered as tail users.
        :param writer: Optional TensorBoard writer for logging.
        :param train_batch_size: Batch size for training.
        :param val_batch_size: Batch size for validation.
        :param num_epochs: Number of training epochs.
        :param optimizer_cls: Optimizer class.
        :param optimizer_kwargs: Keyword arguments for the optimizer.
        :param metrics: A list of metric calculators.
        :param metric_aggregator: A function to aggregate metrics for early stopping.
        :param early_stopping_kwargs: Keyword arguments for early stopping.
        """
        shadow_edge_index = data.shadow_social_edge_index
        self._shadow_user_degrees = degree(shadow_edge_index[0], num_nodes=data.num_users, dtype=torch.float)
        self._max_degree = self._shadow_user_degrees.max()
        self._min_degree = self._shadow_user_degrees.min()
        self._shadow_user_probs = self._shadow_user_degrees / self._shadow_user_degrees.sum()

        train_dataset = TrainAttackDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalAttackDataset(data, split='val')
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        optimizer = optimizer_cls(self.parameters(), **optimizer_kwargs)
        early_stopping = EarlyStopping(self, **early_stopping_kwargs)
        metric_aggregator_fn: Callable[[float, list[float]], float] = eval(metric_aggregator)

        model_name = self.__class__.__name__
        metrics_names = [metric.__class__.__name__.removeprefix('Binary') for metric in metrics]
        head_indices, tail_indices = head_tail_split(data.num_users, data.target_social_edge_index, tail_ratio)

        for epoch in range(1, num_epochs + 1):
            self.train()

            total_loss = 0.0
            total_samples = 0

            for user1, user2, labels in train_loader:
                optimizer.zero_grad()

                user_edge_index = torch.stack([user1, user2])
                logits = self(user_edge_index, data.user_features)

                labels: torch.Tensor
                loss_per_sample: torch.Tensor = self.base_loss_fn(logits, labels.float())

                weights = self.compute_dynamic_weights(user_edge_index, data.user_features)
                loss = (loss_per_sample * weights).mean()

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
                return ', '.join(f'{name}={value:.4f}' for name, value in zip(metrics_names, results))

            logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metrics_to_str(all_results)}')
            logging.info(f'    Head users: {metrics_to_str(head_results)}')
            logging.info(f'    Tail users: {metrics_to_str(tail_results)}')

            if writer is not None:
                for group, results in [('All', all_results), ('Head', head_results), ('Tail', tail_results)]:
                    for name, value in zip(metrics_names, results):
                        writer.add_scalar(f'{model_name}/{name}/{group}', value, epoch)

            aggregate_metric = metric_aggregator_fn(average_loss, all_results)
            if early_stopping.step(aggregate_metric):
                break

        early_stopping.load_best_state()

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
            f'fusion={self.fusion}, '
            f'low_rank={self.low_rank}, '
            f'attention_heads={self.attention_heads}, '
            f'num_experts={self.num_experts}, '
            f'dropout={self.dropout}, '
            f'noise_multiplier={self.noise_multiplier}, '
            f'pos_weight={self.pos_weight}, '
            f'degree_weight_factor={self.degree_weight_factor}, '
            f'feature_weight_factor={self.feature_weight_factor}, '
            f'fingerprint={calculate_model_fingerprint(self)})'
        )

import typing
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Optimizer
from hydra.utils import instantiate
from torch_geometric.utils import to_undirected
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric

from .twin import TwinGCN
from .social import SocialGCN
from .rating import RatingGCN
from sralt.target.model import TargetModel
from sralt.target.data import TargetData, TrainTargetDataset, EvalTargetDataset
from sralt.utils import timer, model_fingerprint, compute_recommendation_metrics, recommend_items
from sralt.utils import TensorBatchLoader, EarlyStopping


class DESIGN(TargetModel):
    """
    Historical SRALT DESIGN adaptation restored from the 2026-08-27 version.
    Retains the shared data/evaluation protocol and graph-cache lifecycle fixes.
    """

    def __init__(
        self, num_users: int, num_items: int, embed_dim: int,
        user_fusion_layers: list[int], item_fusion_layers: list[int],
        decay: float, distill_weight: float
    ):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Dimension of the embeddings.
        :param user_fusion_layers: Indices of layers to sum when computing the final embedding for users.
        :param item_fusion_layers: Indices of layers to sum when computing the final embedding for items.
        :param decay: L2 regularization coefficient.
        :param distill_weight: Weight for the distillation loss.
        """
        super().__init__(num_users, num_items)

        self.social = SocialGCN(num_users, num_items, embed_dim, user_fusion_layers)
        self.rating = RatingGCN(num_users, num_items, embed_dim, user_fusion_layers, item_fusion_layers)
        self.twin = TwinGCN(num_users, num_items, embed_dim, user_fusion_layers, item_fusion_layers)

        self.decay = decay
        self.distill_weight = distill_weight

    def forward(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor, target_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DESIGN model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :param target_edge_index: Edge index tensor of shape (2, num_target_edges) for which to compute scores.
        :return: Stack of score tensors from SocialGCN, RatingGCN, and TwinGCN of shape (num_target_edges, 3).
        """
        social_scores: torch.Tensor = self.social(social_edge_index, target_edge_index)
        rating_scores: torch.Tensor = self.rating(behavioral_edge_index, target_edge_index)
        twin_scores: torch.Tensor = self.twin(social_edge_index, behavioral_edge_index, target_edge_index)

        return torch.stack([social_scores, rating_scores, twin_scores], dim=1)

    def get_embedding(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings of TwinGCN model.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social connections.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing behavioral interactions.
        :return: Embedding tensor of shape (num_nodes, embed_dim).
        """
        return self.twin.get_embedding(social_edge_index, behavioral_edge_index)

    def calculate_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
        Compute loss using Bayesian Personalized Ranking (BPR) loss, L2 regularization, and distillation loss.
        Note that pos_scores[i] and neg_scores[i] correspond to the same user.

        :param pos_scores: Score tensor of shape (num_samples, 3).
        :param neg_scores: Score tensor of shape (num_samples, 3).
        :param nodes: Node indices tensor of shape (3 * num_samples,).
        :return: Computed loss tensor.
        """
        sigmoid = nn.Sigmoid()
        logsigmoid = nn.LogSigmoid()

        loss = torch.tensor(0.0, device=nodes.device)
        batch_size = pos_scores.size(0)

        # BPR loss
        diff_scores = pos_scores - neg_scores
        rank_loss = -typing.cast(torch.Tensor, logsigmoid(diff_scores)).mean()
        loss += rank_loss

        # L2 regularization
        if self.decay != 0.0:
            reg_loss = (
                self.social.embeds.weight[nodes].pow(2).sum() +
                self.rating.embeds.weight[nodes].pow(2).sum() +
                self.twin.embeds.weight[nodes].pow(2).sum()
            ) / batch_size

            loss += self.decay * reg_loss

        # Distillation loss
        if self.distill_weight != 0.0:
            pos_probs: torch.Tensor = sigmoid(pos_scores)
            neg_probs: torch.Tensor = sigmoid(neg_scores)
            distill_loss = torch.tensor(0.0, device=nodes.device)

            for teacher, student in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]:
                distill_loss += F.binary_cross_entropy(pos_probs[:, student], pos_probs[:, teacher].detach())
                distill_loss += F.binary_cross_entropy(neg_probs[:, student], neg_probs[:, teacher].detach())

            loss += self.distill_weight * distill_loss / 6.0

        return loss

    def _train_model(
        self, *, data: TargetData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int, num_epochs: int,
        optimizer: Optimizer, metrics: list[LinkPredMetric], early_stopping: EarlyStopping
    ):
        """
        Train the DESIGN model.

        :param data: The target dataset.
        :param writer: Optional TensorBoard writer for logging.
        :param train_batch_size: Batch size for training.
        :param val_batch_size: Batch size for validation.
        :param val_num_negatives: Number of negative samples per positive sample during validation.
        :param num_epochs: Maximum number of training epochs.
        :param optimizer: Optimizer for training.
        :param metrics: List of link prediction metrics to compute during validation.
        :param early_stopping: EarlyStopping helper for managing early stopping.
        """
        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)

        train_dataset = TrainTargetDataset(data)
        train_loader = TensorBatchLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalTargetDataset(data, split='val', num_negatives=val_num_negatives)
        val_loader = TensorBatchLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

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
                users: torch.Tensor
                optimizer.zero_grad()

                pos_edges = torch.stack([users, pos_items], dim=0)
                neg_edges = torch.stack([users, neg_items], dim=0)
                node_ids = torch.cat([users, pos_items, neg_items])

                pos_scores: torch.Tensor = self(data.social_edge_index, undirected_behavioral_edge_index, pos_edges)
                neg_scores: torch.Tensor = self(data.social_edge_index, undirected_behavioral_edge_index, neg_edges)

                loss = self.calculate_loss(pos_scores, neg_scores, node_ids)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * users.size(0)
                total_samples += users.size(0)

            average_loss = total_loss / total_samples
            if writer is not None:
                writer.add_scalar(f'{model_name}/Loss/train', average_loss, epoch)

            self.eval()
            metric_results = compute_recommendation_metrics(
                embeds=self.get_embedding(data.social_edge_index, undirected_behavioral_edge_index),
                metrics=metrics,
                eval_loader=val_loader
            )
            metric_str = ', '.join([f'{name}={value:.4f}' for name, value in zip(metric_names, metric_results)])
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
        optimizer: DictConfig, metrics: DictConfig, early_stopping: DictConfig,
        **kwargs
    ):
        """
        Train the DESIGN model.

        :param data: The target dataset.
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

        self._train_model(
            data=data,
            writer=writer,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            val_num_negatives=val_num_negatives,
            num_epochs=num_epochs,
            optimizer=instantiate(optimizer, params=self.parameters()),
            metrics=instantiate(metrics),
            early_stopping=instantiate(early_stopping, model=self)
        )

    @torch.no_grad()
    def test_model(
        self, *, data: TargetData, metrics: list[LinkPredMetric],
        test_batch_size: int, test_num_negatives: int, **kwargs
    ) -> list[float]:
        """
        Test the DESIGN model.

        :param data: The target dataset.
        :param metrics: A list of link prediction metrics to compute.
        :param test_batch_size: Batch size for testing.
        :param test_num_negatives: Number of negative samples per positive sample during testing.
        :return: A list of metric values.
        """
        if kwargs:
            logging.warning(f'Ignoring unexpected additional testing config parameters: {kwargs}')

        self.eval()

        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        test_dataset = EvalTargetDataset(data, split='test', num_negatives=test_num_negatives)
        test_loader = TensorBatchLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        metric_results = compute_recommendation_metrics(
            embeds=self.get_embedding(data.social_edge_index, undirected_behavioral_edge_index),
            metrics=metrics,
            eval_loader=test_loader
        )

        return metric_results

    @torch.no_grad()
    def recommend(self, *, data: TargetData, num_recommendations: int, batch_size: int) -> torch.Tensor:
        """
        Generate recommendations for the given users, avoiding items they have already interacted with.

        :param data: The target dataset.
        :param num_recommendations: The number of recommendations to generate per user.
        :param batch_size: The batch size for generating recommendations.
        :return: Edge index tensor of shape (2, num_users * num_recommendations) containing recommendations.
        """
        self.eval()

        undirected_behavioral_edge_index = to_undirected(data.behavioral_edge_index)
        embeds = self.get_embedding(data.social_edge_index, undirected_behavioral_edge_index)

        return recommend_items(
            embeds=embeds,
            num_users=self.num_users,
            num_recommendations=num_recommendations,
            mask_edge_index=data.behavioral_edge_index,
            batch_size=batch_size
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'num_users={self.num_users}, '
            f'num_items={self.num_items}, '
            f'embed_dim={self.twin.embeds.embedding_dim}, '
            f'user_fusion_layers={self.twin.user_fusion_layers}, '
            f'item_fusion_layers={self.twin.item_fusion_layers}, '
            f'decay={self.decay}, '
            f'distill_weight={self.distill_weight}, '
            f'fingerprint={model_fingerprint(self)})'
        )

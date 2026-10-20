import logging
import typing

import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from hydra.utils import instantiate
from torch_geometric.utils import to_undirected
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric

from .weight import AdaptiveWeight, l2_loss
from .dcl import DualContrastiveLearning
from .gap import GraphAdaptivePropagation
from sralt.target.model import TargetModel
from sralt.target.data import EvalTargetDataset, TargetData
from sralt.utils import timer, model_fingerprint, compute_recommendation_metrics, recommend_items
from sralt.utils import DataLoader, EarlyStopping


class EGCL(TargetModel):
    """
    Effective and Efficient Graph Contrastive Learning (EGCL).
    Paper: 'EGCL: An Effective and Efficient Graph Contrastive Learning Framework for Social Recommendation'.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        num_layers: int,
        embedding_reg: float,
        weight_reg: float,
        beta: float,
        temperature: float,
        intra_reg: float,
        inter_reg: float,
        contrastive_batch_size: int
    ):
        """
        :param num_users: Number of users in the dataset.
        :param num_items: Number of items in the dataset.
        :param embed_dim: Dimension of the node embeddings.
        :param num_layers: Number of graph convolution layers in the propagation module.
        :param embedding_reg: Half-squared-norm regularization for initial embeddings.
        :param weight_reg: Half-squared-norm regularization for weight networks and readouts.
        :param beta: Weight of the social supervised loss.
        :param temperature: Temperature parameter for the InfoNCE loss in dual contrastive learning.
        :param intra_reg: Regularization coefficient for intra-view contrastive loss.
        :param inter_reg: Regularization coefficient for inter-view contrastive loss.
        :param contrastive_batch_size: Anchor chunk size; all nodes remain in the denominator.
        """
        super().__init__(num_users, num_items)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.embedding_reg = embedding_reg
        self.weight_reg = weight_reg
        self.beta = beta
        self.contrastive_batch_size = contrastive_batch_size
        self.temperature = temperature
        self.intra_reg = intra_reg
        self.inter_reg = inter_reg

        self.gap = GraphAdaptivePropagation(num_users, num_items, embed_dim, num_layers)
        self.dcl = DualContrastiveLearning(temperature, intra_reg, inter_reg, contrastive_batch_size)
        self.behavioral_weight = AdaptiveWeight(embed_dim, offset=8.0)
        self.social_weight = AdaptiveWeight(embed_dim, offset=10.0)

    def get_embedding(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the node embeddings from the model.
        :param social_edge_index: Undirected edge index of shape (2, num_social_edges) for the user-user social graph.
        :param behavioral_edge_index: Undirected edge index of shape (2, num_behavioral_edges) for the user-item behavioral graph.
        :return: Node embeddings of shape (num_users + num_items, embed_dim).
        """
        behavioral_embeds, _ = self.gap(social_edge_index, behavioral_edge_index)
        return behavioral_embeds

    def forward(
        self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor,
        batch_size: int, *, weight_phase: bool = False
    ) -> torch.Tensor:
        """Full-graph supervised objective, with separate weight/embedding phases.

        ``batch_size`` partitions positive corrections without sampling or averaging.
        Weight networks remain differentiable w.r.t. embeddings in the embedding phase.
        """
        if weight_phase:
            with torch.no_grad():
                behavioral_embeds, social_embeds = self.gap(social_edge_index, behavioral_edge_index)
        else:
            behavioral_embeds, social_embeds = self.gap(social_edge_index, behavioral_edge_index)

        interactions = behavioral_edge_index[:, behavioral_edge_index[0] < self.num_users].clone()
        interactions[1] -= self.num_users
        behavioral_loss = self.behavioral_weight.square_loss(
            behavioral_embeds[:self.num_users], behavioral_embeds[self.num_users:],
            interactions, batch_size, include_constant=weight_phase
        )
        social_loss = self.social_weight.square_loss(
            social_embeds, social_embeds, social_edge_index, batch_size, include_constant=weight_phase
        )
        if weight_phase:
            behavioral_loss = behavioral_loss + self.weight_reg * l2_loss(self.behavioral_weight.parameters())
            social_loss = social_loss + self.weight_reg * l2_loss(self.social_weight.parameters())
            return behavioral_loss + self.beta * social_loss

        behavioral_loss = behavioral_loss + self.embedding_reg * l2_loss([
            self.gap.user_embeds.weight, self.gap.item_embeds.weight
        ]) + self.weight_reg * l2_loss([
            *self.gap.behavioral_user_readout.parameters(), *self.gap.behavioral_item_readout.parameters()
        ])
        social_loss = social_loss + self.weight_reg * l2_loss(self.gap.social_readout.parameters())
        return behavioral_loss + self.beta * social_loss + self.dcl(
            social_edge_index, behavioral_edge_index, behavioral_embeds, social_embeds
        )

    def _train_model(
        self, *, data: TargetData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int, num_epochs: int,
        optimizer: Optimizer, weight_optimizer: Optimizer, metrics: list[LinkPredMetric], early_stopping: EarlyStopping
    ):
        """
        Internal method to train the EGCL model.

        :param data: Target data object.
        :param writer: Optional TensorBoard writer.
        :param train_batch_size: Positive-edge chunk size for the exact full-graph loss.
        :param val_batch_size: Validation batch size.
        :param val_num_negatives: Number of negatives per validation sample.
        :param num_epochs: Number of training epochs.
        :param optimizer: Optimizer instance.
        :param weight_optimizer: Separate optimizer for the adaptive weight networks.
        :param metrics: Validation metric calculators.
        :param early_stopping: Early stopping controller.
        """
        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)

        val_dataset = EvalTargetDataset(data, split='val', num_negatives=val_num_negatives)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        model_name = self.__class__.__name__
        metric_names = [
            f"{typing.cast(str, metric.__class__.__name__).removeprefix('LinkPred')}@{metric.k}"
            for metric in metrics
        ]

        for epoch in range(1, num_epochs + 1):
            self.train()

            weight_optimizer.zero_grad(set_to_none=True)
            weight_loss = self(
                data.social_edge_index, undirected_behavioral_edge_index,
                train_batch_size, weight_phase=True
            )
            weight_loss.backward()
            weight_optimizer.step()
            weight_optimizer.zero_grad(set_to_none=True)

            optimizer.zero_grad(set_to_none=True)
            loss = self(data.social_edge_index, undirected_behavioral_edge_index, train_batch_size)
            torch.autograd.backward(loss, inputs=list(self.gap.parameters()))
            optimizer.step()
            average_loss = loss.item()
            if writer is not None:
                writer.add_scalar(f'{model_name}/Loss/train', average_loss, epoch)

            self.eval()
            metric_results = compute_recommendation_metrics(
                embeds=self.get_embedding(data.social_edge_index, undirected_behavioral_edge_index),
                metrics=metrics,
                eval_loader=val_loader,
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
        Train the EGCL model with the given training configuration.

        :param data: Target data object.
        :param writer: Optional TensorBoard writer.
        :param train_batch_size: Positive-edge chunk size for the exact full-graph loss.
        :param val_batch_size: Validation batch size.
        :param val_num_negatives: Number of negatives per validation sample.
        :param num_epochs: Number of training epochs.
        :param optimizer: Optimizer configuration for Hydra instantiation.
        :param metrics: Validation metrics configuration for Hydra instantiation.
        :param early_stopping: Early stopping configuration for Hydra instantiation.
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
            optimizer=instantiate(optimizer, params=self.gap.parameters()),
            weight_optimizer=instantiate(optimizer, params=[
                *self.behavioral_weight.parameters(), *self.social_weight.parameters()
            ]),
            metrics=instantiate(metrics),
            early_stopping=instantiate(early_stopping, model=self),
        )

    @torch.no_grad()
    def test_model(
        self, *, data: TargetData, metrics: list[LinkPredMetric],
        test_batch_size: int, test_num_negatives: int, **kwargs
    ) -> list[float]:
        """
        Test the EGCL model with the given testing configuration.

        :param data: Target data object.
        :param metrics: List of metric calculators for evaluation.
        :param test_batch_size: Testing batch size.
        :param test_num_negatives: Number of negatives per testing sample.
        """
        if kwargs:
            logging.warning(f'Ignoring unexpected additional testing config parameters: {kwargs}')

        self.eval()

        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        test_dataset = EvalTargetDataset(data, split='test', num_negatives=test_num_negatives)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        metric_results = compute_recommendation_metrics(
            embeds=self.get_embedding(data.social_edge_index, undirected_behavioral_edge_index),
            metrics=metrics,
            eval_loader=test_loader
        )

        return metric_results

    @torch.no_grad()
    def recommend(self, *, data: TargetData, num_recommendations: int, batch_size: int) -> torch.Tensor:
        """
        Generate top-N item recommendations for all users in the dataset.
        :param data: Target data object.
        :param num_recommendations: Number of items to recommend for each user.
        :param batch_size: Batch size for generating recommendations.
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
            f'embed_dim={self.embed_dim}, '
            f'num_layers={self.num_layers}, '
            f'embedding_reg={self.embedding_reg}, '
            f'weight_reg={self.weight_reg}, '
            f'beta={self.beta}, '
            f'contrastive_batch_size={self.contrastive_batch_size}, '
            f'temperature={self.temperature}, '
            f'intra_reg={self.intra_reg}, '
            f'inter_reg={self.inter_reg}, '
            f'fingerprint={model_fingerprint(self)})'
        )

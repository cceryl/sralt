import logging
import typing

import torch
from tqdm import tqdm
from omegaconf import DictConfig
from torch.optim import Optimizer
from hydra.utils import instantiate
from torch_geometric.utils import to_undirected
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn.models.lightgcn import BPRLoss
from torch_geometric.metrics.link_pred import LinkPredMetric

from .data import EGCLDataset
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
        decay: float,
        temperature: float,
        intra_reg: float,
        inter_reg: float
    ):
        """
        :param num_users: Number of users in the dataset.
        :param num_items: Number of items in the dataset.
        :param embed_dim: Dimension of the node embeddings.
        :param num_layers: Number of graph convolution layers in the propagation module.
        :param decay: Regularization coefficient for embeddings and model parameters.
        :param temperature: Temperature parameter for the InfoNCE loss in dual contrastive learning.
        :param intra_reg: Regularization coefficient for intra-view contrastive loss.
        :param inter_reg: Regularization coefficient for inter-view contrastive loss.
        """
        super().__init__(num_users, num_items)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.decay = decay
        self.temperature = temperature
        self.intra_reg = intra_reg
        self.inter_reg = inter_reg

        self.gap = GraphAdaptivePropagation(num_users, num_items, embed_dim, num_layers, decay)
        self.dcl = DualContrastiveLearning(temperature, intra_reg, inter_reg)

    def get_embedding(self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the node embeddings from the model.
        :param social_edge_index: Undirected edge index of shape (2, num_social_edges) for the user-user social graph.
        :param behavioral_edge_index: Undirected edge index of shape (2, num_behavioral_edges) for the user-item behavioral graph.
        :return: Node embeddings of shape (num_users + num_items, embed_dim).
        """
        behavioral_embeds, social_embeds = self.gap(social_edge_index, behavioral_edge_index)

        user_embeds = (behavioral_embeds[:self.num_users] + social_embeds) / 2
        item_embeds = behavioral_embeds[self.num_users:]

        return torch.cat([user_embeds, item_embeds], dim=0)

    def _compute_scores(self, embeds: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute the scores for the given edge index based on the node embeddings.
        :param embeds: Node embeddings of shape (num_users + num_items, embed_dim).
        :param edge_index: Edge index of shape (2, num_edges) for which to compute the scores.
        :return: Scores for each edge in the edge index of shape (num_edges,).
        """
        src, dst = edge_index
        scores = (embeds[src] * embeds[dst]).sum(dim=1)

        return scores

    def forward(
        self, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor,
        users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor, pos_peers: torch.Tensor, neg_peers: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute the BPR loss and the dual contrastive learning loss.
        :param social_edge_index: Undirected edge index of shape (2, num_social_edges) for the user-user social graph.
        :param behavioral_edge_index: Undirected edge index of shape (2, num_behavioral_edges) for the user-item behavioral graph.
        :param users: Tensor of shape (batch_size,) containing user indices in the batch.
        :param pos_items: Tensor of shape (batch_size,) containing positive item indices for the recommendation loss.
        :param neg_items: Tensor of shape (batch_size,) containing negative item indices for the recommendation loss.
        :param pos_peers: Tensor of shape (batch_size,) containing positive peer indices for the social loss.
        :param neg_peers: Tensor of shape (batch_size,) containing negative peer indices for the social loss.
        :return: Scalar tensor of the total loss.
        """
        behavioral_embeds, social_embeds = self.gap(social_edge_index, behavioral_edge_index)
        cl_loss = self.dcl(
            social_edge_index, behavioral_edge_index, behavioral_embeds, social_embeds,
            users=torch.cat([users, pos_peers, neg_peers], dim=0),
            items=torch.cat([pos_items, neg_items], dim=0)
        )

        behavioral_pos_scores = self._compute_scores(behavioral_embeds, torch.stack([users, pos_items], dim=0))
        behavioral_neg_scores = self._compute_scores(behavioral_embeds, torch.stack([users, neg_items], dim=0))
        behavioral_nodes = torch.cat([users, pos_items, neg_items])
        behavioral_loss = self.gap.compute_loss(behavioral_pos_scores, behavioral_neg_scores, behavioral_nodes)

        social_pos_scores = self._compute_scores(social_embeds, torch.stack([users, pos_peers], dim=0))
        social_neg_scores = self._compute_scores(social_embeds, torch.stack([users, neg_peers], dim=0))
        social_nodes = torch.cat([users, pos_peers, neg_peers])
        social_loss = self.gap.compute_loss(social_pos_scores, social_neg_scores, social_nodes)

        return behavioral_loss + social_loss + cl_loss

    def _train_model(
        self, *, data: TargetData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int, num_epochs: int,
        optimizer: Optimizer, metrics: list[LinkPredMetric], early_stopping: EarlyStopping
    ):
        """
        Internal method to train the EGCL model.

        :param data: Target data object.
        :param writer: Optional TensorBoard writer.
        :param train_batch_size: Training batch size.
        :param val_batch_size: Validation batch size.
        :param val_num_negatives: Number of negatives per validation sample.
        :param num_epochs: Number of training epochs.
        :param optimizer: Optimizer instance.
        :param metrics: Validation metric calculators.
        :param early_stopping: Early stopping controller.
        """
        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)

        train_dataset = EGCLDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalTargetDataset(data, split='val', num_negatives=val_num_negatives)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        model_name = self.__class__.__name__
        metric_names = [
            f"{typing.cast(str, metric.__class__.__name__).removeprefix('LinkPred')}@{metric.k}"
            for metric in metrics
        ]

        for epoch in range(1, num_epochs + 1):
            self.train()

            total_loss = 0.0
            total_samples = 0

            for users, pos_items, neg_items, pos_peers, neg_peers in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
                optimizer.zero_grad()

                users: torch.Tensor
                loss: torch.Tensor = self(
                    social_edge_index=data.social_edge_index,
                    behavioral_edge_index=undirected_behavioral_edge_index,
                    users=users,
                    pos_items=pos_items,
                    neg_items=neg_items,
                    pos_peers=pos_peers,
                    neg_peers=neg_peers
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * users.size(0)
                total_samples += users.size(0)

            average_loss = total_loss / total_samples if total_samples > 0 else float('nan')
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
        :param train_batch_size: Training batch size.
        :param val_batch_size: Validation batch size.
        :param val_num_negatives: Number of negatives per validation sample.
        :param num_epochs: Number of training epochs.
        :param optimizer: Optimizer configuration for Hydra instantiation.
        :param metrics: Validation metrics configuration for Hydra instantiation.
        :param early_stopping: Early stopping configuration for Hydra instantiation.
        """
        if kwargs:
            logging.warning(f'Ignoring unexpected additional training config parameters: {kwargs}')

        embed_params = [param for name, param in self.named_parameters() if 'embeds' in name]
        other_params = [param for name, param in self.named_parameters() if 'embeds' not in name]

        self._train_model(
            data=data,
            writer=writer,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            val_num_negatives=val_num_negatives,
            num_epochs=num_epochs,
            optimizer=instantiate(optimizer, _partial_=True)(
                params=[
                    {'params': embed_params, 'weight_decay': 0.0},
                    {'params': other_params, 'weight_decay': self.decay}
                ]
            ),
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
            f'decay={self.decay}, '
            f'temperature={self.temperature}, '
            f'intra_reg={self.intra_reg}, '
            f'inter_reg={self.inter_reg}, '
            f'fingerprint={model_fingerprint(self)})'
        )

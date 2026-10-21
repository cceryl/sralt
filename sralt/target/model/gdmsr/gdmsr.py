import typing
import logging

import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
from torch.optim import Optimizer
from hydra.utils import instantiate
from torch_geometric.utils import to_undirected
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric

from .denoiser import Denoiser
from .data import build_item_history, DenoiserDataset
from sralt.target import TargetData, EvalTargetDataset, TargetModel
from sralt.utils import timer, model_fingerprint, compute_recommendation_metrics, recommend_items
from sralt.utils import DataLoader, EarlyStopping


class GDMSR(TargetModel):
    """
    Graph Denoising Method for Social Recommendation (GDMSR).
    Paper: 'Robust Preference-Guided Denoising for Graph based Social Recommendation'.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        num_layers: int,
        sequence_length: int,
        curriculum_interval: int,
        alpha: float,
        beta: float,
        gamma: float,
        epsilon: int,
        r: float,
        decay: float,
        dim_feedforward: int = 2048
    ):
        """
        Initialize the GDMSR model.

        :param num_users: Total number of users.
        :param num_items: Total number of items.
        :param embed_dim: Embedding dimension for users and items.
        :param num_layers: Number of GCN layers.
        :param sequence_length: Maximum length of user interaction history.
        :param curriculum_interval: Number of epochs between curriculum updates.
        :param alpha: Weight for social relation prediction loss.
        :param beta: Weight for previous confidence scores in EMA update.
        :param gamma: Exponent in adaptive denoising ratio calculation.
        :param epsilon: Threshold for user degree in adaptive denoising ratio calculation.
        :param r: Scale factor in adaptive denoising ratio calculation.
        :param decay: Decay weight in BPR loss.
        :param dim_feedforward: Hidden dimension of the relation Transformer's FFN.
        """
        super().__init__(num_users, num_items)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.curriculum_interval = curriculum_interval
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.r = r
        self.decay = decay
        self.dim_feedforward = dim_feedforward

        self.denoiser = Denoiser(
            num_users=num_users,
            num_items=num_items,
            embed_dim=embed_dim,
            num_layers=num_layers,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            epsilon=epsilon,
            r=r,
            decay=decay,
            dim_feedforward=dim_feedforward
        )

        self.social_edge_mask: torch.Tensor
        self.register_buffer('social_edge_mask', torch.empty(0, dtype=torch.bool))

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, local_metadata: dict, strict: bool,
        missing_keys: list[str], unexpected_keys: list[str], error_msgs: list[str]
    ):
        social_edge_mask: torch.Tensor = state_dict[prefix + 'social_edge_mask']
        self.social_edge_mask = self.social_edge_mask.new_empty(social_edge_mask.size())
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def get_embedding(
        self, *, social_edge_index: torch.Tensor, behavioral_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get final embeddings used for evaluation and recommendation.

        :param social_edge_index: Undirected edge index tensor of shape (2, num_social_edges) representing social relations.
        :param behavioral_edge_index: Undirected edge index tensor of shape (2, num_behavioral_edges) representing user-item interactions.
        :return: Tensor of shape (num_nodes, (num_layers + 1) * embed_dim).
        """
        mask = self.social_edge_mask
        if mask.numel() == 0:
            mask = torch.ones_like(social_edge_index[0], dtype=torch.bool)

        return self.denoiser.get_embedding(
            social_edge_index=social_edge_index,
            behavioral_edge_index=behavioral_edge_index,
            social_edge_mask=mask
        )

    def _train_model(
        self,
        *,
        data: TargetData,
        writer: SummaryWriter | None,
        train_batch_size: int,
        val_batch_size: int,
        val_num_negatives: int,
        num_epochs: int,
        optimizer: Optimizer,
        metrics: list[LinkPredMetric],
        early_stopping: EarlyStopping,
    ):
        """
        Internal method to train the GDMSR model with curriculum learning.

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
        social_edge_index = data.social_edge_index
        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)

        train_dataset = DenoiserDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalTargetDataset(data, split='val', num_negatives=val_num_negatives)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        model_name = self.__class__.__name__
        metric_names = [
            f"{typing.cast(str, metric.__class__.__name__).removeprefix('LinkPred')}@{metric.k}"
            for metric in metrics
        ]

        self.social_edge_mask = torch.ones_like(social_edge_index[0], dtype=torch.bool)
        confidence_scores = torch.zeros_like(social_edge_index[0], dtype=torch.float)
        item_history, item_history_mask = build_item_history(
            behavioral_edge_index=data.behavioral_train_edge_index,
            num_users=self.num_users,
            num_items=self.num_items,
            sequence_length=self.sequence_length
        )

        for epoch in range(1, num_epochs + 1):
            self.train()

            train_dataset.social_edge_index = to_undirected(social_edge_index[:, self.social_edge_mask])

            total_loss = 0.0
            total_samples = 0

            for users, pos_items, neg_items, pos_peers, neg_peers in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
                optimizer.zero_grad()

                users: torch.Tensor
                loss: torch.Tensor = self.denoiser(
                    social_edge_index=social_edge_index,
                    behavioral_edge_index=undirected_behavioral_edge_index,
                    social_edge_mask=self.social_edge_mask,
                    item_history=item_history,
                    item_history_mask=item_history_mask,
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

            average_loss = total_loss / total_samples
            if writer is not None:
                writer.add_scalar(f'{model_name}/Loss/train', average_loss, epoch)

            self.eval()
            # The release initializes confidence after epoch 0 and updates after D, 2D, ... .
            if epoch == 1 or (epoch - 1) % self.curriculum_interval == 0:
                confidence_scores, self.social_edge_mask = self.denoiser.denoise_social_graph(
                    item_history=item_history,
                    item_history_mask=item_history_mask,
                    social_edge_index=social_edge_index,
                    confidence_scores=confidence_scores,
                    batch_size=val_batch_size,
                    social_edge_mask=self.social_edge_mask,
                    initialize=epoch == 1
                )
            metric_results = compute_recommendation_metrics(
                embeds=self.get_embedding(
                    social_edge_index=social_edge_index,
                    behavioral_edge_index=undirected_behavioral_edge_index
                ),
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
        self,
        *,
        data: TargetData,
        writer: SummaryWriter | None,
        train_batch_size: int,
        val_batch_size: int,
        val_num_negatives: int,
        num_epochs: int,
        optimizer: DictConfig,
        metrics: DictConfig,
        early_stopping: DictConfig,
        **kwargs
    ):
        """
        Train the GDMSR model.

        :param data: Target data object.
        :param writer: Optional TensorBoard writer.
        :param train_batch_size: Training batch size.
        :param val_batch_size: Validation batch size.
        :param val_num_negatives: Number of negatives per validation sample.
        :param num_epochs: Number of training epochs.
        :param optimizer: Optimizer instantiation config.
        :param metrics: Validation metric instantiation configs.
        :param early_stopping: EarlyStopping instantiation config.
        :param kwargs: Extra unused training args.
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
            early_stopping=instantiate(early_stopping, model=self),
        )

    @torch.no_grad()
    def test_model(
        self,
        *,
        data: TargetData,
        metrics: list[LinkPredMetric],
        test_batch_size: int,
        test_num_negatives: int,
        **kwargs
    ) -> list[float]:
        """
        Test the GDMSR model.

        :param data: Target data object.
        :param metrics: Metric calculators.
        :param test_batch_size: Test batch size.
        :param test_num_negatives: Number of negatives per test sample.
        :param kwargs: Extra unused testing args.
        :return: List of metric values.
        """
        if kwargs:
            logging.warning(f'Ignoring unexpected additional testing config parameters: {kwargs}')

        self.eval()

        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)
        test_dataset = EvalTargetDataset(data, split='test', num_negatives=test_num_negatives)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        metric_results = compute_recommendation_metrics(
            embeds=self.get_embedding(
                social_edge_index=data.social_edge_index,
                behavioral_edge_index=undirected_behavioral_edge_index,
            ),
            metrics=metrics,
            eval_loader=test_loader
        )

        return metric_results

    @torch.no_grad()
    def recommend(self, *, data: TargetData, num_recommendations: int, batch_size: int) -> torch.Tensor:
        """
        Generate top-k recommendations for all users.

        :param data: Target data object.
        :param num_recommendations: Number of items to recommend per user.
        :param batch_size: Batch size for recommendation generation.
        :return: Tensor of shape (2, num_users * num_recommendations) containing recommendation edges.
        """
        self.eval()

        undirected_behavioral_edge_index = to_undirected(data.behavioral_edge_index)
        embeds = self.get_embedding(
            social_edge_index=data.social_edge_index,
            behavioral_edge_index=undirected_behavioral_edge_index
        )

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
            f'dim_feedforward={self.dim_feedforward}, '
            f'sequence_length={self.sequence_length}, '
            f'curriculum_interval={self.curriculum_interval}, '
            f'alpha={self.alpha}, '
            f'beta={self.beta}, '
            f'gamma={self.gamma}, '
            f'epsilon={self.epsilon}, '
            f'r={self.r}, '
            f'decay={self.decay}, '
            f'fingerprint={model_fingerprint(self)})'
        )

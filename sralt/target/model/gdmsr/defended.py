"""SAL adapter with separate model state, preserving the baseline class."""

import logging
import typing
from itertools import cycle

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from .data import build_item_history, DenoiserDataset
from .gdmsr import GDMSR
from sralt.target.data import TargetData, EvalTargetDataset
from sralt.target.defense import SocialAdversarialLearning, SocialPairSampler
from sralt.utils import DataLoader, EarlyStopping, compute_recommendation_metrics, timer


class DefendedGDMSR(GDMSR):
    """Apply SAL to propagated user embeddings; zero weight uses baseline training."""

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
        defense: DictConfig,
        dim_feedforward: int = 2048
    ):
        super().__init__(
            num_users=num_users, num_items=num_items, embed_dim=embed_dim, num_layers=num_layers,
            sequence_length=sequence_length, curriculum_interval=curriculum_interval, alpha=alpha,
            beta=beta, gamma=gamma, epsilon=epsilon, r=r, decay=decay, dim_feedforward=dim_feedforward
        )
        self.defense: SocialAdversarialLearning | None = (
            instantiate(defense, embed_dim=(num_layers + 1) * embed_dim) if defense.loss_weight != 0 else None
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
        """Fit SAL on the recommendation readout while preserving base training."""

        if self.defense is None:
            return super()._train_model(
                data=data, writer=writer, train_batch_size=train_batch_size,
                val_batch_size=val_batch_size, val_num_negatives=val_num_negatives,
                num_epochs=num_epochs, optimizer=optimizer, metrics=metrics,
                early_stopping=early_stopping
            )
        defense = self.defense
        pair_sampler = SocialPairSampler(data.social_edge_index, self.num_users, defense.sample_ratio)
        surrogate_optimizer = defense.create_optimizer()

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

            if epoch > defense.surrogate_warmup_epochs:
                with torch.no_grad():
                    snapshot = self.get_embedding(
                        social_edge_index=data.social_edge_index,
                        behavioral_edge_index=undirected_behavioral_edge_index
                    )[:self.num_users]
                surrogate_loss = defense.fit_surrogate(snapshot, pair_sampler, surrogate_optimizer)
                logging.info(f'SAL epoch {epoch:04d}: surrogate_bce={surrogate_loss:.6f}')
                if writer is not None:
                    writer.add_scalar(f'{model_name}/SAL/surrogate_bce', surrogate_loss, epoch)
            defense_batches = cycle(pair_sampler.batches(len(train_loader)))

            train_dataset.social_edge_index = to_undirected(social_edge_index[:, self.social_edge_mask])

            total_recommendation_loss = 0.0
            total_defense_loss = 0.0
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

                recommendation_loss = loss
                positive_pairs, negative_pairs = next(defense_batches)
                user_embeds = self.get_embedding(
                    social_edge_index=social_edge_index,
                    behavioral_edge_index=undirected_behavioral_edge_index
                )[:self.num_users]
                defense_loss = defense.distribution_loss(user_embeds, positive_pairs, negative_pairs)
                loss = recommendation_loss + defense.loss_weight * defense_loss
                total_recommendation_loss += recommendation_loss.item() * users.size(0)
                total_defense_loss += defense_loss.item() * users.size(0)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * users.size(0)
                total_samples += users.size(0)

            average_loss = total_loss / total_samples
            recommendation_loss_value = total_recommendation_loss / total_samples
            defense_loss_value = total_defense_loss / total_samples
            logging.info(
                f'SAL epoch {epoch:04d}: recommendation_loss={recommendation_loss_value:.6f}, '
                f'defense_loss={defense_loss_value:.6f}, total_loss={average_loss:.6f}'
            )
            if writer is not None:
                writer.add_scalar(f'{model_name}/SAL/recommendation_loss', recommendation_loss_value, epoch)
                writer.add_scalar(f'{model_name}/SAL/defense_loss', defense_loss_value, epoch)
                writer.add_scalar(f'{model_name}/SAL/weighted_defense_loss', defense.loss_weight * defense_loss_value, epoch)
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
        self, *, data: TargetData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int, num_epochs: int,
        optimizer: DictConfig, metrics: DictConfig, early_stopping: DictConfig,
        **kwargs
    ):
        """
        Train GDMSR and its surrogate with separate optimizers.

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
            optimizer=instantiate(optimizer, params=(
                parameter for name, parameter in self.named_parameters() if not name.startswith('defense.')
            )),
            metrics=instantiate(metrics),
            early_stopping=instantiate(early_stopping, model=self)
        )

    def __repr__(self) -> str:
        description = super().__repr__()
        if self.defense is None:
            return description
        return description[:-1] + f', defense=SAL(loss_weight={self.defense.loss_weight}))'

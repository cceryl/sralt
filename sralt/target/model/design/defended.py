"""DESIGN with isolated defense state and training, preserving the original model class."""

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

from .design import DESIGN
from sralt.target.data import TargetData, TrainTargetDataset, EvalTargetDataset
from sralt.target.defense import SocialAdversarialLearning, SocialPairSampler
from sralt.utils import DataLoader, EarlyStopping, compute_recommendation_metrics, timer


class DefendedDESIGN(DESIGN):
    """Add optional SAL training to DESIGN without changing baseline pickle state.

    Recommendation equations, evaluation and deployment queries are inherited.
    A zero defense weight follows the original training path without initializing
    or sampling a surrogate, while retaining a separate model class/cache identity.
    """

    def __init__(
        self, num_users: int, num_items: int, embed_dim: int,
        user_fusion_layers: list[int], item_fusion_layers: list[int],
        distill_weight: float, defense: DictConfig
    ):
        super().__init__(num_users, num_items, embed_dim, user_fusion_layers, item_fusion_layers, distill_weight)
        self.defense: SocialAdversarialLearning | None = (
            instantiate(defense, embed_dim=embed_dim) if defense.loss_weight != 0 else None
        )

    def _train_model(
        self, *, data: TargetData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int, num_epochs: int,
        optimizer: Optimizer, metrics: list[LinkPredMetric], early_stopping: EarlyStopping
    ):
        """Alternate surrogate fitting and recommendation updates on training graphs."""
        if self.defense is None:
            return super()._train_model(
                data=data, writer=writer, train_batch_size=train_batch_size,
                val_batch_size=val_batch_size, val_num_negatives=val_num_negatives,
                num_epochs=num_epochs, optimizer=optimizer, metrics=metrics, early_stopping=early_stopping
            )
        defense = self.defense
        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)

        train_dataset = TrainTargetDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalTargetDataset(data, split='val', num_negatives=val_num_negatives)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        pair_sampler = SocialPairSampler(data.social_edge_index, self.num_users, defense.sample_ratio)
        surrogate_optimizer = defense.create_optimizer()

        model_name = self.__class__.__name__
        metric_names = [
            f'{typing.cast(str, metric.__class__.__name__).removeprefix("LinkPred")}@{metric.k}'
            for metric in metrics
        ]

        for epoch in range(1, num_epochs + 1):
            self.train()

            if epoch > defense.surrogate_warmup_epochs:
                with torch.no_grad():
                    snapshot = self.get_embedding(data.social_edge_index, undirected_behavioral_edge_index)[:self.num_users]
                surrogate_loss = defense.fit_surrogate(snapshot, pair_sampler, surrogate_optimizer)
                logging.info(f'SAL epoch {epoch:04d}: surrogate_bce={surrogate_loss:.6f}')
                if writer is not None:
                    writer.add_scalar(f'{model_name}/SAL/surrogate_bce', surrogate_loss, epoch)
            defense_batches = cycle(pair_sampler.batches(len(train_loader)))

            total_loss = 0.0
            total_recommendation_loss = 0.0
            total_defense_loss = 0.0
            total_samples = 0

            for users, pos_items, neg_items in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
                users: torch.Tensor
                optimizer.zero_grad()

                pos_edges = torch.stack([users, pos_items], dim=0)
                neg_edges = torch.stack([users, neg_items], dim=0)
                batch_edges = torch.cat([pos_edges, neg_edges], dim=1)
                labels = torch.cat([torch.ones_like(users), torch.zeros_like(users)])
                scores = self(data.social_edge_index, undirected_behavioral_edge_index, batch_edges)
                recommendation_loss = self.calculate_loss(scores, labels)
                positive_pairs, negative_pairs = next(defense_batches)
                # SAL matches the deployed Twin readout on the training graph.
                # Recommendation supervision above retains its batch edge mask.
                user_embeds = self.get_embedding(data.social_edge_index, undirected_behavioral_edge_index)[:self.num_users]
                defense_loss = defense.distribution_loss(user_embeds, positive_pairs, negative_pairs)
                loss = recommendation_loss + defense.loss_weight * defense_loss
                total_defense_loss += defense_loss.item() * users.size(0)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * users.size(0)
                total_recommendation_loss += recommendation_loss.item() * users.size(0)
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
        Train DESIGN and its surrogate with separate optimizers.

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

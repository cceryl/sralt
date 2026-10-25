"""SAL adapter with separate model state, preserving the baseline class."""

import logging
import typing

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric
from torch_geometric.utils import to_undirected

from .egcl import EGCL
from sralt.target.data import TargetData, EvalTargetDataset
from sralt.target.defense import SocialAdversarialLearning, SocialPairSampler
from sralt.utils import TensorBatchLoader, EarlyStopping, compute_recommendation_metrics


class DefendedEGCL(EGCL):
    """Apply SAL to propagated user embeddings; zero weight uses baseline training."""

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
        contrastive_batch_size: int,
        defense: DictConfig
    ):
        super().__init__(
            num_users=num_users, num_items=num_items, embed_dim=embed_dim, num_layers=num_layers,
            embedding_reg=embedding_reg, weight_reg=weight_reg, beta=beta, temperature=temperature,
            intra_reg=intra_reg, inter_reg=inter_reg, contrastive_batch_size=contrastive_batch_size
        )
        self.defense: SocialAdversarialLearning | None = (
            instantiate(defense, embed_dim=embed_dim) if defense.loss_weight != 0 else None
        )

    def _train_model(
        self, *, data: TargetData, writer: SummaryWriter | None,
        train_batch_size: int, val_batch_size: int, val_num_negatives: int, num_epochs: int,
        optimizer: Optimizer, weight_optimizer: Optimizer, metrics: list[LinkPredMetric], early_stopping: EarlyStopping
    ):
        """Fit SAL on the recommendation readout while preserving base training."""

        if self.defense is None:
            return super()._train_model(
                data=data, writer=writer, train_batch_size=train_batch_size,
                val_batch_size=val_batch_size, val_num_negatives=val_num_negatives,
                num_epochs=num_epochs, optimizer=optimizer, metrics=metrics,
                early_stopping=early_stopping, weight_optimizer=weight_optimizer
            )
        defense = self.defense
        pair_sampler = SocialPairSampler(data.social_edge_index, self.num_users, defense.sample_ratio)
        surrogate_optimizer = defense.create_optimizer()

        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)

        val_dataset = EvalTargetDataset(data, split='val', num_negatives=val_num_negatives)
        val_loader = TensorBatchLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        model_name = self.__class__.__name__
        metric_names = [
            f"{typing.cast(str, metric.__class__.__name__).removeprefix('LinkPred')}@{metric.k}"
            for metric in metrics
        ]

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
            recommendation_loss = loss
            # One full-graph embedding update per epoch: use the full sampled pairs.
            positive_pairs, negative_pairs = pair_sampler.sample()
            user_embeds = self.get_embedding(data.social_edge_index, undirected_behavioral_edge_index)[:self.num_users]
            defense_loss = defense.distribution_loss(user_embeds, positive_pairs, negative_pairs)
            loss = recommendation_loss + defense.loss_weight * defense_loss
            torch.autograd.backward(loss, inputs=list(self.gap.parameters()))
            optimizer.step()
            average_loss = loss.item()
            recommendation_loss_value = recommendation_loss.item()
            defense_loss_value = defense_loss.item()
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

    def __repr__(self) -> str:
        description = super().__repr__()
        if self.defense is None:
            return description
        return description[:-1] + f', defense=SAL(loss_weight={self.defense.loss_weight}))'

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
from sralt.utils import TensorBatchLoader, EarlyStopping, compute_recommendation_metrics, timer


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
        diagnostics = defense.create_diagnostics(self, pair_sampler)
        fit_sampler = diagnostics.fit_sampler if diagnostics else pair_sampler

        social_edge_index = data.social_edge_index
        undirected_behavioral_edge_index = to_undirected(data.behavioral_train_edge_index)

        train_dataset = DenoiserDataset(data)
        train_loader = TensorBatchLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataset = EvalTargetDataset(data, split='val', num_negatives=val_num_negatives)
        val_loader = TensorBatchLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

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

        def diagnostic_embedding():
            return self.get_embedding(social_edge_index=data.social_edge_index, behavioral_edge_index=undirected_behavioral_edge_index)[:self.num_users]

        refresh_schedule = defense.refresh_schedule(len(train_loader))
        logging.info('SAL refresh schedule: rec_steps=%d, fit_epochs_per_epoch=%d, schedule=%s',
                     len(train_loader), defense.surrogate_epochs, refresh_schedule)

        if diagnostics:
            diagnostics.emit(kind='refresh_schedule', recommendation_steps=len(train_loader),
                             fit_epochs_per_epoch=defense.surrogate_epochs,
                             warmup_epochs=defense.surrogate_warmup_epochs,
                             schedule=refresh_schedule)

        def fit_stage(epoch, step, sampled_pairs):
            with torch.no_grad():
                snapshot = diagnostic_embedding()
            rounds = refresh_schedule[step]
            value = defense.fit_surrogate(
                snapshot, fit_sampler, surrogate_optimizer,
                sampled_pairs=sampled_pairs, epochs=rounds
            )
            pair_count = fit_sampler.num_samples
            updates = rounds * ((2 * pair_count + defense.surrogate_batch_size - 1)
                                // defense.surrogate_batch_size)
            logging.info('SAL refresh epoch %04d: surrogate_bce=%.6f, refresh_step=%d, fit_epochs=%d, optimizer_steps=%d',
                         epoch, value, step, rounds, updates)
            if diagnostics:
                diagnostics.emit(kind='refresh', epoch=epoch, batch=step, fit_epochs=rounds,
                                 pairs_per_class=pair_count, optimizer_steps=updates,
                                 online_bce=value)
            return value

        for epoch in range(1, num_epochs + 1):
            self.train()
            if diagnostics:
                diagnostics.observe(epoch, "before_fit", diagnostic_embedding)

            fit_pairs = None
            epoch_surrogate_loss = 0.0
            if epoch > defense.surrogate_warmup_epochs:
                # One pair sample per epoch, reused across refreshes. Same total
                # passes and optimizer-step budget as the once-per-epoch control.
                if defense.surrogate_refreshes > 1:
                    fit_pairs = fit_sampler.sample()
                surrogate_loss = fit_stage(epoch, 0, fit_pairs)
                epoch_surrogate_loss += surrogate_loss * refresh_schedule[0]

            if diagnostics:
                diagnostics.observe(epoch, "after_fit", diagnostic_embedding)
            defense.surrogate.train(not defense.frozen_eval)
            defense_batches = cycle(pair_sampler.batches(len(train_loader)))

            train_dataset.social_edge_index = to_undirected(social_edge_index[:, self.social_edge_mask])

            total_recommendation_loss = 0.0
            total_defense_loss = 0.0
            total_loss = 0.0
            total_samples = 0

            for batch_index, (users, pos_items, neg_items, pos_peers, neg_peers) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')):
                if (epoch > defense.surrogate_warmup_epochs and batch_index > 0
                        and batch_index in refresh_schedule):
                    if diagnostics:
                        diagnostics.observe(epoch, f'before_refresh_{batch_index}', diagnostic_embedding)
                    epoch_surrogate_loss += (fit_stage(epoch, batch_index, fit_pairs)
                                             * refresh_schedule[batch_index])
                    if diagnostics:
                        diagnostics.observe(epoch, f'after_refresh_{batch_index}', diagnostic_embedding)
                    defense.surrogate.train(not defense.frozen_eval)
                optimizer.zero_grad()

                users: torch.Tensor
                loss, components = self.denoiser(
                    social_edge_index=social_edge_index,
                    behavioral_edge_index=undirected_behavioral_edge_index,
                    social_edge_mask=self.social_edge_mask,
                    item_history=item_history,
                    item_history_mask=item_history_mask,
                    users=users,
                    pos_items=pos_items,
                    neg_items=neg_items,
                    pos_peers=pos_peers,
                    neg_peers=neg_peers, return_components=True
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
                if diagnostics:
                    diagnostics.gradients(epoch, batch_index, components, defense_loss,
                                          {"embedding": [self.denoiser.embeds.weight], "gcn": self.denoiser.gcn.parameters()})
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * users.size(0)
                total_samples += users.size(0)

            if epoch > defense.surrogate_warmup_epochs:
                surrogate_loss = epoch_surrogate_loss / defense.surrogate_epochs
                logging.info('SAL epoch %04d: surrogate_bce=%.6f', epoch, surrogate_loss)
                if writer is not None:
                    writer.add_scalar(f'{model_name}/SAL/surrogate_bce', surrogate_loss, epoch)
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

            if diagnostics:
                diagnostics.observe(epoch, "after_recommendation", diagnostic_embedding)
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
                if diagnostics:
                    diagnostics.observe(epoch, "after_mask", diagnostic_embedding, force=True)
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
        defense.log_checkpoint(early_stopping.best_epoch)
        if diagnostics:
            diagnostics.observe(early_stopping.best_epoch, "checkpoint", diagnostic_embedding, force=True)
            diagnostics.refit(early_stopping.best_epoch, diagnostic_embedding)
            deployment_graph = to_undirected(data.behavioral_edge_index)
            diagnostics.observe(
                early_stopping.best_epoch, "checkpoint_deployment",
                lambda: self.get_embedding(
                    social_edge_index=data.social_edge_index,
                    behavioral_edge_index=deployment_graph
                )[:self.num_users], force=True
            )

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

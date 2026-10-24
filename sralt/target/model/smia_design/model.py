"""SocialPP/DESIGN22 target, isolated from the DESIGN supplementary model.

Source snapshot and intentional protocol adaptations are documented in
docs/on-going/smia-gap/socialpp-design.md. Item indices use SRALT's global IDs.
"""

import logging
from itertools import cycle

import torch
from torch import nn
from torch.nn import functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from sralt.target.model import TargetModel
from sralt.target.data import TargetData, TrainTargetDataset, EvalTargetDataset
from sralt.target.defense import SocialAdversarialLearning, SocialPairSampler
from sralt.utils import (
    DataLoader, compute_recommendation_metrics, recommend_items, model_fingerprint, timer,
)


class SMIADESIGN(TargetModel):
    """Three independent BPR branches with the SMIA authors' mean GCN readout."""

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 64,
                 hop: int = 2, decay: float = .001, kl_reg: float = 0.):
        super().__init__(num_users, num_items)
        if hop < 0:
            raise ValueError('hop must be nonnegative.')
        self.embed_dim, self.hop, self.decay, self.kl_reg = embed_dim, hop, decay, kl_reg
        # Preserve both the author's table names and initialization/RNG order.
        self.user_embs = nn.Embedding(num_users, embed_dim)
        self.item_embs = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embs.weight)
        nn.init.xavier_uniform_(self.item_embs.weight)
        self.user1_embs = nn.Embedding(num_users, embed_dim)
        self.item1_embs = nn.Embedding(num_items, embed_dim)
        self.user2_embs = nn.Embedding(num_users, embed_dim)
        self.item2_embs = nn.Embedding(num_items, embed_dim)
        for table in (self.user1_embs, self.item1_embs, self.user2_embs, self.item2_embs):
            nn.init.xavier_uniform_(table.weight)
        self._graphs = {}

    def _apply(self, fn, recurse=True):
        self._graphs.clear()
        return super()._apply(fn, recurse=recurse)

    def __getstate__(self):
        state = super().__getstate__().copy()
        state['_graphs'] = {}
        return state

    def _graph(self, name, edges, num_nodes):
        """Sparse row-normalization, no self-loops or symmetric normalization.

        SocialPP sets S[u,v]=1, then multiplies S @ X. Isolated rows remain zero.
        Only adjacency is cached; propagated embeddings always retain gradients.
        """
        cached = self._graphs.get(name)
        if cached is None or cached[0] is not edges:
            adjacency = torch.sparse_coo_tensor(
                edges, torch.ones(edges.size(1), device=edges.device, dtype=self.user_embs.weight.dtype),
                (num_nodes, num_nodes),
            ).coalesce()
            indices = adjacency.indices()
            # The author's matrices are binary, including duplicate input edges.
            degree = torch.bincount(indices[0], minlength=num_nodes).to(adjacency.dtype)
            values = degree[indices[0]].reciprocal()
            adjacency = torch.sparse_coo_tensor(indices, values, adjacency.shape).coalesce()
            self._graphs[name] = (edges, adjacency)
        return self._graphs[name][1]

    def _propagate(self, embeds, graph):
        layers = [embeds]
        for _ in range(self.hop):
            embeds = torch.sparse.mm(graph, embeds)
            layers.append(embeds)
        return torch.stack(layers, dim=1).mean(dim=1)

    def get_branch_embeddings(self, social_edge_index, behavioral_edge_index):
        """Return Twin, Social, Rating matrices, each [users+items, d]."""
        social = self._graph('social', social_edge_index, self.num_users)
        rating = self._graph('rating', behavioral_edge_index, self.num_users + self.num_items)
        twin_rating = self._propagate(torch.cat([self.user_embs.weight, self.item_embs.weight]), rating)
        twin_social = self._propagate(self.user_embs.weight, social)
        twin = torch.cat([.5 * twin_social + .5 * twin_rating[:self.num_users],
                          twin_rating[self.num_users:]])
        social_embeds = torch.cat([self._propagate(self.user1_embs.weight, social), self.item1_embs.weight])
        rating_embeds = self._propagate(torch.cat([self.user2_embs.weight, self.item2_embs.weight]), rating)
        return twin, social_embeds, rating_embeds

    def get_embedding(self, social_edge_index, behavioral_edge_index):
        """Twin representation used for SAL, recommendation validation and queries."""
        social = self._graph('social', social_edge_index, self.num_users)
        rating = self._graph('rating', behavioral_edge_index, self.num_users + self.num_items)
        ui = self._propagate(torch.cat([self.user_embs.weight, self.item_embs.weight]), rating)
        return torch.cat([.5 * self._propagate(self.user_embs.weight, social) + .5 * ui[:self.num_users],
                          ui[self.num_users:]])

    def forward(self, social_edge_index, behavioral_edge_index, target_edge_index):
        users, items = target_edge_index
        return torch.stack([
            (embeds[users] * embeds[items]).sum(-1)
            for embeds in self.get_branch_embeddings(social_edge_index, behavioral_edge_index)
        ], dim=1)

    def calculate_loss(self, users, pos_items, neg_items, branches):
        """Author BPR + batch raw-table L2 + six non-detached distillation terms.

        BPR uses dot sums; distillation uses dot means (division by d). L2 is
        .5 * sum(mean(table_batch**2)), including repeated user/item occurrences.
        """
        scores = []
        loss = self.user_embs.weight.new_zeros(())
        for embeds, user_table, item_table in zip(
            branches, (self.user_embs, self.user1_embs, self.user2_embs),
            (self.item_embs, self.item1_embs, self.item2_embs), strict=True,
        ):
            user = embeds[users]
            pos = (user * embeds[pos_items]).sum(-1)
            neg = (user * embeds[neg_items]).sum(-1)
            loss = loss - F.logsigmoid(pos - neg).mean()
            loss = loss + self.decay * .5 * (
                user_table(users).square().mean() + item_table(pos_items - self.num_users).square().mean()
                + item_table(neg_items - self.num_users).square().mean()
            )
            scores.append(torch.stack([pos, neg], dim=1) / self.embed_dim)
        if self.kl_reg != 0:
            for student, teacher in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
                # Stable equivalent of -p_teacher log(p_student) - (1-p_teacher) log(1-p_student).
                # Unlike the supplementary DESIGN, the teacher receives gradients.
                loss = loss + self.kl_reg * F.binary_cross_entropy_with_logits(
                    scores[student], scores[teacher].sigmoid())
        return loss

    @timer('Training model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_model(self, *, data: TargetData, writer, train_batch_size: int,
                    val_batch_size: int, val_num_negatives: int, num_epochs: int,
                    optimizer: DictConfig, metrics: DictConfig, early_stopping: DictConfig, **kwargs):
        if kwargs:
            raise TypeError(f'Unexpected training settings: {kwargs}')
        optimizer = instantiate(optimizer, params=(
            p for name, p in self.named_parameters() if not name.startswith('defense.')
        ))
        metrics = instantiate(metrics)
        stopper = instantiate(early_stopping, model=self)
        behavior = to_undirected(data.behavioral_train_edge_index)
        train_loader = DataLoader(TrainTargetDataset(data), batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(EvalTargetDataset(data, split='val', num_negatives=val_num_negatives),
                                batch_size=val_batch_size, shuffle=False)
        defense = getattr(self, 'defense', None)
        sampler = (SocialPairSampler(data.social_edge_index, self.num_users, defense.sample_ratio)
                   if defense is not None else None)
        surrogate_optimizer = defense.create_optimizer() if defense is not None else None
        self.training_history = []
        for epoch in range(1, num_epochs + 1):
            self.train()
            surrogate_loss = None
            if defense is not None:
                if epoch > defense.surrogate_warmup_epochs:
                    with torch.no_grad():
                        snapshot = self.get_embedding(data.social_edge_index, behavior)[:self.num_users]
                    surrogate_loss = defense.fit_surrogate(snapshot, sampler, surrogate_optimizer)
                defense_batches = cycle(sampler.batches(len(train_loader)))
            totals = dict(loss=0., recommendation_loss=0., defense_loss=0.)
            count = 0
            for users, pos, neg in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
                optimizer.zero_grad(set_to_none=True)
                branches = self.get_branch_embeddings(data.social_edge_index, behavior)
                rec_loss = self.calculate_loss(users, pos, neg, branches)
                defense_loss = rec_loss.new_zeros(())
                if defense is not None:
                    positives, negatives = next(defense_batches)
                    defense_loss = defense.distribution_loss(branches[0][:self.num_users], positives, negatives)
                loss = rec_loss + (defense.loss_weight * defense_loss if defense is not None else 0)
                loss.backward()
                optimizer.step()
                batch_size = users.numel()
                for key, value in zip(totals, (loss, rec_loss, defense_loss), strict=True):
                    totals[key] += value.item() * batch_size
                count += batch_size
            totals = {key: value / count for key, value in totals.items()}
            self.eval()
            # Only training interactions enter propagation; validation has no test-label dependency.
            values = compute_recommendation_metrics(
                embeds=self.get_embedding(data.social_edge_index, behavior), metrics=metrics, eval_loader=val_loader,
            )
            names = [f'{type(m).__name__.removeprefix("LinkPred")}@{m.k}' for m in metrics]
            logging.info('Epoch %04d: %s, val=%s, surrogate_bce=%s', epoch, totals,
                         dict(zip(names, values, strict=True)), surrogate_loss)
            record = dict(epoch=epoch, **totals, surrogate_bce=surrogate_loss, validation=values)
            self.training_history.append(record)
            if writer is not None:
                for name, value in totals.items():
                    writer.add_scalar(f'{type(self).__name__}/{name}/train', value, epoch)
                for name, value in zip(names, values, strict=True):
                    writer.add_scalar(f'{type(self).__name__}/{name}/val', value, epoch)
                if surrogate_loss is not None:
                    writer.add_scalar(f'{type(self).__name__}/surrogate_bce/train', surrogate_loss, epoch)
            if stopper.step(sum(values)):
                break
        stopper.load_best_state()
        self.training_summary = dict(
            best_epoch=stopper.best_epoch, stop_epoch=stopper.epoch, best_metric=stopper.best_metric,
            selection='validation-only', metrics=names,
            best_epoch_has_fitted_surrogate=(defense is not None and stopper.best_epoch > defense.surrogate_warmup_epochs),
        )

    @torch.no_grad()
    def test_model(self, *, data: TargetData, metrics, test_batch_size: int, test_num_negatives: int, **kwargs):
        if kwargs:
            raise TypeError(f'Unexpected test settings: {kwargs}')
        self.eval()
        loader = DataLoader(EvalTargetDataset(data, split='test', num_negatives=test_num_negatives),
                            batch_size=test_batch_size, shuffle=False)
        return compute_recommendation_metrics(
            embeds=self.get_embedding(data.social_edge_index, to_undirected(data.behavioral_train_edge_index)),
            metrics=metrics, eval_loader=loader,
        )

    @torch.no_grad()
    def recommend(self, *, data: TargetData, num_recommendations: int, batch_size: int):
        self.eval()
        # Retain SRALT's established full-history deployment and masking protocol.
        history = data.behavioral_edge_index
        return recommend_items(
            embeds=self.get_embedding(data.social_edge_index, to_undirected(history)),
            num_users=self.num_users, num_recommendations=num_recommendations,
            mask_edge_index=history, batch_size=batch_size,
        )

    def __repr__(self):
        defense = getattr(self, 'defense', None)
        return (f'{type(self).__name__}(num_users={self.num_users}, num_items={self.num_items}, '
                f'embed_dim={self.embed_dim}, hop={self.hop}, decay={self.decay}, kl_reg={self.kl_reg}, '
                f'sal_weight={defense.loss_weight if defense is not None else 0}, '
                f'fingerprint={model_fingerprint(self)})')


class DefendedSMIADESIGN(SMIADESIGN):
    """SocialPP target plus the existing SAL equations and protocol-safe sampler."""

    def __init__(self, num_users: int, num_items: int, defense: DictConfig,
                 embed_dim: int = 64, hop: int = 2, decay: float = .001, kl_reg: float = 0.):
        super().__init__(num_users, num_items, embed_dim, hop, decay, kl_reg)
        self.defense: SocialAdversarialLearning | None = (
            instantiate(defense, embed_dim=embed_dim) if defense.loss_weight != 0 else None
        )

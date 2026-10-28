"""Frozen item RVQ prefixes shared by social propagation and relation matching."""

import logging
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch import nn

from .base import ShadowModel, load_shadow_model
from .item_rvq import ItemResidualQuantizer, load_item_rvq_model
from .prefix import prefix_dictionary, user_prefix_sets, prefix_frequency, prefix_mass, semantic_links
from .tail_proxy import GraphBranch, TailProxyShadow
from sralt.shadow.data import TrainShadowDataset, EvalShadowDataset
from sralt.utils import TensorBatchLoader, build_normalized_adj, compute_recommendation_metrics, model_fingerprint, set_seed


class ItemPrefixBehavior(TailProxyShadow):
    """Separately cached continuous behavior encoder shared by all B ablations."""

    def __init__(self, num_users, num_items, embed_dim=64, num_layers=2, decay=.01):
        ShadowModel.__init__(self, num_users, num_items)
        self.embed_dim, self.num_layers, self.decay = embed_dim, num_layers, decay
        self.idf_weighting = True
        self.behavioral_branch = GraphBranch(num_users + num_items, embed_dim, num_layers)

    def behavior(self, adj, edges, idf):
        embeds = self.behavioral_branch(adj)
        mean, variance = self.aggregate(embeds, edges, idf)
        return (embeds, embeds[:self.num_users], mean, variance,
                edges.new_empty((self.num_users, 0)), embeds.new_zeros(()))

    def train_model(self, *, data, writer, train_batch_size, val_batch_size, val_num_negatives,
                    num_epochs, optimizer, metrics, early_stopping):
        idf = self.recommendation_idf(data.behavioral_edge_index)
        edges = data.behavioral_train_edge_index
        train_loader, val_loader = recommendation_loaders(data, train_batch_size, val_batch_size, val_num_negatives)
        self._train_branch('behavioral', self.behavior_adj(edges, idf), edges, idf,
                           train_loader, val_loader, num_epochs, optimizer, instantiate(metrics), early_stopping, writer)

    @torch.no_grad()
    def test_model(self, *, data, metrics, test_batch_size, test_num_negatives):
        idf = self.recommendation_idf(data.behavioral_edge_index)
        embeds = self.behavioral_branch(self.behavior_adj(data.behavioral_train_edge_index, idf))
        loader = TensorBatchLoader(EvalShadowDataset(data, split='test', num_negatives=test_num_negatives),
                                   batch_size=test_batch_size, shuffle=False)
        return compute_recommendation_metrics(embeds, metrics, loader)

    @torch.no_grad()
    def get_user_features(self, data):
        idf = self.recommendation_idf(data.behavioral_edge_index)
        return self.behavioral_branch(self.behavior_adj(data.behavioral_edge_index, idf))[:self.num_users]

    def __repr__(self):
        return (f'ItemPrefixBehavior(num_users={self.num_users}, num_items={self.num_items}, '
                f'embed_dim={self.embed_dim}, num_layers={self.num_layers}, decay={self.decay}, '
                f'fingerprint={model_fingerprint(self)})')


def recommendation_loaders(data, train_batch_size, val_batch_size, val_num_negatives):
    return (TensorBatchLoader(TrainShadowDataset(data), batch_size=train_batch_size, shuffle=True),
            TensorBatchLoader(EvalShadowDataset(data, split='val', num_negatives=val_num_negatives),
                              batch_size=val_batch_size, shuffle=False))


class PrefixGraphBranch(GraphBranch):
    """Learn user/item embeddings plus a shared projection of fixed proxies."""

    def __init__(self, num_nodes, embed_dim, num_layers):
        super().__init__(num_nodes, embed_dim, num_layers)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.projection.weight)
        self.register_buffer('prefix_reconstruction', torch.empty(0, embed_dim))

    def forward(self, adj):
        x = torch.cat([self.embedding.weight, self.projection(self.prefix_reconstruction)])
        result = x
        for _ in range(self.num_layers):
            x = torch.sparse.mm(adj, x)
            result = result + x
        return result / (self.num_layers + 1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        value = state_dict.get(prefix + 'prefix_reconstruction')
        if value is not None:
            self.prefix_reconstruction = self.prefix_reconstruction.new_empty(value.shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class ItemPrefixShadow(ItemPrefixBehavior):
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=2, decay=.01,
                 num_quantizers=8, codebook_size=256, levels=(8,), rvq_max_iter=50,
                 semantic_augmentation=True, frequency_weighting=True, proxy_kind='rvq'):
        super().__init__(num_users, num_items, embed_dim, num_layers, decay)
        if not levels or sorted(set(levels)) != list(levels) or min(levels) < 1 or max(levels) > num_quantizers:
            raise ValueError('Prefix levels must be distinct, increasing and within the RVQ depth.')
        if proxy_kind not in ('rvq', 'identity'):
            raise ValueError('proxy_kind must be rvq or the explicit identity control.')
        self.num_quantizers, self.codebook_size = num_quantizers, codebook_size
        self.levels, self.rvq_max_iter = tuple(levels), rvq_max_iter
        self.semantic_augmentation, self.frequency_weighting = semantic_augmentation, frequency_weighting
        self.proxy_kind = proxy_kind
        # Training history is metadata, not required to use a strictly restored
        # state_dict. All inference inputs live in parameters/persistent buffers.
        self.quantization_summary = {}
        self.behavioral_branch.requires_grad_(False)
        self.social_branch = PrefixGraphBranch(num_users + num_items, embed_dim, num_layers)
        self.rvq = ItemResidualQuantizer(torch.empty(0, 0, embed_dim, dtype=torch.float64))
        for name, shape in dict(queried_item_ids=(0,), item_codes=(0, num_quantizers),
                                item_prefix_ids=(0, len(levels)), prefix_paths=(0, num_quantizers),
                                prefix_levels=(0,), prefix_user_frequency=(0,)).items():
            self.register_buffer(name, torch.empty(shape, dtype=torch.long))
        self.register_buffer('prefix_information', torch.empty(0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name, buffer in self._buffers.items():
            value = state_dict.get(prefix + name)
            if value is not None:
                setattr(self, name, buffer.new_empty(value.shape))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def prefix_sets(self, edges, k):
        return user_prefix_sets(edges, self.queried_item_ids, self.item_prefix_ids,
                                self.num_users, len(self.prefix_levels), k)

    def prefix_adj(self, public_edges, prefix_ids):
        offset = self.num_users + self.num_items
        if self.semantic_augmentation:
            semantic, weight = semantic_links(prefix_ids, self.prefix_information, offset)
        else:
            semantic = public_edges.new_empty((2, 0))
            weight = self.prefix_information.new_empty(0)
        edges = torch.cat([public_edges, semantic], dim=1)
        weights = torch.cat([weight.new_ones(public_edges.size(1)), weight])
        return build_normalized_adj(edges, offset + len(self.prefix_levels), weights)

    @torch.no_grad()
    def prepare_prefixes(self, data, seed):
        started = time.monotonic()
        device = data.device
        full_edges = data.behavioral_edge_index.cpu()
        train_edges = data.behavioral_train_edge_index.cpu()
        idf = self.recommendation_idf(full_edges)
        # A canonical CPU propagation keeps the offline fit/cache input identical
        # across ablations even when GPU sparse propagation is non-deterministic.
        self.behavioral_branch.cpu()
        embeds = self.behavioral_branch(self.behavior_adj(train_edges, idf))[self.num_users:]
        queried_ids = (full_edges[1] - self.num_users).unique(sorted=True)
        train_ids = (train_edges[1] - self.num_users).unique(sorted=True)
        self.rvq = load_item_rvq_model(
            embeddings=embeds[train_ids].cpu(), item_ids=train_ids.cpu(),
            source=dict(behavior=model_fingerprint(self.behavioral_branch), query=repr(data), view='train'),
            seed=seed + 17027, num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size, max_iter=self.rvq_max_iter)
        # Offline CPU float64 encoding; no RVQ work in graph/Attack minibatches.
        item_vectors = embeds[queried_ids]
        self.item_codes, errors = self.rvq.encode(item_vectors)
        dictionary = prefix_dictionary(self.item_codes, self.levels, self.rvq.codebooks)
        if self.proxy_kind == 'identity':
            self.item_codes = torch.arange(len(item_vectors))[:, None].expand(-1, self.num_quantizers).clone()
            dictionary = prefix_dictionary(self.item_codes, self.levels, self.rvq.codebooks, item_vectors)
        for name, value in dictionary.items():
            if name == 'prefix_reconstruction':
                self.social_branch.prefix_reconstruction = value.to(device=device, dtype=embeds.dtype)
            else:
                setattr(self, name, value.to(device))
        self.item_codes = self.item_codes.to(device)
        self.queried_item_ids = queried_ids.to(device)
        self.rvq.to(device)
        self.behavioral_branch.to(device)
        full = self.prefix_sets(data.behavioral_edge_index, data.num_recommendations)
        self.prefix_user_frequency, self.prefix_information = prefix_frequency(
            full, len(self.prefix_levels), self.frequency_weighting)
        fit_mask = torch.isin(queried_ids, train_ids)
        self.quantization_summary = dict(
            fit=self.rvq.fit_summary, behavior_fingerprint=model_fingerprint(self.behavioral_branch),
            proxy_kind=self.proxy_kind, queried_items=len(item_vectors), fit_items=len(train_ids),
            query_only_items=int((~fit_mask).sum()), fixed_item_paths=True,
            item_embedding_view='train-edges; canonical CPU propagation',
            reconstruction_mse_per_level=errors.mean(0).tolist(),
            query_only_reconstruction_mse_per_level=errors[~fit_mask.cpu()].mean(0).tolist() if (~fit_mask).any() else None,
            elapsed_seconds=time.monotonic() - started)

    def train_model(self, *, data, writer, train_batch_size, val_batch_size, val_num_negatives,
                    num_epochs, optimizer, metrics, early_stopping, seed, deterministic):
        training = dict(train_batch_size=train_batch_size, val_batch_size=val_batch_size,
                        val_num_negatives=val_num_negatives, num_epochs=num_epochs,
                        optimizer=OmegaConf.to_container(optimizer, resolve=True),
                        metrics=OmegaConf.to_container(metrics, resolve=True),
                        early_stopping=OmegaConf.to_container(early_stopping, resolve=True))
        behavior_config = OmegaConf.create(dict(name='ItemPrefixBehavior', revision='item-prefix-behavior-e100-v1',
            attribute=dict(_target_='sralt.shadow.model.item_prefix.ItemPrefixBehavior', embed_dim=self.embed_dim,
                           num_layers=self.num_layers, decay=self.decay), train=training))
        encoder = load_shadow_model(seed=seed, deterministic=deterministic, device=data.device,
                                    config=behavior_config, data=data, writer=None)
        self.behavioral_branch = encoder.behavioral_branch
        self.behavioral_branch.requires_grad_(False)
        self.branch_training = dict(encoder.branch_training)
        self.eval()
        # Same RNG before social initialization/sampling for every graph ablation,
        # regardless of whether the behavior/RVQ cache was hit.
        set_seed(seed + 1, deterministic=deterministic)
        self.social_branch = PrefixGraphBranch(self.num_users + self.num_items, self.embed_dim, self.num_layers).to(data.device)
        self.prepare_prefixes(data, seed)
        train_loader, val_loader = recommendation_loaders(data, train_batch_size, val_batch_size, val_num_negatives)
        idf = self.recommendation_idf(data.behavioral_edge_index)
        adj = self.prefix_adj(data.social_edge_index, self.prefix_sets(data.behavioral_train_edge_index, data.num_recommendations))
        self._train_branch('social', adj, data.behavioral_train_edge_index, idf,
                           train_loader, val_loader, num_epochs, optimizer, instantiate(metrics), early_stopping, writer)
        self.branch_training['social']['regularization'] = 'sampled user/positive/negative embedding rows; projection and frozen proxies excluded'

    @torch.no_grad()
    def test_model(self, *, data, metrics, test_batch_size, test_num_negatives):
        self.eval()
        edges = data.behavioral_train_edge_index
        idf = self.recommendation_idf(data.behavioral_edge_index)
        behavior = self.behavioral_branch(self.behavior_adj(edges, idf))
        social = self.social_branch(self.prefix_adj(data.social_edge_index, self.prefix_sets(edges, data.num_recommendations)))
        embeds = torch.cat([behavior, social[:self.num_users + self.num_items]], dim=-1)
        loader = TensorBatchLoader(EvalShadowDataset(data, split='test', num_negatives=test_num_negatives),
                                   batch_size=test_batch_size, shuffle=False)
        return compute_recommendation_metrics(embeds, metrics, loader)

    @torch.no_grad()
    def get_attack_payload(self, data):
        self.eval()
        edges = data.behavioral_edge_index
        idf = self.recommendation_idf(edges)
        embeds, behavior, mean, variance, *_ = self.behavior(self.behavior_adj(edges, idf), edges, idf)
        prefix_ids = self.prefix_sets(edges, data.num_recommendations)
        social = self.social_branch(self.prefix_adj(data.social_edge_index, prefix_ids))[:self.num_users]
        sparse = self.social_branch(self.prefix_adj(data.social_edge_index[:, :0], prefix_ids))[:self.num_users]
        layers = []
        train_ids = (data.behavioral_train_edge_index[1] - self.num_users).unique()
        query_only = ~torch.isin(self.queried_item_ids, train_ids)
        for column, level in enumerate(self.levels):
            _, inverse, bucket = self.item_prefix_ids[:, column].unique(return_inverse=True, return_counts=True)
            frequency = self.prefix_user_frequency[self.prefix_levels == level]
            layers.append(dict(level=level, prefixes=len(bucket), collision_pairs=int((bucket * (bucket - 1) // 2).sum()),
                               singleton_ratio=float((bucket == 1).sum() / len(self.queried_item_ids)),
                               distinct_path_ratio=len(bucket) / len(self.queried_item_ids),
                               max_bucket=int(bucket.max()), max_user_frequency=int(frequency.max()),
                               query_only_singleton_ratio=float((bucket[inverse[query_only]] == 1).float().mean()) if query_only.any() else None))
        semantic, _ = semantic_links(prefix_ids, self.prefix_information, self.num_users + self.num_items)
        degree = torch.bincount(data.social_edge_index[0], minlength=self.num_users)
        linked = torch.bincount(semantic[0, semantic[0] < self.num_users], minlength=self.num_users) if self.semantic_augmentation else degree.new_zeros(degree.shape)
        self.last_diagnostics = dict(quantization=self.quantization_summary, levels=layers,
            semantic_nodes=len(self.prefix_levels), semantic_edges=semantic.size(1) // 2 if self.semantic_augmentation else 0,
            frequency_weighting=self.frequency_weighting, frequency_view='full-query',
            graph_view='full-query', public_zero_degree=int((degree == 0).sum()),
            augmented_zero_degree=int(((degree + linked) == 0).sum()),
            train_prefix_edges=int((self.prefix_sets(data.behavioral_train_edge_index, data.num_recommendations) >= 0).sum()))
        logging.info('ItemPrefixShadow diagnostics: %s', self.last_diagnostics)
        payload = dict(prefix_ids=prefix_ids, prefix_information=self.prefix_information,
                       prefix_mass=prefix_mass(prefix_ids, self.prefix_information), sparse_social=sparse)
        return torch.cat([behavior, social, mean, variance], dim=-1), payload

    def get_user_features(self, data):
        return self.get_attack_payload(data)[0]

    def __repr__(self):
        settings = {k: getattr(self, k) for k in ('num_users', 'num_items', 'embed_dim', 'num_layers', 'decay',
            'num_quantizers', 'codebook_size', 'levels', 'rvq_max_iter', 'semantic_augmentation', 'frequency_weighting', 'proxy_kind')}
        return f'ItemPrefixShadow({settings}, fingerprint={model_fingerprint(self)})'

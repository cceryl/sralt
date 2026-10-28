"""Joint user/item RVQ followed by jointly trained, late-fused social graphs."""

import logging
import time

import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from vector_quantize_pytorch import ResidualVQ

from .base import load_shadow_model
from .item_rvq import ItemResidualQuantizer, load_item_rvq_model
from .item_prefix import (ItemPrefixBehavior, JointItemPrefixBehavior, ItemPrefixShadow,
                          QuantizedBehaviorBranch, recommendation_loaders)
from .prefix import semantic_links, prefix_mass
from .tail_proxy import GraphBranch
from sralt.shadow.data import EvalShadowDataset
from sralt.utils import (TensorBatchLoader, build_normalized_adj,
                         compute_recommendation_metrics, model_fingerprint, set_seed)


class DualQuantizedBehaviorBranch(QuantizedBehaviorBranch):
    def __init__(self, num_users, num_items, embed_dim, num_layers,
                 num_quantizers=8, codebook_size=256, user_num_quantizers=8,
                 user_codebook_size=256, quantized_scoring='both', user_loss_weight=1., item_loss_weight=1.):
        super().__init__(num_users, num_items, embed_dim, num_layers, num_quantizers, codebook_size)
        if quantized_scoring not in ('both', 'user', 'item', 'none'):
            raise ValueError('Unknown behavior quantized_scoring.')
        self.quantized_scoring = quantized_scoring
        self.user_loss_weight, self.item_loss_weight = user_loss_weight, item_loss_weight
        self.user_quantizer = ResidualVQ(
            dim=embed_dim, num_quantizers=user_num_quantizers, codebook_size=user_codebook_size,
            learnable_codebook=True, ema_update=False, rotation_trick=False,
            commitment_weight=.25, kmeans_init=True, kmeans_iters=10,
            threshold_ema_dead_code=0, quantize_dropout=False)
        for name in ('fit_user_ids', 'non_fit_user_ids'):
            self.register_buffer(name, torch.empty(0, dtype=torch.long))

    def quantize(self, vectors, quantizer, fit_ids, other_ids):
        if not self.training:
            result, _, loss = quantizer(vectors.unsqueeze(0))
            return result.squeeze(0), loss.mean()
        fit, _, loss = quantizer(vectors[fit_ids].unsqueeze(0))
        result = torch.empty_like(vectors).index_copy(0, fit_ids, fit.squeeze(0))
        if other_ids.numel():
            other, _, _ = quantizer(vectors[other_ids].unsqueeze(0), freeze_codebook=True)
            result = result.index_copy(0, other_ids, other.squeeze(0))
        return result, loss.mean()

    def forward_with_loss(self, adj):
        raw = GraphBranch.forward(self, adj)
        users, items = raw[:self.num_users], raw[self.num_users:]
        uq, uloss = self.quantize(users, self.user_quantizer, self.fit_user_ids, self.non_fit_user_ids)
        iq, iloss = self.quantize(items, self.quantizer, self.fit_item_ids, self.non_fit_item_ids)
        return torch.cat([uq if self.quantized_scoring in ('both', 'user') else users,
                          iq if self.quantized_scoring in ('both', 'item') else items]), (
                              self.user_loss_weight * uloss + self.item_loss_weight * iloss)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ('fit_user_ids', 'non_fit_user_ids'):
            value = state_dict.get(prefix + name)
            if value is not None:
                setattr(self, name, getattr(self, name).new_empty(value.shape))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class ItemPrefixCSABehavior(JointItemPrefixBehavior):
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=2, decay=.01,
                 num_quantizers=8, codebook_size=256, user_num_quantizers=8,
                 user_codebook_size=256, quantized_scoring='both', user_loss_weight=1., item_loss_weight=1.):
        super().__init__(num_users, num_items, embed_dim, num_layers, decay, num_quantizers, codebook_size)
        branch = DualQuantizedBehaviorBranch(num_users, num_items, embed_dim, num_layers,
            num_quantizers, codebook_size, user_num_quantizers, user_codebook_size,
            quantized_scoring, user_loss_weight, item_loss_weight)
        branch.embedding = self.behavioral_branch.embedding
        self.behavioral_branch = branch

    def train_model(self, *, data, writer, seed, deterministic, **training):
        branch = self.behavioral_branch
        edges = data.behavioral_train_edge_index
        for kind, ids, size in (('user', edges[0].unique(sorted=True), self.num_users),
                               ('item', (edges[1] - self.num_users).unique(sorted=True), self.num_items)):
            setattr(branch, f'fit_{kind}_ids', ids)
            covered = torch.zeros(size, dtype=torch.bool, device=data.device)
            covered[ids] = True
            setattr(branch, f'non_fit_{kind}_ids', (~covered).nonzero(as_tuple=True)[0])
        idf = self.recommendation_idf(data.behavioral_edge_index)
        with torch.no_grad():
            raw = GraphBranch.forward(branch, self.behavior_adj(edges, idf))
            set_seed(seed + 17027, deterministic=deterministic)
            branch.quantizer(raw[self.num_users:][branch.fit_item_ids].unsqueeze(0), freeze_codebook=True)
            set_seed(seed + 27027, deterministic=deterministic)
            branch.user_quantizer(raw[branch.fit_user_ids].unsqueeze(0), freeze_codebook=True)
        set_seed(seed, deterministic=deterministic)
        ItemPrefixBehavior.train_model(self, data=data, writer=writer, **training)

    def __repr__(self):
        return f'ItemPrefixCSABehavior(scoring={self.behavioral_branch.quantized_scoring}, fingerprint={model_fingerprint(self)})'


class FusedSocialBranch(nn.Module):
    """Independent user tables and one item scoring table; no item message edges."""

    def __init__(self, num_users, num_items, embed_dim, num_layers, num_codes,
                 graph_mode='dual', fusion='linear'):
        super().__init__()
        if graph_mode not in ('dual', 'prefix', 'user', 'joint') or fusion not in ('linear', 'mean', 'gate'):
            raise ValueError('Unknown social graph mode or fusion.')
        self.num_users, self.num_layers = num_users, num_layers
        self.graph_mode, self.fusion = graph_mode, fusion
        self.items = nn.Embedding(num_items, embed_dim)
        self.prefix_users = nn.Embedding(num_users, embed_dim) if graph_mode != 'user' else None
        self.code_users = nn.Embedding(num_users, embed_dim) if graph_mode in ('dual', 'user') else None
        self.codes = nn.Embedding(num_codes, embed_dim) if graph_mode != 'prefix' else None
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False) if graph_mode != 'user' else None
        self.register_buffer('prefix_reconstruction', torch.empty(0, embed_dim))
        for module in (self.items, self.prefix_users, self.code_users, self.codes):
            if module is not None:
                nn.init.xavier_uniform_(module.weight)
        if self.projection is not None:
            nn.init.eye_(self.projection.weight)
        self.fuser = nn.Linear(2 * embed_dim, embed_dim) if graph_mode == 'dual' and fusion != 'mean' else None

    def propagate(self, adj, x):
        result = x
        for _ in range(self.num_layers):
            x = torch.sparse.mm(adj, x)
            result = result + x
        return (result / (self.num_layers + 1))[:self.num_users]

    def user_views(self, graphs):
        prefix_adj, code_adj, joint_adj = graphs
        sp = sc = None
        if self.graph_mode == 'joint':
            return self.propagate(joint_adj, torch.cat([self.prefix_users.weight,
                self.projection(self.prefix_reconstruction), self.codes.weight])), None
        if self.prefix_users is not None:
            sp = self.propagate(prefix_adj, torch.cat([
                self.prefix_users.weight, self.projection(self.prefix_reconstruction)]))
        if self.code_users is not None:
            sc = self.propagate(code_adj, torch.cat([self.code_users.weight, self.codes.weight]))
        return sp, sc

    def forward(self, graphs):
        sp, sc = self.user_views(graphs)
        if sp is None:
            users = sc
        elif sc is None:
            users = sp
        elif self.fusion == 'mean':
            users = (sp + sc) / 2
        elif self.fusion == 'gate':
            gate = self.fuser(torch.cat([sp, sc], dim=-1)).sigmoid()
            users = gate * sp + (1 - gate) * sc
        else:
            users = self.fuser(torch.cat([sp, sc], dim=-1))
        return torch.cat([users, self.items.weight])

    def regularization(self, users, positive, negative):
        reg = self.items(torch.cat([positive, negative]) - self.num_users).square().sum()
        for table in (self.prefix_users, self.code_users):
            if table is not None:
                reg = reg + table(users).square().sum()
        return reg / users.numel()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        value = state_dict.get(prefix + 'prefix_reconstruction')
        if value is not None:
            self.prefix_reconstruction = self.prefix_reconstruction.new_empty(value.shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class ItemPrefixCSAShadow(ItemPrefixShadow):
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=2, decay=.01,
                 num_quantizers=8, codebook_size=256, levels=(8,), user_num_quantizers=8,
                 user_codebook_size=256, prefix_augmentation=True, user_augmentation=True,
                 graph_mode='dual', fusion='linear', random_user_graph=False,
                 fixed_user_codes=False, quantized_scoring='both', user_loss_weight=1., item_loss_weight=1.,
                 rvq_training='end_to_end', prefix_edge_scale=.1):
        super().__init__(num_users, num_items, embed_dim, num_layers, decay,
                         num_quantizers, codebook_size, levels, rvq_training=rvq_training,
                         prefix_edge_scale=prefix_edge_scale)
        self.user_num_quantizers, self.user_codebook_size = user_num_quantizers, user_codebook_size
        self.prefix_augmentation, self.user_augmentation = prefix_augmentation, user_augmentation
        self.graph_mode, self.fusion = graph_mode, fusion
        self.random_user_graph, self.fixed_user_codes = random_user_graph, fixed_user_codes
        self.quantized_scoring = quantized_scoring
        # The parent assigns this field during initialization; use the same
        # writable interface, then namespace it for the dual-RVQ encoder.
        self.encoder_cache_group = f'item_prefix_csa_{self.rvq_training}_{self.quantized_scoring}'
        self.user_loss_weight, self.item_loss_weight = user_loss_weight, item_loss_weight
        self.behavioral_branch = (DualQuantizedBehaviorBranch(**self.behavior_attributes())
            if rvq_training == 'end_to_end' else GraphBranch(num_users + num_items, embed_dim, num_layers))
        self.behavioral_branch.requires_grad_(False)
        self.user_rvq = ItemResidualQuantizer(torch.empty(0, 0, embed_dim, dtype=torch.float64))
        self.social_branch = self.new_social_branch()
        self.register_buffer('train_user_codes', torch.empty(0, user_num_quantizers, dtype=torch.long))
        self.register_buffer('user_permutations', torch.empty(0, user_num_quantizers, dtype=torch.long))

    def behavior_attributes(self):
        return {key: getattr(self, key) for key in ('num_users', 'num_items', 'embed_dim', 'num_layers',
            'num_quantizers', 'codebook_size', 'user_num_quantizers', 'user_codebook_size',
            'quantized_scoring', 'user_loss_weight', 'item_loss_weight')}

    def new_social_branch(self):
        return FusedSocialBranch(self.num_users, self.num_items, self.embed_dim, self.num_layers,
            self.user_num_quantizers * self.user_codebook_size, self.graph_mode, self.fusion)

    def train(self, mode=True):
        super().train(mode)
        # Social/Attack must never initialize or update a frozen quantizer buffer.
        self.behavioral_branch.eval()
        return self

    @torch.no_grad()
    def encode_users(self, raw):
        if self.rvq_training == 'offline':
            return self.user_rvq.encode(raw[:self.num_users])[0]
        _, codes, _ = self.behavioral_branch.user_quantizer(raw[:self.num_users].unsqueeze(0))
        return codes.squeeze(0)

    def graph_inputs(self, public, prefix_ids, user_codes):
        n = self.num_users
        p = len(self.prefix_levels)
        if self.prefix_augmentation:
            pe, pw = semantic_links(prefix_ids, self.prefix_information, n)
            pw = pw * self.prefix_edge_scale
        else:
            pe, pw = public.new_empty(2, 0), self.prefix_information.new_empty(0)
        codes = user_codes
        if self.random_user_graph:
            codes = codes.gather(0, self.user_permutations)
        if self.user_augmentation:
            layers = torch.arange(self.user_num_quantizers, device=codes.device) * self.user_codebook_size
            users = torch.arange(n, device=codes.device).repeat_interleave(self.user_num_quantizers)
            ce = torch.stack([users, (codes + layers + n).flatten()])
            ce = torch.cat([ce, ce.flip(0)], dim=1)
        else:
            ce = public.new_empty(2, 0)
        nc = self.user_num_quantizers * self.user_codebook_size
        prefix_adj = code_adj = joint_adj = None
        if self.graph_mode in ('dual', 'prefix'):
            prefix_adj = build_normalized_adj(torch.cat([public, pe], dim=1), n + p,
                torch.cat([pw.new_ones(public.size(1)), pw]))
        if self.graph_mode in ('dual', 'user'):
            code_adj = build_normalized_adj(torch.cat([public, ce], dim=1), n + nc)
        if self.graph_mode == 'joint':
            joint_codes = ce + (ce >= n).to(ce.dtype) * p
            joint_adj = build_normalized_adj(torch.cat([public, pe, joint_codes], dim=1), n + p + nc,
                torch.cat([pw.new_ones(public.size(1)), pw, pw.new_ones(ce.size(1))]))
        return prefix_adj, code_adj, joint_adj

    def train_model(self, *, data, writer, train_batch_size, val_batch_size, val_num_negatives,
                    num_epochs, optimizer, metrics, early_stopping, seed, deterministic):
        training = dict(train_batch_size=train_batch_size, val_batch_size=val_batch_size,
            val_num_negatives=val_num_negatives, num_epochs=num_epochs,
            optimizer=OmegaConf.to_container(optimizer, resolve=True),
            metrics=OmegaConf.to_container(metrics, resolve=True),
            early_stopping=OmegaConf.to_container(early_stopping, resolve=True), seed=seed, deterministic=deterministic)
        attributes = self.behavior_attributes()
        for key in ('num_users', 'num_items'):
            attributes.pop(key)
        config = OmegaConf.create(dict(name='ItemPrefixCSABehavior', revision='item-prefix-csa-behavior-v1',
            attribute=dict(_target_=f'{__name__}.ItemPrefixCSABehavior', decay=self.decay, **attributes), train=training))
        if self.rvq_training == 'offline':
            training.pop('seed')
            training.pop('deterministic')
            config = OmegaConf.create(dict(name='ItemPrefixBehavior', revision='item-prefix-behavior-e100-v1',
                attribute=dict(_target_='sralt.shadow.model.item_prefix.ItemPrefixBehavior',
                    embed_dim=self.embed_dim, num_layers=self.num_layers, decay=self.decay), train=training))
        encoder = load_shadow_model(seed=seed, deterministic=deterministic, device=data.device,
                                    config=config, data=data, writer=None)
        self.behavioral_branch = encoder.behavioral_branch.requires_grad_(False)
        self.branch_training = dict(encoder.branch_training)
        self.eval()
        set_seed(seed + 1, deterministic=deterministic)
        self.social_branch = self.new_social_branch().to(data.device)
        self.prepare_prefixes(data, seed)
        self.quantization_summary['training'] = self.rvq_training
        self.quantization_summary['quantized_scoring'] = self.quantized_scoring
        edges = data.behavioral_train_edge_index
        idf = self.recommendation_idf(data.behavioral_edge_index)
        with torch.no_grad():
            raw = GraphBranch.forward(self.behavioral_branch, self.behavior_adj(edges, idf))
            if self.rvq_training == 'offline':
                ids = edges[0].unique(sorted=True)
                self.user_rvq = load_item_rvq_model(embeddings=raw[ids].cpu(), item_ids=ids.cpu(),
                    source=dict(kind='user', behavior=model_fingerprint(self.behavioral_branch), query=repr(data), view='train'),
                    seed=seed + 27027, num_quantizers=self.user_num_quantizers,
                    codebook_size=self.user_codebook_size, max_iter=self.rvq_max_iter).to(data.device)
            self.train_user_codes = self.encode_users(raw)
        generator = torch.Generator().manual_seed(seed + 37027)
        self.user_permutations = torch.stack([torch.randperm(self.num_users, generator=generator)
            for _ in range(self.user_num_quantizers)], dim=-1).to(data.device)
        graphs = self.graph_inputs(data.social_edge_index,
            self.prefix_sets(edges, data.num_recommendations), self.train_user_codes)
        train_loader, val_loader = recommendation_loaders(data, train_batch_size, val_batch_size, val_num_negatives)
        self.train_social(graphs, train_loader, val_loader, num_epochs, optimizer,
                          instantiate(metrics), early_stopping, writer)

    def train_social(self, graphs, train_loader, val_loader, num_epochs, optimizer, metrics, early_stopping, writer):
        module = self.social_branch
        opt = instantiate(optimizer, params=module.parameters())
        stop = instantiate(early_stopping, model=module)
        history, steps = [], 0
        started = time.monotonic()
        for epoch in range(1, num_epochs + 1):
            self.train()
            totals, count = dict(bpr=0., raw_l2=0., loss=0.), 0
            for users, positive, negative in train_loader:
                opt.zero_grad()
                embeds = module(graphs)
                pos = (embeds[users] * embeds[positive]).sum(-1)
                neg = (embeds[users] * embeds[negative]).sum(-1)
                bpr = nn.functional.softplus(neg - pos).mean()
                reg = module.regularization(users, positive, negative)
                loss = bpr + self.decay * reg
                loss.backward()
                opt.step()
                for key, value in (('bpr', bpr), ('raw_l2', reg), ('loss', loss)):
                    totals[key] += value.detach().item() * users.numel()
                count += users.numel()
                steps += 1
            self.eval()
            with torch.no_grad():
                values = compute_recommendation_metrics(module(graphs), metrics, val_loader)
            record = dict(epoch=epoch, validation=values, **{key: value / count for key, value in totals.items()})
            history.append(record)
            logging.info('ItemPrefixCSA fused social: %s', record)
            if writer is not None:
                writer.add_scalar('ItemPrefixCSA/social/Loss/train', record['loss'], epoch)
            if stop.step(sum(values)):
                break
        stop.load_best_state()
        self.eval()
        self.branch_training['social'] = dict(max_epochs=num_epochs, actual_epochs=len(history),
            best_epoch=stop.best_epoch, selected_epoch=stop.best_epoch, best_metric=stop.best_metric,
            selection='fused_recommendation_validation_best', steps=steps,
            stop_reason='early_stopping' if stop.counter >= stop.patience else 'epoch_limit',
            regularization='sum sampled raw user rows from active branches and shared item rows / batch size; code tables, projection and fusion excluded',
            elapsed_seconds=time.monotonic() - started, history=history)

    @torch.no_grad()
    def test_model(self, *, data, metrics, test_batch_size, test_num_negatives):
        self.eval()
        graphs = self.graph_inputs(data.social_edge_index,
            self.prefix_sets(data.behavioral_train_edge_index, data.num_recommendations), self.train_user_codes)
        loader = TensorBatchLoader(EvalShadowDataset(data, split='test', num_negatives=test_num_negatives),
                                   batch_size=test_batch_size, shuffle=False)
        return compute_recommendation_metrics(self.social_branch(graphs), metrics, loader)

    @torch.no_grad()
    def get_attack_payload(self, data):
        self.eval()
        edges = data.behavioral_edge_index
        idf = self.recommendation_idf(edges)
        raw = GraphBranch.forward(self.behavioral_branch, self.behavior_adj(edges, idf))
        mean, variance = self.aggregate(raw, edges, idf)
        full_codes = self.encode_users(raw)
        codes = self.train_user_codes if self.fixed_user_codes else full_codes
        prefixes = self.prefix_sets(edges, data.num_recommendations)
        graphs = self.graph_inputs(data.social_edge_index, prefixes, codes)
        social = self.social_branch(graphs)[:self.num_users]
        sparse = self.social_branch(self.graph_inputs(data.social_edge_index[:, :0], prefixes, codes))[:self.num_users]
        offset_codes = codes + torch.arange(self.user_num_quantizers, device=codes.device) * self.user_codebook_size
        frequency = torch.bincount(offset_codes.flatten(), minlength=self.user_num_quantizers * self.user_codebook_size)
        information = ((self.num_users + 1.) / (frequency.float() + 1.)).log()
        sp, sc = self.social_branch.user_views(graphs)
        prefix_frequency = self.prefix_user_frequency
        public_degree = torch.bincount(data.social_edge_index[0], minlength=self.num_users).float()
        pe, pw = semantic_links(prefixes, self.prefix_information, self.num_users)
        prefix_active = self.prefix_augmentation and self.graph_mode != 'user'
        user_active = self.user_augmentation and self.graph_mode != 'prefix'
        source_users = pe[0] < self.num_users
        prefix_degree = public_degree.new_zeros(self.num_users)
        if prefix_active:
            prefix_degree.index_add_(0, pe[0, source_users], pw[source_users] * self.prefix_edge_scale)
        code_degree = public_degree.new_full((self.num_users,), self.user_num_quantizers if user_active else 0)
        def semantic_share(degree):
            total = public_degree + degree
            return float((degree / total.clamp_min(1e-12))[total > 0].mean()) if (total > 0).any() else 0.
        self.last_diagnostics = dict(quantization=self.quantization_summary, graph_mode=self.graph_mode,
            fusion=self.fusion, prefix_augmentation=self.prefix_augmentation, user_augmentation=self.user_augmentation,
            prefix_edge_scale=self.prefix_edge_scale,
            random_user_graph=self.random_user_graph, user_code_view='train' if self.fixed_user_codes else 'full-query',
            user_code_change_per_level=(full_codes != self.train_user_codes).float().mean(0).tolist(),
            user_code_active_per_level=(frequency.reshape(self.user_num_quantizers, -1) > 0).sum(1).tolist(),
            user_code_collision_pairs_per_level=(frequency.reshape(self.user_num_quantizers, -1) *
                (frequency.reshape(self.user_num_quantizers, -1) - 1) // 2).sum(1).tolist(),
            prefix_nodes=len(prefix_frequency),
            prefix_edges=int(source_users.sum()) if prefix_active else 0,
            user_code_edges=self.num_users * self.user_num_quantizers if user_active else 0,
            semantic_weight_share=dict(prefix=semantic_share(prefix_degree), user_code=semantic_share(code_degree),
                joint=semantic_share(prefix_degree + code_degree) if self.graph_mode == 'joint' else None),
            mean_norms=dict(behavior=float(raw[:self.num_users].norm(dim=-1).mean()),
                fused=float(social.norm(dim=-1).mean()), prefix=float(sp.norm(dim=-1).mean()) if sp is not None else None,
                user_code=float(sc.norm(dim=-1).mean()) if sc is not None else None),
            social_parameters=sum(p.numel() for p in self.social_branch.parameters()),
            behavior_parameters=sum(p.numel() for p in self.behavioral_branch.parameters()))
        payload = dict(prefix_ids=prefixes, prefix_information=self.prefix_information,
            prefix_mass=prefix_mass(prefixes, self.prefix_information), sparse_social=sparse,
            user_code_ids=offset_codes, user_code_information=information)
        return torch.cat([raw[:self.num_users], social, mean, variance], dim=-1), payload

    def __repr__(self):
        settings = {key: getattr(self, key) for key in ('num_users', 'num_items', 'embed_dim', 'num_layers', 'decay',
            'num_quantizers', 'codebook_size', 'levels', 'user_num_quantizers', 'user_codebook_size',
            'prefix_augmentation', 'user_augmentation', 'prefix_edge_scale', 'graph_mode', 'fusion', 'random_user_graph',
            'fixed_user_codes', 'quantized_scoring', 'user_loss_weight', 'item_loss_weight', 'rvq_training')}
        return f'ItemPrefixCSAShadow({settings}, fingerprint={model_fingerprint(self)})'

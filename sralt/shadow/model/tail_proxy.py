"""IDF-conditioned RVQ shadow with public-social and semantic propagation."""

import json
import logging

import torch
import torch.nn as nn
from hydra.utils import instantiate
from vector_quantize_pytorch import ResidualVQ
from torch_geometric.utils import coalesce, to_undirected

from .base import ShadowModel
from sralt.shadow.data import TrainShadowDataset, EvalShadowDataset
from sralt.utils import TensorBatchLoader, build_normalized_adj, compute_recommendation_metrics, model_fingerprint


class GraphBranch(nn.Module):
    """LightGCN with an explicitly weighted, symmetrically normalized adjacency."""

    def __init__(self, num_nodes, embed_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.num_layers = num_layers

    def forward(self, adj):
        x = self.embedding.weight
        result = x
        for _ in range(self.num_layers):
            x = torch.sparse.mm(adj, x)
            result = result + x
        return result / (self.num_layers + 1)


class TailProxyShadow(ShadowModel):
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=2, decay=1e-2,
                 num_quantizers=8, codebook_size=256, commitment_weight=.01,
                 idf_weighting=True, rvq_enabled=True, semantic_augmentation=True):
        super().__init__(num_users, num_items)
        self.embed_dim, self.num_layers, self.decay = embed_dim, num_layers, decay
        self.num_quantizers, self.codebook_size = num_quantizers, codebook_size
        self.commitment_weight = commitment_weight
        self.idf_weighting, self.rvq_enabled = idf_weighting, rvq_enabled
        # RVQ-off removes dependent semantic edges and code matching as well.
        self.semantic_augmentation = semantic_augmentation
        self.num_codes = num_quantizers * codebook_size if rvq_enabled and semantic_augmentation else 0
        self.behavioral_branch = GraphBranch(num_users + num_items, embed_dim, num_layers)
        self.social_branch = GraphBranch(num_users + num_items + self.num_codes, embed_dim, num_layers)
        self.rvq = ResidualVQ(dim=embed_dim, num_quantizers=num_quantizers,
                              codebook_size=codebook_size, commitment_weight=commitment_weight,
                              kmeans_init=True, kmeans_iters=10) if rvq_enabled else None

    def recommendation_idf(self, edges):
        """Count distinct queried users, never recommendation occurrences."""
        edges = coalesce(edges)
        counts = torch.bincount(edges[1] - self.num_users, minlength=self.num_items)
        return ((self.num_users + 1) / (counts.float() + 1)).log()

    def behavior_adj(self, edges, idf):
        weights = idf[edges[1] - self.num_users] if self.idf_weighting else torch.ones_like(edges[0], dtype=idf.dtype)
        return build_normalized_adj(torch.cat([edges, edges.flip(0)], dim=1),
                                    self.num_users + self.num_items, weights.repeat(2))

    def aggregate(self, embeds, edges, idf):
        """Weighted population mean/variance; zero information mass gives zero."""
        users, items = edges
        weights = idf[items - self.num_users] if self.idf_weighting else torch.ones_like(users, dtype=idf.dtype)
        mass = embeds.new_zeros(self.num_users).index_add(0, users, weights)
        mean = embeds.new_zeros(self.num_users, self.embed_dim).index_add(
            0, users, weights[:, None] * embeds[items]) / mass[:, None].clamp_min(1e-12)
        variance = embeds.new_zeros(self.num_users, self.embed_dim).index_add(
            0, users, weights[:, None] * (embeds[items] - mean[users]).square()) / mass[:, None].clamp_min(1e-12)
        return mean, variance

    def behavior(self, adj, edges, idf):
        embeds = self.behavioral_branch(adj)
        mean, variance = self.aggregate(embeds, edges, idf)
        # Fixed residual conditioning keeps the path active without a gate that can close.
        users = (embeds[:self.num_users] + mean) / 2
        if self.rvq is None:
            codes = torch.empty((self.num_users, 0), dtype=torch.long, device=embeds.device)
            quantized, loss = users, users.new_zeros(())
        else:
            quantized, codes, losses = self.rvq(users)
            loss = losses.sum()
        scoring = torch.cat([quantized, embeds[self.num_users:]])
        return scoring, users, mean, variance, codes, loss

    def semantic_edges(self, codes):
        if not self.num_codes:
            return codes.new_empty((2, 0))
        offset = torch.arange(self.num_quantizers, device=codes.device) * self.codebook_size
        users = torch.arange(self.num_users, device=codes.device).repeat_interleave(self.num_quantizers)
        nodes = (codes + offset + self.num_users + self.num_items).flatten()
        return to_undirected(torch.stack([users, nodes]))

    def social_adj(self, public_edges, semantic_edges):
        return build_normalized_adj(torch.cat([public_edges, semantic_edges], dim=1),
                                    self.num_users + self.num_items + self.num_codes)

    def _train_branch(self, branch, adj, edges, idf, train_loader, val_loader,
                      num_epochs, optimizer, metrics, early_stopping, writer):
        module = self.behavioral_branch if branch == 'behavioral' else self.social_branch
        opt = instantiate(optimizer, params=module.parameters())
        # Includes RVQ EMA buffers, rather than checkpointing only the embedding table.
        stop = instantiate(early_stopping, model=self)
        for epoch in range(1, num_epochs + 1):
            self.train()
            total_loss, count = 0., 0
            for users, positive, negative in train_loader:
                opt.zero_grad()
                if branch == 'behavioral':
                    embeds, *_, commitment = self.behavior(adj, edges, idf)
                else:
                    embeds, commitment = module(adj), 0.
                pos = (embeds[users] * embeds[positive]).sum(-1)
                neg = (embeds[users] * embeds[negative]).sum(-1)
                nodes = torch.cat([users, positive, negative])
                reg = module.embedding(nodes).square().sum() / users.numel()
                loss = nn.functional.softplus(neg - pos).mean() + self.decay * reg + commitment
                loss.backward()
                opt.step()
                total_loss += loss.item() * users.numel()
                count += users.numel()
            self.eval()
            with torch.no_grad():
                embeds = self.behavior(adj, edges, idf)[0] if branch == 'behavioral' else module(adj)
                values = compute_recommendation_metrics(embeds, metrics, val_loader)
            logging.info('TailProxyShadow %s epoch=%d loss=%.6f validation=%s', branch, epoch, total_loss / count, values)
            if writer is not None:
                writer.add_scalar(f'TailProxyShadow/{branch}/Loss/train', total_loss / count, epoch)
                for metric, value in zip(metrics, values):
                    writer.add_scalar(f'TailProxyShadow/{branch}/{type(metric).__name__}/val', value, epoch)
            if stop.step(sum(values)):
                break
        stop.load_best_state()
        self.eval()

    def train_model(self, *, data, writer, train_batch_size, val_batch_size, val_num_negatives,
                    num_epochs, optimizer, metrics, early_stopping):
        # All queried lists may determine IDF; propagation/aggregation here uses train edges only.
        idf = self.recommendation_idf(data.behavioral_edge_index)
        edges = data.behavioral_train_edge_index
        adj = self.behavior_adj(edges, idf)
        train_loader = TensorBatchLoader(TrainShadowDataset(data), batch_size=train_batch_size, shuffle=True)
        val_loader = TensorBatchLoader(
            EvalShadowDataset(data, split='val', num_negatives=val_num_negatives),
            batch_size=val_batch_size, shuffle=False
        )
        common = dict(edges=edges, idf=idf, train_loader=train_loader, val_loader=val_loader,
                      num_epochs=num_epochs, optimizer=optimizer, metrics=instantiate(metrics),
                      early_stopping=early_stopping, writer=writer)
        self._train_branch('behavioral', adj=adj, **common)
        with torch.no_grad():
            codes = self.behavior(adj, edges, idf)[4]
            social_adj = self.social_adj(data.social_edge_index, self.semantic_edges(codes))
        self._train_branch('social', adj=social_adj, **common)

    @torch.no_grad()
    def test_model(self, *, data, metrics, test_batch_size, test_num_negatives):
        self.eval()
        idf = self.recommendation_idf(data.behavioral_edge_index)
        edges = data.behavioral_train_edge_index
        behavior, *_, codes, _ = self.behavior(self.behavior_adj(edges, idf), edges, idf)
        social = self.social_branch(self.social_adj(data.social_edge_index, self.semantic_edges(codes)))
        # Concatenated branches implement the sum of their recommendation scores.
        embeds = torch.cat([behavior, social[:self.num_users + self.num_items]], dim=-1)
        loader = TensorBatchLoader(
            EvalShadowDataset(data, split='test', num_negatives=test_num_negatives),
            batch_size=test_batch_size, shuffle=False
        )
        return compute_recommendation_metrics(embeds, metrics, loader)

    @torch.no_grad()
    def get_attack_payload(self, data):
        """Only queried recommendation lists and attack-training public edges are read."""
        self.eval()
        edges = coalesce(data.behavioral_edge_index)
        idf = self.recommendation_idf(edges)
        _, behavior, mean, variance, codes, _ = self.behavior(self.behavior_adj(edges, idf), edges, idf)
        semantic = self.semantic_edges(codes)
        social = self.social_branch(self.social_adj(data.social_edge_index, semantic))[:self.num_users]
        # Repropagate the same learned parameters on a graph with NO real social edges.
        sparse = self.social_branch(self.social_adj(data.social_edge_index[:, :0], semantic))[:self.num_users]
        features = torch.cat([behavior, social, mean, variance], dim=-1)
        payload = dict(sparse_social=sparse, codes=codes,
                       recommendations=(edges[1] - self.num_users).reshape(self.num_users, -1), idf=idf)
        degree = torch.bincount(data.social_edge_index[0], minlength=self.num_users)
        augmented_degree = torch.bincount(torch.cat([data.social_edge_index, semantic], 1)[0],
                                         minlength=self.num_users + self.num_items + self.num_codes)[:self.num_users]
        usage = []
        for level in range(codes.size(1)):
            counts = torch.bincount(codes[:, level], minlength=self.codebook_size).float()
            probs = counts[counts > 0] / self.num_users
            usage.append(dict(level=level, active_codes=int((counts > 0).sum()),
                              max_fraction=float(probs.max()), perplexity=float((-probs * probs.log()).sum().exp())))
        self.last_diagnostics = dict(code_usage=usage, semantic_nodes=self.num_codes,
                                     active_semantic_nodes=int(semantic[1, semantic[1] >= self.num_users].unique().numel()),
                                     semantic_edges=semantic.size(1) // 2,
                                     public_zero_degree=int((degree == 0).sum()),
                                     augmented_zero_degree=int((augmented_degree == 0).sum()))
        logging.info('TailProxyShadow diagnostics: %s', json.dumps(self.last_diagnostics))
        if any(level['max_fraction'] > .9 for level in usage):
            logging.warning('TailProxyShadow RVQ has a level with >90%% users in one code; inspect collapse before interpreting results.')
        return features, payload

    def get_user_features(self, data):
        return self.get_attack_payload(data)[0]

    def __repr__(self):
        settings = {key: getattr(self, key) for key in (
            'num_users', 'num_items', 'embed_dim', 'num_layers', 'decay', 'num_quantizers',
            'codebook_size', 'commitment_weight', 'idf_weighting', 'rvq_enabled', 'semantic_augmentation')}
        return f'TailProxyShadow({settings}, fingerprint={model_fingerprint(self)})'

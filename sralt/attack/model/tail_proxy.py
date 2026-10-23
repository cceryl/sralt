"""Relation classifier using normalized blocks, RVQ agreement and queried overlap."""

import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

from .base import AttackModel
from sralt.attack.data import TrainAttackDataset, EvalAttackDataset
from sralt.utils import DataLoader, head_tail_split, compute_attack_metrics, model_fingerprint


class TailProxyAttack(AttackModel):
    recommendation_dim = 3

    def __init__(self, user_feature_dim, num_quantizers=8, rvq_enabled=True,
                 normalize_features=True, code_matching=True, idf_pair_signal=True,
                 sparse_training=True, sparse_probability=.5, hidden_dim=128, dropout=.2):
        super().__init__(user_feature_dim)
        self.num_quantizers, self.rvq_enabled = num_quantizers, rvq_enabled
        self.normalize_features, self.code_matching = normalize_features, code_matching
        self.idf_pair_signal = idf_pair_signal
        self.sparse_training, self.sparse_probability = sparse_training, sparse_probability
        self.hidden_dim, self.dropout = hidden_dim, dropout
        if not 0 <= sparse_probability <= 1:
            raise ValueError('sparse_probability must be in [0, 1].')
        if user_feature_dim % 4:
            raise ValueError('TailProxyAttack requires four equally sized continuous blocks.')
        semantic_dim = (num_quantizers if code_matching and rvq_enabled else 0) + (self.recommendation_dim if idf_pair_signal else 0)
        self.continuous = nn.Sequential(nn.Linear(4 * user_feature_dim, hidden_dim), nn.LeakyReLU(.2))
        self.alignment = nn.Sequential(nn.Linear(semantic_dim, hidden_dim // 2), nn.LeakyReLU(.2)) if semantic_dim else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim + (hidden_dim // 2 if semantic_dim else 0), hidden_dim),
            nn.LeakyReLU(.2), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.loss_fn = nn.BCEWithLogitsLoss()

    def endpoint_features(self, users, data, sparse_mask=None):
        blocks = data.user_features[users].reshape(*users.shape, 4, -1).clone()
        if sparse_mask is not None:
            blocks[..., 1, :] = torch.where(sparse_mask[..., None],
                                            data.side_features['sparse_social'][users], blocks[..., 1, :])
        return F.normalize(blocks, p=2, dim=-1) if self.normalize_features else blocks

    def recommendation_evidence(self, users, data):
        """Raw count, exact sum(IDF over intersection), weighted Jaccard; O(B K²)."""
        recommendations, idf = data.side_features['recommendations'], data.side_features['idf']
        left, right = recommendations[users[0]], recommendations[users[1]]
        overlap = (left[:, :, None] == right[:, None, :]).any(-1)
        weighted = (overlap * idf[left]).sum(-1)
        union_mass = idf[left].sum(-1) + idf[right].sum(-1) - weighted
        return torch.stack([overlap.sum(-1).to(idf.dtype), weighted,
                            weighted / union_mass.clamp_min(1e-12)], dim=-1)

    def code_evidence(self, users, data):
        codes = data.side_features['codes']
        if codes.size(1) != self.num_quantizers:
            raise ValueError('Attack RVQ levels differ from the shadow codes.')
        return (codes[users[0]] == codes[users[1]]).to(data.user_features.dtype)

    def forward(self, user_edge_index, data, sparse_mask=None):
        # Canonicalize both IDs and explicit endpoint masks for undirected relations.
        users, order = user_edge_index.sort(dim=0)
        if sparse_mask is not None:
            sparse_mask = sparse_mask.gather(0, order)
        blocks = self.endpoint_features(users, data, sparse_mask)
        left, right = blocks.flatten(-2).unbind(0)
        pair = torch.cat([left, right, (left - right).abs(), left * right], dim=-1)
        representations = [self.continuous(pair)]
        evidence = []
        if self.code_matching and self.rvq_enabled:
            evidence.append(self.code_evidence(users, data))
        if self.idf_pair_signal:
            evidence.append(self.recommendation_evidence(users, data))
        if evidence:
            representations.append(self.alignment(torch.cat(evidence, dim=-1)))
        return self.classifier(torch.cat(representations, dim=-1)).squeeze(-1)

    @torch.no_grad()
    def predict(self, user_edge_index, data):
        return self(user_edge_index, data).sigmoid()

    def sample_sparse_mask(self, users):
        # No labels or degrees enter this decision. At p=.5, 75% of pairs have >=1 sparse endpoint.
        return torch.rand(users.shape, device=users.device) < self.sparse_probability

    def train_model(self, *, data, tail_ratio, writer, train_batch_size, val_batch_size,
                    negative_ratio, num_epochs, optimizer, metrics, early_stopping):
        train_loader = DataLoader(TrainAttackDataset(data, negative_ratio), batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(EvalAttackDataset(data, split='val'), batch_size=val_batch_size, shuffle=False)
        opt = instantiate(optimizer, params=self.parameters())
        metrics = instantiate(metrics)
        stop = instantiate(early_stopping, model=self)
        head, tail = head_tail_split(data.num_users, data.target_social_edge_index, tail_ratio)
        self.training_diagnostics = []
        degree = torch.bincount(data.shadow_social_edge_index[0], minlength=data.num_users)
        with torch.no_grad():
            blocks = self.endpoint_features(torch.arange(data.num_users, device=data.device), data)
            norm_report = {}
            for name, group in [('isolated', degree == 0), ('covered', degree > 0)]:
                norms = blocks[group].norm(dim=-1)
                norm_report[name] = dict(users=int(group.sum()),
                    mean_per_block=norms.mean(0).tolist() if norms.numel() else [],
                    zero_per_block=(norms == 0).sum(0).tolist())
            logging.info('TailProxyAttack block norms: %s', json.dumps(norm_report))
            self.normalization_diagnostics = norm_report
        for epoch in range(1, num_epochs + 1):
            self.train()
            total_loss, count = 0., 0
            counts = {f'{label}_{kind}': 0 for label in ('positive', 'negative')
                      for kind in ('endpoints', 'masked_endpoints', 'masked_pairs')}
            for u, v, labels in train_loader:
                users = torch.stack([u, v])
                mask = self.sample_sparse_mask(users) if self.sparse_training else torch.zeros_like(users, dtype=torch.bool)
                opt.zero_grad()
                loss = self.loss_fn(self(users, data, mask), labels.float())
                loss.backward()
                opt.step()
                total_loss += loss.item() * labels.numel()
                count += labels.numel()
                for label, value in [('positive', 1), ('negative', 0)]:
                    selected = labels == value
                    counts[f'{label}_endpoints'] += int(selected.sum()) * 2
                    counts[f'{label}_masked_endpoints'] += int(mask[:, selected].sum())
                    counts[f'{label}_masked_pairs'] += int(mask[:, selected].any(0).sum())
            self.training_diagnostics.append(dict(epoch=epoch, **counts))
            self.eval()
            values = compute_attack_metrics(data.device,
                lambda u, v: self.predict(torch.stack([u, v]), data), metrics, val_loader, head, tail)
            logging.info('TailProxyAttack epoch=%d loss=%.6f validation=%s sparse=%s',
                         epoch, total_loss / count, values[0], json.dumps(counts))
            if writer is not None:
                writer.add_scalar('TailProxyAttack/Loss/train', total_loss / count, epoch)
                for metric, value in zip(metrics, values[0]):
                    writer.add_scalar(f'TailProxyAttack/{type(metric).__name__}/val', value, epoch)
            if stop.step(sum(values[0])):
                break
        stop.load_best_state()
        self.eval()

    @torch.no_grad()
    def test_model(self, *, data, tail_ratio, metrics, test_batch_size):
        self.eval()
        head, tail = head_tail_split(data.num_users, data.target_social_edge_index, tail_ratio)
        loader = DataLoader(EvalAttackDataset(data, split='test'), batch_size=test_batch_size, shuffle=False)
        return compute_attack_metrics(data.device, lambda u, v: self.predict(torch.stack([u, v]), data),
                                      metrics, loader, head, tail)

    def __repr__(self):
        settings = {key: getattr(self, key) for key in (
            'user_feature_dim', 'num_quantizers', 'rvq_enabled', 'normalize_features', 'code_matching',
            'idf_pair_signal', 'sparse_training', 'sparse_probability', 'hidden_dim', 'dropout')}
        return f'{type(self).__name__}({settings}, fingerprint={model_fingerprint(self)})'


class TailProxyCountAttack(TailProxyAttack):
    """Change only the explicit recommendation branch from three signals to count."""

    recommendation_dim = 1

    def recommendation_evidence(self, users, data):
        recommendations = data.side_features['recommendations']
        left, right = recommendations[users[0]], recommendations[users[1]]
        return (left[:, :, None] == right[:, None, :]).any(-1).sum(-1, keepdim=True).float()

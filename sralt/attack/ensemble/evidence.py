"""Read-only adapters, aligned public identities and candidate-excluded degrees."""

from dataclasses import dataclass

import torch
from torch import nn

from sralt.attack import load_attack_data, load_attack_model
from sralt.shadow import load_shadow_model
from sralt.utils import model_fingerprint
from sralt.utils.model import tensor_fingerprint


def pair_ids(edges, num_users):
    edges = edges.sort(dim=0).values
    return edges[0] * num_users + edges[1]


def canonical_pairs(edges, num_users):
    ids = pair_ids(edges, num_users).unique(sorted=True)
    return torch.stack([ids // num_users, ids % num_users])


def public_identity(data):
    """No hidden labels, target graph, evaluation groups or test pairs in this key."""
    return dict(num_users=data.num_users, observed=tensor_fingerprint(data.shadow_social_edge_index),
                train=tensor_fingerprint(data.train_social_edge_index),
                validation=tensor_fingerprint(data.val_social_edge_index),
                features=tensor_fingerprint(data.user_features),
                side={k: tensor_fingerprint(v) for k, v in getattr(data, 'side_features', {}).items()},
                shadow_source=data.shadow_model_repr)


class ExpertBundle(nn.Module):
    """Own formal experts, but force eval even when an enclosing module trains."""

    def __init__(self, configs, data, models, sources):
        super().__init__()
        self.configs, self.data, self.sources = configs, data, sources
        self.models = nn.ModuleList(models)
        for model in self.models:
            model.requires_grad_(False)
            model.zero_grad(set_to_none=True)
        self.eval()

    @classmethod
    def load(cls, *, configs, target, shadow, protocol, device, seed, deterministic, needs_attack=True,
             shadow_loader=load_shadow_model, attack_loader=load_attack_model):
        data, models, sources = [], [], []
        common = dict(seed=seed, deterministic=deterministic)
        for cfg in configs:
            target.to(device)
            shadow.to(device)
            model = shadow_loader(**common, device=device, config=cfg.shadow.model, data=shadow, writer=None)
            model.eval().requires_grad_(False)
            payload = load_attack_data(**common, target_data=target, shadow_data=shadow,
                                       shadow_model=model, config=cfg.attack.data)
            for name, fingerprint in protocol['identity'].items():
                if tensor_fingerprint(getattr(payload, name)) != fingerprint:
                    raise ValueError(f'Expert public/evaluation identity differs: {name}')
            sources.append(dict(shadow=model_fingerprint(model),
                                shadow_config=str(cfg.shadow.model), public=public_identity(payload)))
            del model
            payload.fixed_eval_pairs = {'val': protocol['val']}
            payload.to(device)
            if needs_attack:
                attack = attack_loader(**common, device=device, config=cfg.attack.model, data=payload,
                                           tail_ratio=cfg.attack.tail_ratio, writer=None)
                attack.eval().requires_grad_(False)
                models.append(attack)
                sources[-1]['attack'] = model_fingerprint(attack)
            data.append(payload)
        if data[0].num_users != data[1].num_users:
            raise ValueError('Expert user indices are not aligned.')
        return cls(configs, data, models, sources)

    def train(self, mode=True):
        super().train(False)
        return self

    def fingerprints(self):
        # model_fingerprint includes BatchNorm running buffers.
        return [model_fingerprint(model) for model in self.models]

    def cache_key(self):
        return self.sources


@dataclass
class EnsembleEvidence:
    bundle: ExpertBundle

    def __post_init__(self):
        data = self.bundle.data[0]
        self.num_users = data.num_users
        for native in self.bundle.data:
            if not torch.equal(canonical_pairs(native.shadow_social_edge_index, self.num_users),
                               canonical_pairs(native.train_social_edge_index, self.num_users)):
                raise ValueError('Public degree graph must equal observed training T, excluding V.')
        self.observed = canonical_pairs(data.shadow_social_edge_index, data.num_users)
        self.observed = self.observed[:, self.observed[0] != self.observed[1]]
        self.observed_ids = pair_ids(self.observed, self.num_users)
        self.degree = torch.bincount(self.observed.flatten(), minlength=self.num_users).float()
        nonzero = self.degree[self.degree > 0]
        self.tau = float(torch.quantile(nonzero, .5)) if nonzero.numel() else 1.
        self.user_dims = tuple(2 * int(cfg.shadow.model.attribute.embed_dim) for cfg in self.bundle.configs)
        self.hidden_dims = tuple((model.net[-1] if hasattr(model, 'net') else model.classifier[-1]).in_features
                                 for model in self.bundle.models)
        self.users = tuple(data.user_features[:, :dim].detach()
                           for data, dim in zip(self.bundle.data, self.user_dims, strict=True))
        if any(features.size(1) != dim for features, dim in zip(self.users, self.user_dims, strict=True)):
            raise ValueError('Expert payload lacks the two native Shadow user blocks.')

    def degrees(self, edges):
        ids = pair_ids(edges, self.num_users)
        position = torch.searchsorted(self.observed_ids, ids)
        if self.observed_ids.numel():
            present = ((position < self.observed_ids.numel()) &
                       (self.observed_ids[position.clamp_max(self.observed_ids.numel() - 1)] == ids))
        else:
            present = torch.zeros_like(ids, dtype=torch.bool)
        # PyTorch rejects boolean subtraction; the indicator contributes 0/1
        # in the same numeric dtype as the observed degree counts.
        degree = self.degree[edges].T - present[:, None].to(self.degree.dtype)
        if (degree < 0).any():
            raise ValueError('Negative candidate-excluded public degree.')
        return degree

    @staticmethod
    def degree_features(degrees):
        a, b = degrees.log1p().sort(dim=-1).values.unbind(-1)
        return torch.stack([a, b, b - a, (degrees == 0).sum(-1)], dim=-1)

    @torch.no_grad()
    def batch(self, edges, *, scores=False, hidden=False, users=False):
        edges = edges.sort(dim=0).values
        degree = self.degrees(edges)
        result = dict(degrees=degree, degree_features=self.degree_features(degree))
        if users:
            result['users'] = tuple(features[edges].detach() for features in self.users)
        if scores or hidden:
            if len(self.bundle.models) != 2:
                raise ValueError('This condition requires both formal Attack models.')
            self.bundle.eval()
            logits, representations = [], []
            for model, data in zip(self.bundle.models, self.bundle.data, strict=True):
                values = []
                handle = None
                if hidden:
                    layer = model.net[-1] if hasattr(model, 'net') else model.classifier[-1]
                    handle = layer.register_forward_pre_hook(lambda module, args: values.append(args[0].detach()))
                try:
                    inputs = data if hasattr(model, 'endpoint_features') else data.user_features
                    logits.append(model(edges, inputs).detach())
                finally:
                    if handle is not None:
                        handle.remove()
                if hidden:
                    if len(values) != 1:
                        raise ValueError('Expected exactly one final Linear input.')
                    representations.append(values[0])
            result['logits'] = torch.stack(logits, -1)
            if hidden:
                result['hidden'] = tuple(representations)
        return result


def requirements(condition):
    return dict(scores=condition in ('csa', 'item_prefix_joint', 'score_025', 'score_mean', 'score_075',
                                    'score_degree', 'gate_degree', 'gate_user', 'platt_mean'),
                hidden=condition.startswith('hidden_'),
                users=condition == 'gate_user' or condition.startswith('shadow_'))

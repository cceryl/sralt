"""Frozen expert fusion. These modules own only the new fusion parameters."""

import torch
from torch import nn
from torch.nn import functional as F


def classifier(width, dropout=.2):
    return nn.Sequential(nn.Linear(width, 128), nn.LeakyReLU(.2),
                         nn.Dropout(dropout), nn.Linear(128, 1))


class FrozenScoreEnsemble(nn.Module):
    needs_attack = True

    def __init__(self, alpha=.5, heuristic=False, tau=1.):
        super().__init__()
        self.alpha, self.heuristic, self.tau = alpha, heuristic, tau

    def forward(self, evidence):
        probabilities = evidence['logits'].sigmoid()
        if self.heuristic:
            degree = evidence['degrees']
            alpha = .1 + .8 * (degree / (degree + self.tau)).prod(-1)
        else:
            alpha = self.alpha
        return alpha * probabilities[:, 0] + (1 - alpha) * probabilities[:, 1]


class DegreeGatedEnsemble(nn.Module):
    needs_attack = True

    def __init__(self, user_dims=None):
        super().__init__()
        self.projections = (nn.ModuleList(nn.Linear(dim, 16) for dim in user_dims)
                            if user_dims is not None else None)
        self.gate = nn.Sequential(nn.Linear(4 + (96 if user_dims else 0), 32),
                                  nn.LeakyReLU(.2), nn.Dropout(.1), nn.Linear(32, 1))

    def forward(self, evidence):
        features = [evidence['degree_features']]
        if self.projections is not None:
            for project, users in zip(self.projections, evidence['users'], strict=True):
                left, right = project(users).unbind(0)
                features.extend([left + right, (left - right).abs(), left * right])
        alpha = self.gate(torch.cat(features, -1)).squeeze(-1).sigmoid()
        p = evidence['logits'].detach().sigmoid()
        return alpha * p[:, 0] + (1 - alpha) * p[:, 1]


class PlattMean(nn.Module):
    needs_attack = True

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.full((2,), .54132485))  # softplus = 1
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, evidence):
        return (F.softplus(self.scale) * evidence['logits'].detach() + self.bias).sigmoid().mean(-1)


class DegreeOnly(nn.Module):
    needs_attack = False

    def __init__(self):
        super().__init__()
        self.head = classifier(4)

    def forward(self, evidence):
        return self.head(evidence['degree_features']).squeeze(-1).sigmoid()


class AttackRepresentationFusion(nn.Module):
    needs_attack = True

    def __init__(self, dims, project=False, degree=False, only=None):
        super().__init__()
        self.indices = (only,) if only is not None else (0, 1)
        self.degree = degree
        self.projections = nn.ModuleList(nn.Linear(dims[i], 64) if project else nn.Identity()
                                         for i in self.indices)
        self.head = classifier(sum(64 if project else dims[i] for i in self.indices) + 4 * degree)

    def forward(self, evidence):
        parts = [project(F.normalize(evidence['hidden'][i], dim=-1))
                 for i, project in zip(self.indices, self.projections, strict=True)]
        if self.degree:
            parts.append(evidence['degree_features'])
        return self.head(torch.cat(parts, -1)).squeeze(-1).sigmoid()


class ShadowUserFusion(nn.Module):
    needs_attack = False

    def __init__(self, dims, project=False, degree=False, only=None):
        super().__init__()
        self.indices = (only,) if only is not None else (0, 1)
        self.degree = degree
        self.projections = nn.ModuleList(nn.Linear(dims[i], 64) if project else nn.Identity()
                                         for i in self.indices)
        self.head = classifier(4 * sum(64 if project else dims[i] for i in self.indices) + 4 * degree)

    def forward(self, evidence):
        parts = []
        for i, project in zip(self.indices, self.projections, strict=True):
            users = evidence['users'][i]
            # Normalize the two native d-dimensional blocks separately.
            users = F.normalize(users.reshape(*users.shape[:-1], 2, -1), dim=-1).flatten(-2)
            parts.append(project(users))
        left, right = torch.cat(parts, -1).unbind(0)
        features = [left, right, (left - right).abs(), left * right]
        if self.degree:
            features.append(evidence['degree_features'])
        return self.head(torch.cat(features, -1)).squeeze(-1).sigmoid()


def make_fusion(condition, user_dims, hidden_dims, tau):
    fixed = {'csa': 1., 'item_prefix_joint': 0., 'score_025': .25,
             'score_mean': .5, 'score_075': .75}
    if condition in fixed:
        return FrozenScoreEnsemble(fixed[condition])
    if condition == 'score_degree':
        return FrozenScoreEnsemble(heuristic=True, tau=tau)
    if condition in ('gate_degree', 'gate_user'):
        return DegreeGatedEnsemble(user_dims if condition == 'gate_user' else None)
    if condition == 'platt_mean':
        return PlattMean()
    if condition == 'degree_only':
        return DegreeOnly()
    if condition.startswith('hidden_'):
        only = {'hidden_csa_only': 0, 'hidden_prefix_only': 1}.get(condition)
        return AttackRepresentationFusion(hidden_dims, project='project' in condition,
                                          degree=condition.endswith('_degree'), only=only)
    if condition.startswith('shadow_'):
        only = {'shadow_csa_only': 0, 'shadow_prefix_only': 1}.get(condition)
        return ShadowUserFusion(user_dims, project='project' in condition,
                                degree=condition.endswith('_degree'), only=only)
    raise ValueError(f'Unknown fusion condition: {condition}')

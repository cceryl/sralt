"""Independent recommendation objectives for prefix and user-code social graphs."""

import torch

from .item_prefix_csa import ItemPrefixCSAShadow, FusedSocialBranch
from sralt.utils import set_seed


class ItemPrefixCSAIndependentShadow(ItemPrefixCSAShadow):
    def __init__(self, num_users, num_items, **kwargs):
        super().__init__(num_users, num_items, **kwargs)
        if self.graph_mode != 'dual':
            raise ValueError('Independent social optimization requires both separate graphs.')
        self.user_social_branch = self.new_user_social_branch()

    def new_social_branch(self):
        # Keep this branch in the parent's prefix preparation/cache interface.
        return FusedSocialBranch(self.num_users, self.num_items, self.embed_dim, self.num_layers,
            self.user_num_quantizers * self.user_codebook_size, graph_mode='prefix', fusion='mean')

    def new_user_social_branch(self):
        return FusedSocialBranch(self.num_users, self.num_items, self.embed_dim, self.num_layers,
            self.user_num_quantizers * self.user_codebook_size, graph_mode='user', fusion='mean')

    def train_model(self, *, seed, deterministic, **kwargs):
        self.independent_social_seed = seed + 2
        self.independent_deterministic = deterministic
        super().train_model(seed=seed, deterministic=deterministic, **kwargs)

    def train_social(self, graphs, train_loader, val_loader, num_epochs, optimizer, metrics, early_stopping, writer):
        # Neither branch's training/early stopping updates or selects the other.
        set_seed(self.independent_social_seed, deterministic=self.independent_deterministic)
        self.user_social_branch = self.new_user_social_branch().to(self.train_user_codes.device)
        super().train_social(graphs, train_loader, val_loader, num_epochs, optimizer, metrics,
            early_stopping, writer, module=self.social_branch, branch_name='social_prefix')
        set_seed(self.independent_social_seed + 1, deterministic=self.independent_deterministic)
        super().train_social(graphs, train_loader, val_loader, num_epochs, optimizer, metrics,
            early_stopping, writer, module=self.user_social_branch, branch_name='social_user')
        for name in ('social_prefix', 'social_user'):
            self.branch_training[name]['regularization'] = (
                'sampled raw users and the branch-specific positive/negative item rows / batch size; '
                'code tables and prefix projection excluded')

    def social_embeddings(self, graphs):
        # Dot products give the sum of independently trained scores for the
        # recommendation report only. No such combined score is optimized.
        return torch.cat([self.social_branch(graphs), self.user_social_branch(graphs)], dim=-1)

    def social_user_views(self, graphs):
        return (self.social_branch.user_views(graphs)[0],
                self.user_social_branch.user_views(graphs)[1])

    def social_parameter_count(self):
        return sum(p.numel() for module in (self.social_branch, self.user_social_branch)
                   for p in module.parameters())

    @torch.no_grad()
    def get_attack_payload(self, data):
        features, payload = super().get_attack_payload(data)
        self.last_diagnostics.update(fusion='none', social_optimization='independent_bpr',
            feature_layout=['behavior', 'prefix_social', 'user_social', 'item_mean', 'item_variance'],
            recommendation_readout='sum_of_independently_optimized_scores')
        norms = self.last_diagnostics['mean_norms']
        norms['concatenated_social'] = norms.pop('fused')
        # The parent already concatenates [behavior, 2d social, mean, variance].
        # sparse_social similarly contains both sparse social blocks in order.
        return features, payload

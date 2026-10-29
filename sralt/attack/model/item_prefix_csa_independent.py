"""Five continuous blocks with separate normalized social endpoints."""

import torch
from torch.nn import functional as F

from .item_prefix_csa import ItemPrefixCSAAttack


class ItemPrefixCSAIndependentAttack(ItemPrefixCSAAttack):
    feature_blocks = 5

    def endpoint_features(self, users, data, sparse_mask=None):
        blocks = data.user_features[users].reshape(*users.shape, self.feature_blocks, -1).clone()
        if sparse_mask is not None:
            sparse = data.side_features['sparse_social'][users].reshape(*users.shape, 2, -1)
            blocks[..., 1:3, :] = torch.where(sparse_mask[..., None, None], sparse, blocks[..., 1:3, :])
        return F.normalize(blocks, p=2, dim=-1) if self.normalize_features else blocks

    def __repr__(self):
        return super().__repr__().replace('ItemPrefixCSAAttack(', 'ItemPrefixCSAIndependentAttack(', 1)

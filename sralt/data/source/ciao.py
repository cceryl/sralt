import os
import logging
from dataclasses import dataclass

import torch
import scipy.io
from numpy.typing import NDArray
from torch_geometric.utils import to_undirected, coalesce

from . import register_dataset
from .data import Data


@register_dataset('ciao')
class CiaoData(Data):
    """
    The Ciao social recommendation dataset.
    """

    @dataclass
    class Config(Data.Config):
        # The minimum rating to consider an interaction as positive
        min_rating: int

        # The minimum number of social connections a user must have to be included
        min_social_connections: int

        # The minimum number of positive item interactions a user must have to be included
        min_user_interactions: int

        # The minimum number of positive user interactions an item must have to be included
        min_item_interactions: int

    def load(self, config: Data.Config):
        """
        Loads and preprocesses the Ciao dataset.

        :param config: Configuration parameters for loading the dataset.
        """
        assert isinstance(config, CiaoData.Config)
        logging.info(f'Loading Ciao dataset from {config.data_dir}...')

        # Rows of (user_id, user_id)
        social_mat: NDArray = scipy.io.loadmat(os.path.join(config.data_dir, 'trustnetwork.mat'))['trustnetwork']
        # Rows of (user_id, item_id, category_id, rating, helpfulness)
        rating_mat: NDArray = scipy.io.loadmat(os.path.join(config.data_dir, 'rating.mat'))['rating']

        # Filter bad interactions and convert to edge indices
        social_data = torch.from_numpy(social_mat).long()
        social_edge_index = to_undirected(social_data.t())

        rating_data = torch.from_numpy(rating_mat).long()
        behavioral_edge_index = coalesce(rating_data[rating_data[:, 3] >= config.min_rating][:, :2].t())

        # Iteratively filter users and items based on minimum interactions and social connections
        max_user_id = int(torch.cat([behavioral_edge_index[0], social_edge_index[0]]).max().item()) + 1
        max_item_id = int(behavioral_edge_index[1].max().item()) + 1
        while True:
            num_social_edges = social_edge_index.size(1)
            num_behavioral_edges = behavioral_edge_index.size(1)

            user_social_counts = torch.bincount(social_edge_index[0], minlength=max_user_id)
            user_interaction_counts = torch.bincount(behavioral_edge_index[0], minlength=max_user_id)
            item_interaction_counts = torch.bincount(behavioral_edge_index[1], minlength=max_item_id)

            valid_users = (
                (user_social_counts >= config.min_social_connections) &
                (user_interaction_counts >= config.min_user_interactions)
            ).nonzero().flatten()
            valid_items = (item_interaction_counts >= config.min_item_interactions).nonzero().flatten()

            social_mask = torch.isin(social_edge_index[0], valid_users) & torch.isin(social_edge_index[1], valid_users)
            social_edge_index = social_edge_index[:, social_mask]
            behavioral_mask = (
                torch.isin(behavioral_edge_index[0], valid_users) &
                torch.isin(behavioral_edge_index[1], valid_items)
            )
            behavioral_edge_index = behavioral_edge_index[:, behavioral_mask]

            if social_edge_index.size(1) == num_social_edges and behavioral_edge_index.size(1) == num_behavioral_edges:
                break

        # Re-index users and items
        user_ids: torch.Tensor = torch.cat([behavioral_edge_index[0], social_edge_index[0]]).unique()
        item_ids: torch.Tensor = behavioral_edge_index[1].unique()

        self.num_users = len(user_ids)
        self.num_items = len(item_ids)

        max_user_id = int(user_ids.max().item())
        user_id_map = torch.full((max_user_id + 1,), -1, dtype=torch.long)
        user_id_map[user_ids] = torch.arange(self.num_users, dtype=torch.long)

        max_item_id = int(item_ids.max().item())
        item_id_map = torch.full((max_item_id + 1,), -1, dtype=torch.long)
        item_id_map[item_ids] = torch.arange(self.num_items, dtype=torch.long) + self.num_users

        behavioral_edge_index = coalesce(torch.stack([
            user_id_map[behavioral_edge_index[0]],
            item_id_map[behavioral_edge_index[1]]
        ], dim=0))

        self.social_edge_index = coalesce(torch.stack([
            user_id_map[social_edge_index[0]],
            user_id_map[social_edge_index[1]]
        ], dim=0))

        # Split behavioral edges into train/val/test sets
        val_mask = torch.zeros_like(behavioral_edge_index[0], dtype=torch.bool)
        test_mask = torch.zeros_like(behavioral_edge_index[0], dtype=torch.bool)

        for u in range(self.num_users):
            user_edge_idx = (behavioral_edge_index[0] == u).nonzero(as_tuple=True)[0]
            num_interactions = user_edge_idx.numel()
            if num_interactions < 3:
                raise ValueError(f'User {u} has less than 3 interactions.')
            perm = torch.randperm(num_interactions)
            val_edge = user_edge_idx[perm[0]]
            test_edge = user_edge_idx[perm[1]]
            val_mask[val_edge] = True
            test_mask[test_edge] = True

        train_mask = ~(val_mask | test_mask)

        self.behavioral_train_edge_index = behavioral_edge_index[:, train_mask]
        self.behavioral_val_edge_index = behavioral_edge_index[:, val_mask]
        self.behavioral_test_edge_index = behavioral_edge_index[:, test_mask]

        logging.info(
            f'Ciao dataset preprocessing completed with '
            f'{self.num_users} users, '
            f'{self.num_items} items, '
            f'{self.num_social_edges} social edges, '
            f'{self.num_behavioral_edges} behavioral edges.'
        )

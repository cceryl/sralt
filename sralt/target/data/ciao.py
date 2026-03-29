import os
import logging

import torch
import scipy.io
import numpy as np
from numpy.typing import NDArray
from torch_geometric.utils import to_undirected, coalesce

from sralt.target.data import TargetData
from sralt.utils import leave_one_out_split


class Ciao(TargetData):
    """
    The Ciao social recommendation dataset.
    """

    def __init__(
        self, name: str, directory: str,
        min_rating: int, min_social_connections: int, min_user_interactions: int, min_item_interactions: int
    ):
        """
        Load and preprocess the Ciao dataset.

        :param name: The name of the dataset.
        :param directory: The directory where the dataset is stored.
        :param min_rating: The minimum rating to consider an interaction as positive.
        :param min_social_connections: The minimum number of social connections a user must have to be included.
        :param min_user_interactions: The minimum number of positive item interactions a user must have to be included.
        :param min_item_interactions: The minimum number of positive user interactions an item must have to be included.
        """
        super().__init__(name, directory)
        logging.info(f'Loading Ciao dataset from {directory}...')

        self.min_rating = min_rating
        self.min_social_connections = min_social_connections
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions

        # Rows of (user_id, user_id)
        social_mat: NDArray = scipy.io.loadmat(os.path.join(directory, 'trustnetwork.mat'))['trustnetwork']
        social_mat = social_mat.astype(np.int64)
        # Rows of (user_id, item_id, category_id, rating, helpfulness)
        rating_mat: NDArray = scipy.io.loadmat(os.path.join(directory, 'rating.mat'))['rating']
        rating_mat = rating_mat.astype(np.int64)

        # Filter bad interactions and convert to edge indices
        social_data = torch.from_numpy(social_mat).long()
        social_edge_index = to_undirected(social_data.t())

        rating_data = torch.from_numpy(rating_mat).long()
        behavioral_edge_index = coalesce(rating_data[rating_data[:, 3] >= min_rating][:, :2].t())

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
                (user_social_counts >= min_social_connections) &
                (user_interaction_counts >= min_user_interactions)
            ).nonzero().flatten()
            valid_items = (item_interaction_counts >= min_item_interactions).nonzero().flatten()

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
        train_edge_index, val_edge_index, test_edge_index = leave_one_out_split(behavioral_edge_index, self.num_users)
        self.behavioral_train_edge_index = train_edge_index
        self.behavioral_val_edge_index = val_edge_index
        self.behavioral_test_edge_index = test_edge_index

        logging.info(
            f'Ciao dataset preprocessing completed with '
            f'{self.num_users} users, '
            f'{self.num_items} items, '
            f'{self.num_social_edges} social edges, '
            f'{self.num_behavioral_edges} behavioral edges.'
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'directory={self.directory}, '
            f'min_rating={self.min_rating}, '
            f'min_social_connections={self.min_social_connections}, '
            f'min_user_interactions={self.min_user_interactions}, '
            f'min_item_interactions={self.min_item_interactions})'
        )

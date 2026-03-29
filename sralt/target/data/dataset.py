from typing import Literal

import torch
from torch.utils.data import Dataset

from .data import TargetData
from sralt.utils import sample_negative_items


class TrainTargetDataset(Dataset):
    """
    Dataset class for training data in target models.
    """

    def __init__(self, data: TargetData):
        """
        Initializes the dataset with the provided data object.

        :param data: An instance of the Data class containing graph data.
        """
        self.data = data

        self.users, self.positive_items = data.behavioral_train_edge_index
        self.negative_items: torch.Tensor = torch.tensor([])

        self.access_counter = 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the data sample at the specified index.

        :param index: The index of the sample to retrieve.
        :return: A tuple of (user, positive_item, negative_item) at the specified index, all tensors are scalar.
        """
        if self.access_counter == 0:
            self.negative_items = self.sample_negative_items()

        self.access_counter = (self.access_counter + 1) % len(self)

        return self.users[index], self.positive_items[index], self.negative_items[index]

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: The length of the dataset.
        """
        return self.users.size(0)

    def sample_negative_items(self) -> torch.Tensor:
        """
        Samples an negative item for each positive user-item interaction in the training data.

        :return: A tensor of shape (num_edges,) containing negative item samples.
        """
        return sample_negative_items(
            users=self.users,
            edge_index=self.data.behavioral_train_edge_index,
            num_users=self.data.num_users,
            num_items=self.data.num_items
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(data={self.data})'


class EvalTargetDataset(Dataset):
    """
    Dataset class for evaluation data in target models.
    """

    def __init__(self, data: TargetData, split: Literal['val', 'test'] = 'val', num_negatives: int = 999):
        """
        Initializes the evaluation dataset with the provided data object and number of negative samples.

        :param data: An instance of the Data class containing graph data.
        :param split: The data split to use ('val' or 'test').
        :param num_negatives: The number of negative samples to generate for each positive interaction.
        """
        self.data = data
        self.split = split
        self.num_negatives = num_negatives

        self.users, self.positive_items = data.behavioral_val_edge_index if self.split == 'val' else data.behavioral_test_edge_index
        self.negative_items = self.sample_negative_items()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the data sample at the specified index.

        :param index: The index of the sample to retrieve.
        :return: A tuple of (user, positive_item, negative_items) at the specified index, where negative_items is of shape
                 (num_negatives,) and the others are scalars.
        """
        return self.users[index], self.positive_items[index], self.negative_items[index]

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: The length of the dataset.
        """
        return self.users.size(0)

    def sample_negative_items(self) -> torch.Tensor:
        """
        Samples negative items for each positive user-item interaction in the evaluation data.

        :return: A tensor of shape (num_edges, num_negatives) containing negative item samples.
        """
        users = self.users.repeat_interleave(self.num_negatives)

        if self.split == 'val':
            conflict_edges = torch.cat([
                self.data.behavioral_train_edge_index,
                self.data.behavioral_val_edge_index
            ], dim=1)
        else:
            conflict_edges = self.data.behavioral_edge_index

        return sample_negative_items(
            users=users,
            edge_index=conflict_edges,
            num_users=self.data.num_users,
            num_items=self.data.num_items
        ).view(self.users.size(0), self.num_negatives)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(data={self.data}, split={self.split}, num_negatives={self.num_negatives})'

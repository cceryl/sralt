from abc import ABC, abstractmethod

import torch
import numpy as np
from torch.utils.data import Dataset

from .data import TargetData


class BaseTargetDataset(Dataset, ABC):
    """
    Base dataset class for target social recommendation models.
    """

    def __init__(self, data: TargetData):
        """
        Initializes the dataset with the provided data object.

        :param data: An instance of the Data class containing graph data.
        """
        self.data = data

        self.users, self.positive_items = self.get_positive_edge_index()
        self.negative_items: torch.Tensor = torch.tensor([])

        self.access_counter = 0

    def __getitem__(self, index: int | slice | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the data sample at the specified indices.

        :param index: Indices of the data sample to retrieve.
        :return: A tuple of (user, positive_item, negative_item) at the specified indices.
        """
        if self.access_counter == 0:
            self.negative_items = self.sample_negative_items()

        self.access_counter = (self.access_counter + 1) % len(self)
        torch.Tensor.__getitem__

        return self.users[index], self.positive_items[index], self.negative_items[index]

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: The length of the dataset.
        """
        return self.users.size(0)

    @abstractmethod
    def get_positive_edge_index(self) -> torch.Tensor:
        """
        Retrieves the positive user-item edge indices.

        :return: An user-item edge index tensor of shape (2, num_edges).
        """
        pass

    @abstractmethod
    def sample_negative_items(self) -> torch.Tensor:
        """
        Samples negative items for each user in the dataset.

        :return: A tensor containing negative item samples for each user.
        """
        pass


class TrainTargetDataset(BaseTargetDataset):
    """
    Dataset class for training data in target social recommendation models.
    """

    def get_positive_edge_index(self) -> torch.Tensor:
        """
        Retrieves the positive user-item edge indices from the training data.

        :return: An user-item edge index tensor of shape (2, num_edges).
        """
        return self.data.behavioral_train_edge_index

    def sample_negative_items(self) -> torch.Tensor:
        """
        Samples an negative item for each positive user-item interaction in the training data.

        :return: A tensor of shape (num_edges,) containing negative item samples.
        """
        edge_index = self.data.behavioral_train_edge_index
        num_edges = edge_index.size(1)

        num_users = self.data.num_users
        num_items = self.data.num_items

        users, pos_items = edge_index.cpu()
        pos_idx = users * num_items + pos_items

        neg_items = torch.randint(num_items, (num_edges,)) + num_users
        neg_idx = users * num_items + neg_items

        conflict_mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        conflicts = conflict_mask.nonzero(as_tuple=False).view(-1)
        while conflicts.numel() > 0:
            replacement = torch.randint(num_items, (conflicts.size(0),)) + num_users
            neg_items[conflicts] = replacement
            neg_idx = users[conflicts] * num_items + replacement

            conflict_mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
            conflicts = conflicts[conflict_mask]

        return neg_items.to(self.data.device)

    def __getitem__(self, index: int | slice | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the data sample at the specified indices.
        The returned tensors are all scalar if index is an integer, or of shape (batch_size,) if index is a slice or tensor.

        :param index: Indices of the data sample to retrieve.
        :return: A tuple of (user, positive_item, negative_item) at the specified indices
        """
        return super().__getitem__(index)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(data={self.data})'


class EvalTargetDataset(BaseTargetDataset):
    """
    Dataset class for evaluation data in target social recommendation models.
    """

    def __init__(self, data: TargetData, split: str = 'val', num_negative_samples: int = 999):
        """
        Initializes the evaluation dataset with the provided data object and number of negative samples.

        :param data: An instance of the Data class containing graph data.
        :param split: The data split to use ('val' or 'test').
        :param num_negative_samples: The number of negative samples to generate for each positive interaction.
        """
        assert split in ['val', 'test'], 'split must be either "val" or "test"'
        self.split = split
        self.num_negative_samples = num_negative_samples

        super().__init__(data)

    def get_positive_edge_index(self) -> torch.Tensor:
        """
        Retrieves the positive user-item edge indices from the evaluation data.

        :return: An user-item edge index tensor of shape (2, num_edges).
        """
        return self.data.behavioral_val_edge_index if self.split == 'val' else self.data.behavioral_test_edge_index

    def sample_negative_items(self) -> torch.Tensor:
        """
        Samples negative items for each positive user-item interaction in the evaluation data.

        :return: A tensor of shape (num_edges, num_negative_samples) containing negative item samples.
        """
        edge_index = self.get_positive_edge_index()
        num_edges = edge_index.size(1)

        num_users = self.data.num_users
        num_items = self.data.num_items

        placeholder = torch.empty((2, 0), dtype=torch.long, device=self.data.device)
        masked_edges = torch.cat([
            self.data.behavioral_train_edge_index,
            self.data.behavioral_val_edge_index,
            self.data.behavioral_test_edge_index if self.split == 'test' else placeholder
        ], dim=1)
        masked_users, masked_pos_items = masked_edges.cpu()
        pos_idx = masked_users * num_items + masked_pos_items

        users = edge_index[0].cpu().repeat_interleave(self.num_negative_samples)
        neg_items = torch.randint(num_items, (num_edges * self.num_negative_samples,)) + num_users
        neg_idx = users * num_items + neg_items

        conflict_mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        conflicts = conflict_mask.nonzero(as_tuple=False).view(-1)
        while conflicts.numel() > 0:
            replacement = torch.randint(num_items, (conflicts.size(0),)) + num_users
            neg_items[conflicts] = replacement
            neg_idx = users[conflicts] * num_items + replacement

            conflict_mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
            conflicts = conflicts[conflict_mask]

        return neg_items.view(num_edges, self.num_negative_samples).to(self.data.device)

    def __getitem__(self, index: int | slice | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the data sample at the specified indices.
        The returned tensors are respectively scalar, scalar, and of shape (num_negative_samples,) if index is an integer,
        or of shape (batch_size,), (batch_size,), and (batch_size, num_negative_samples) if index is a slice or tensor.

        :param index: Indices of the data sample to retrieve.
        :return: A tuple of (user, positive_item, negative_items) at the specified indices
        """
        return super().__getitem__(index)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(data={self.data}, split={self.split}, num_negative_samples={self.num_negative_samples})'

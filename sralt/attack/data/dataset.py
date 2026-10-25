from typing import Literal

import torch
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected, negative_sampling

from .data import AttackData


def sample_negative_pairs(edge_index: torch.Tensor, num_users: int, num_samples: int) -> torch.Tensor:
    """
    Sample distinct undirected user pairs, represented once with u < v.
    
    :param edge_index: The undirected edge index of shape (2, num_edges).
    :param num_users: The total number of users in the graph.
    :param num_samples: The number of negative samples to generate.
    :return: A tensor of shape (2, num_samples) containing the sampled negative user pairs.
    """
    edges = negative_sampling(
        edge_index=to_undirected(edge_index),
        num_nodes=num_users,
        num_neg_samples=2 * num_samples,
        method='dense',
        force_undirected=True
    )
    return edges[:, edges[0] < edges[1]]


class TrainAttackDataset(Dataset):
    """
    Dataset class for training data in attack models.
    """

    def __init__(self, attack_data: AttackData, negative_ratio: float):
        """
        Initialize the training attack dataset.

        :param attack_data: The attack data containing social graph edges.
        :param negative_ratio: The ratio of negative samples to generate relative to positive samples.
        """
        self.attack_data = attack_data
        self.negative_ratio = negative_ratio

        self.pos_edge_index = attack_data.train_social_edge_index
        self.neg_edge_index = torch.tensor([], device=attack_data.device)
        self.num_negatives = int(self.negative_ratio * self.pos_edge_index.size(1))

        self.edge_index = torch.tensor([], device=attack_data.device)
        self.labels = torch.cat(
            [torch.ones(self.pos_edge_index.size(1)), torch.zeros(self.num_negatives)]
        ).long().to(attack_data.device)

        self.access_counter = 0

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: The length of the dataset.
        """
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve the positive and negative samples at the specified index.

        :param idx: The index of the sample to retrieve.
        :return: A tuple containing the two user indices and the label.
        """
        if self.access_counter == 0:
            self.neg_edge_index = sample_negative_pairs(
                self.pos_edge_index, self.attack_data.num_users, self.num_negatives
            )
            self.edge_index = torch.cat([self.pos_edge_index, self.neg_edge_index], dim=1)

        self.access_counter = (self.access_counter + 1) % len(self)

        return self.edge_index[0, idx], self.edge_index[1, idx], self.labels[idx]

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'attack_data={repr(self.attack_data)}, '
            f'negative_ratio={self.negative_ratio})'
        )


class EvalAttackDataset(Dataset):
    """
    Dataset class for evaluating data in attack models.
    """

    def __init__(self, attack_data: AttackData, split: Literal['val', 'test'] = 'val'):
        """
        Initialize the evaluation attack dataset.

        :param attack_data: The attack data containing social graph edges.
        :param split: The data split to use ('val' or 'test').
        """
        self.attack_data = attack_data
        self.split = split

        pos_edge_index = attack_data.val_social_edge_index if split == 'val' else attack_data.test_social_edge_index
        neg_edge_index = sample_negative_pairs(
            edge_index=attack_data.public_social_edge_index if split == 'val' else attack_data.target_social_edge_index,
            num_users=attack_data.num_users,
            num_samples=pos_edge_index.size(1)
        )

        self.edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        self.labels = torch.cat(
            [torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))]
        ).long().to(attack_data.device)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: The length of the dataset.
        """
        return self.edge_index.size(1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve the positive and negative samples at the specified index.

        :param idx: The index of the sample to retrieve.
        :return: A tuple containing the two user indices and the label.
        """
        return self.edge_index[0, idx], self.edge_index[1, idx], self.labels[idx]

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'attack_data={repr(self.attack_data)}, '
            f'split={self.split})'
        )

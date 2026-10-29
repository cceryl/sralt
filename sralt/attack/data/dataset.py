from collections.abc import Iterator
from typing import Literal

import torch
from torch.utils.data import Dataset, Sampler
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

        # Auxiliary Attack views can partition direct supervision without
        # changing the native graph, feature payload or formal expert path.
        split = getattr(attack_data, 'attack_training_split', None)
        self.pos_edge_index = (attack_data.train_social_edge_index if split is None
                               else split['positive_edge_index'].to(attack_data.device))
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
            sampler = getattr(self.attack_data, 'sample_attack_training_negatives', None)
            self.neg_edge_index = (sample_negative_pairs(
                self.pos_edge_index, self.attack_data.num_users, self.num_negatives
            ) if sampler is None else sampler(self.num_negatives))
            self.edge_index = torch.cat([self.pos_edge_index, self.neg_edge_index], dim=1)

        self.access_counter = (self.access_counter + 1) % len(self)

        return self.edge_index[0, idx], self.edge_index[1, idx], self.labels[idx]

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'attack_data={repr(self.attack_data)}, '
            f'negative_ratio={self.negative_ratio})'
        )


class BalancedAttackBatchSampler(Sampler[list[int]]):
    """Use every sample once per epoch, with equal classes in each batch.

    A tail smaller than half the nominal batch size is merged into
    the preceding batch.
    Negative edges are still resampled by the dataset once
    per full traversal. Auxiliary OOF views additionally partition supervision;
    the ordinary expert path uses only training positives for exclusion.
    """

    def __init__(self, dataset: TrainAttackDataset, batch_size: int):
        self.num_positive = dataset.pos_edge_index.size(1)
        if dataset.num_negatives != self.num_positive or batch_size < 2 or batch_size % 2:
            raise ValueError('Balanced attack batches require 1:1 classes and an even batch size >= 2.')

        half_batch_size = batch_size // 2
        full, remainder = divmod(self.num_positive, half_batch_size)
        self.positive_batch_sizes = [half_batch_size] * full
        if remainder:
            if self.positive_batch_sizes and 2 * remainder < half_batch_size:
                self.positive_batch_sizes[-1] += remainder
            else:
                self.positive_batch_sizes.append(remainder)

    def __len__(self) -> int:
        return len(self.positive_batch_sizes)

    def __iter__(self) -> Iterator[list[int]]:
        positives = torch.randperm(self.num_positive).tolist()
        negatives = (torch.randperm(self.num_positive) + self.num_positive).tolist()
        start = 0
        for size in self.positive_batch_sizes:
            batch = positives[start:start + size] + negatives[start:start + size]
            start += size
            order = torch.randperm(len(batch)).tolist()
            yield [batch[index] for index in order]


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
        fixed = getattr(attack_data, 'fixed_eval_pairs', {}).get(split)
        if fixed is not None:
            edges, labels = fixed['edge_index'], fixed['labels']
            if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long
                    or labels.ndim != 1 or edges.shape[1] != labels.numel()
                    or not torch.all((labels == 0) | (labels == 1))):
                raise ValueError('Invalid fixed evaluation edges/labels.')
            if not torch.equal(edges[:, labels == 1].cpu(), pos_edge_index.cpu()):
                raise ValueError('Fixed evaluation positives differ from the attack split.')
            self.edge_index = edges.to(attack_data.device)
            self.labels = labels.long().to(attack_data.device)
            return
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

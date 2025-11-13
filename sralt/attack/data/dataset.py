import torch
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected, negative_sampling

from .data import AttackData


class TrainAttackDataset(Dataset):
    """
    Dataset for training social connection attacks.
    """

    def __init__(self, attack_data: AttackData):
        """
        Initialize the training attack dataset.

        :param attack_data: The attack data containing social graph edges.
        """
        self.attack_data = attack_data

        self.pos_edge_index = attack_data.train_social_edge_index
        self.neg_edge_index = torch.tensor([], device=attack_data.device)
        self.edge_index = torch.tensor([], device=attack_data.device)
        self.labels = torch.cat(
            [torch.ones(self.pos_edge_index.size(1)), torch.zeros(self.pos_edge_index.size(1))]
        ).long().to(attack_data.device)

        self.access_counter = 0

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: The length of the dataset.
        """
        return self.pos_edge_index.size(1) * 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve the positive and negative samples at the specified index.

        :param idx: The index of the sample to retrieve.
        :return: A tuple containing the two user indices and the label.
        """
        if self.access_counter == 0:
            self.neg_edge_index = negative_sampling(
                edge_index=to_undirected(self.pos_edge_index),
                num_nodes=self.attack_data.num_users,
                num_neg_samples=self.pos_edge_index.size(1),
                method='dense',
                force_undirected=False
            )
            self.edge_index = torch.cat([self.pos_edge_index, self.neg_edge_index], dim=1)

        self.access_counter = (self.access_counter + 1) % len(self)

        return self.edge_index[0, idx], self.edge_index[1, idx], self.labels[idx]

    def __repr__(self) -> str:
        return f'TrainAttackDataset(attack_data={repr(self.attack_data)})'


class EvalAttackDataset(Dataset):
    """
    Dataset for evaluating social connection attacks.
    """

    def __init__(self, attack_data: AttackData, split: str = 'val'):
        """
        Initialize the evaluation attack dataset.

        :param attack_data: The attack data containing social graph edges.
        :param split: The data split to use ('val' or 'test').
        """
        assert split in ['val', 'test'], "Split must be either 'val' or 'test'."

        self.attack_data = attack_data
        self.split = split

        self.pos_edge_index = attack_data.val_social_edge_index if split == 'val' else attack_data.test_social_edge_index
        self.neg_edge_index = negative_sampling(
            edge_index=attack_data.shadow_social_edge_index if split == 'val' else attack_data.target_social_edge_index,
            num_nodes=attack_data.num_users,
            num_neg_samples=self.pos_edge_index.size(1),
            method='dense',
            force_undirected=False
        )
        self.edge_index = torch.cat([self.pos_edge_index, self.neg_edge_index], dim=1)
        self.labels = torch.cat(
            [torch.ones(self.pos_edge_index.size(1)), torch.zeros(self.neg_edge_index.size(1))]
        ).long().to(attack_data.device)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: The length of the dataset.
        """
        return self.pos_edge_index.size(1) * 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve the positive and negative samples at the specified index.

        :param idx: The index of the sample to retrieve.
        :return: A tuple containing the two user indices and the label.
        """
        return self.edge_index[0, idx], self.edge_index[1, idx], self.labels[idx]

    def __repr__(self) -> str:
        return f'EvalAttackDataset(attack_data={repr(self.attack_data)}, split={self.split})'

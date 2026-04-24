import torch
from torch.utils.data import Dataset
from torch_geometric.utils import coalesce, degree

from sralt.target import TargetData, TrainTargetDataset


@torch.no_grad()
def sample_social_relations(
    users: torch.Tensor,
    edge_index: torch.Tensor,
    num_users: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample positive and negative social relations for the given users.

    :param users: A tensor of user indices of shape (num_samples,).
    :param edge_index: Undirected edge index tensor of shape (2, num_edges) representing social relations.
    :param num_users: Total number of users.
    :return: A tuple of tensors (positive_peers, negative_peers) each of shape (num_samples,).
    """
    device = users.device
    num_samples = users.size(0)
    users = users.long()

    edge_index = coalesce(edge_index.long(), num_nodes=num_users)
    src, dst = edge_index

    # Positive sampling
    degrees = degree(src, num_nodes=num_users, dtype=torch.long)
    ptr = torch.cat([torch.tensor([0], device=device), degrees.cpu().cumsum(0).to(device)])

    user_starts = ptr[users]
    user_degrees = degrees[users]
    rand_offsets = (torch.rand(num_samples, device=device) * user_degrees).floor().long()

    pos_indices = user_starts + rand_offsets
    pos_samples = dst[pos_indices]

    # Negative sampling
    pos_idx = src * num_users + dst

    neg_samples = torch.randint(0, num_users, (num_samples,), device=device, dtype=torch.long)
    neg_idx = users * num_users + neg_samples

    conflict_mask = torch.isin(neg_idx, pos_idx) | (neg_samples == users)
    conflicts = conflict_mask.nonzero(as_tuple=True)[0]

    while conflicts.numel() > 0:
        replacement = torch.randint(0, num_users, (conflicts.size(0),), device=device, dtype=torch.long)
        neg_samples[conflicts] = replacement
        neg_idx = users[conflicts] * num_users + replacement

        conflict_mask = torch.isin(neg_idx, pos_idx) | (replacement == users[conflicts])
        conflicts = conflicts[conflict_mask]

    return pos_samples, neg_samples


class EGCLDataset(Dataset):
    """
    Dataset class for training EGCL, providing relation samples along with recommendation samples.
    """

    def __init__(self, data: TargetData):
        """
        :param data: An instance of the Data class containing graph data.
        """
        self.rec_dataset = TrainTargetDataset(data)

        self.positive_peers = torch.tensor([])
        self.negative_peers = torch.tensor([])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the data sample at the specified index.

        :param index: The index of the sample to retrieve.
        :return: A tuple of (user, positive_item, negative_item, positive_peer, negative_peer) at the specified index, all tensors are scalar.
        """
        if self.rec_dataset.access_counter == 0:
            self.positive_peers, self.negative_peers = sample_social_relations(
                users=self.rec_dataset.users,
                edge_index=self.rec_dataset.data.social_edge_index,
                num_users=self.rec_dataset.data.num_users,
            )

        user, pos_item, neg_item = self.rec_dataset[index]
        pos_peer, neg_peer = self.positive_peers[index], self.negative_peers[index]

        return user, pos_item, neg_item, pos_peer, neg_peer

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: The length of the dataset.
        """
        return len(self.rec_dataset)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(data={self.rec_dataset.data})'

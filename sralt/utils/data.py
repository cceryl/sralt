import torch
from torch.utils.data import DataLoader as TorchDataLoader


def leave_one_out_split(edge_index: torch.Tensor, num_users: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits user-item interactions into training, validation, and test sets using random leave-one-out strategy.
    For each user, one random interaction is assigned to the validation set and another to the test set, while the rest are used for training.
    All indices must be global, with users in [0, num_users) and items in [num_users, num_users + num_items).

    :param edge_index: Directed user-item edge index tensor of shape (2, num_edges).
    :param num_users: Total number of users.
    :return: A tuple containing training, validation, and test edge indices.
    """
    val_mask = torch.zeros_like(edge_index[0], dtype=torch.bool)
    test_mask = torch.zeros_like(edge_index[0], dtype=torch.bool)

    for u in range(num_users):
        user_edge_idx = (edge_index[0] == u).nonzero(as_tuple=True)[0]
        num_interactions = user_edge_idx.numel()
        if num_interactions < 3:
            raise ValueError(f'User {u} has less than 3 interactions.')
        perm = torch.randperm(num_interactions)
        val_edge = user_edge_idx[perm[0]]
        test_edge = user_edge_idx[perm[1]]
        val_mask[val_edge] = True
        test_mask[test_edge] = True

    train_mask = ~(val_mask | test_mask)

    return edge_index[:, train_mask], edge_index[:, val_mask], edge_index[:, test_mask]


def sample_negative_items(
    users: torch.Tensor,
    edge_index: torch.Tensor,
    num_users: int,
    num_items: int
) -> torch.Tensor:
    """
    Samples negative items for each user in the provided users tensor.
    All indices must be global, with users in [0, num_users) and items in [num_users, num_users + num_items).

    :param users: A tensor of user indices of shape (num_samples,).
    :param edge_index: Directed user-item edge index tensor of shape (2, num_edges) representing positive interactions.
    :param num_users: Total number of users.
    :param num_items: Total number of items.
    :return: A tensor of shape (num_samples,) containing negative item samples for each user.
    """
    device = users.device
    num_samples = users.size(0)

    pos_users, pos_items = edge_index
    pos_idx = pos_users * num_items + pos_items

    neg_items = torch.randint(num_items, (num_samples,), device=device) + num_users
    neg_idx = users * num_items + neg_items

    conflict_mask = torch.isin(neg_idx, pos_idx)
    conflicts = conflict_mask.nonzero(as_tuple=False).view(-1)
    while conflicts.numel() > 0:
        replacement = torch.randint(num_items, (conflicts.size(0,),), device=device) + num_users
        neg_items[conflicts] = replacement
        neg_idx = users[conflicts] * num_items + replacement

        conflict_mask = torch.isin(neg_idx, pos_idx)
        conflicts = conflicts[conflict_mask]

    return neg_items


class DataLoader(TorchDataLoader):
    """
    A custom DataLoader that forbids multiprocessing to avoid negative sampling inconsistency.
    """

    def __new__(cls, *args, **kwargs):
        if 'num_workers' in kwargs and kwargs['num_workers'] > 0:
            raise ValueError('Multiprocessing is forbidden to avoid negative sampling inconsistency.')
        return super().__new__(cls)

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.utils import degree


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


def head_tail_split(num_nodes: int, edge_index: torch.Tensor, tail_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split nodes into head and tail based on their degrees.

    :param num_nodes: Total number of nodes.
    :param edge_index: Undirected edge index tensor of shape (2, num_edges).
    :param tail_ratio: The ratio to be considered as tail nodes.
    :return: A tuple containing head node indices and tail node indices.
    """
    degrees = degree(edge_index[0], num_nodes=num_nodes)
    _, sorted_indices = torch.sort(degrees)

    split_idx = int(num_nodes * tail_ratio)
    tail_indices = sorted_indices[:split_idx]
    head_indices = sorted_indices[split_idx:]

    return head_indices, tail_indices


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
    users, edge_index = users.long(), edge_index.long()

    pos_users, pos_items = edge_index
    pos_idx = pos_users * num_items + pos_items

    neg_items = torch.randint(num_items, (num_samples,), device=device, dtype=torch.long) + num_users
    neg_idx = users * num_items + neg_items

    conflict_mask = torch.isin(neg_idx, pos_idx)
    conflicts = conflict_mask.nonzero(as_tuple=False).view(-1)
    while conflicts.numel() > 0:
        replacement = torch.randint(num_items, (conflicts.size(0),), device=device, dtype=torch.long) + num_users
        neg_items[conflicts] = replacement
        neg_idx = users[conflicts] * num_items + replacement

        conflict_mask = torch.isin(neg_idx, pos_idx)
        conflicts = conflicts[conflict_mask]

    return neg_items


def rebalanced_negative_sampling(
    edge_index: torch.Tensor, num_nodes: int, num_neg_samples: int, alpha: float = 0.5
) -> torch.Tensor:
    """
    Performs rebalanced negative sampling on the given edge index.
    Positive edges and self-loops are excluded from negative samples.
    The node weights are computed as (degree(node) + 1) ^ (-alpha).

    :param edge_index: Undirected positive edge index tensor of shape (2, num_edges).
    :param num_nodes: Total number of nodes.
    :param num_neg_samples: Number of negative samples to generate.
    :param alpha: Rebalancing exponent.
    :return: A tensor of shape (2, num_neg_samples) containing negative edge indices.
    """
    device = edge_index.device
    deg = degree(edge_index[0], num_nodes=num_nodes)
    node_weights = (deg + 1).pow(-alpha)

    # torch.multinomial does not support deterministic algorithms on GPU
    if torch.are_deterministic_algorithms_enabled():
        node_weights = node_weights.cpu()

    pos_u, pos_v = edge_index.long()
    self_loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    exclude_idx = torch.cat([pos_u * num_nodes + pos_v, self_loops * num_nodes + self_loops])

    neg_u = torch.multinomial(node_weights, num_neg_samples, replacement=True).long().to(device)
    neg_v = torch.multinomial(node_weights, num_neg_samples, replacement=True).long().to(device)
    neg_idx = neg_u * num_nodes + neg_v

    conflict_mask = torch.isin(neg_idx, exclude_idx)
    conflicts = conflict_mask.nonzero(as_tuple=False).view(-1)
    while conflicts.numel() > 0:
        replacement_u = torch.multinomial(node_weights, conflicts.size(0), replacement=True).long().to(device)
        replacement_v = torch.multinomial(node_weights, conflicts.size(0), replacement=True).long().to(device)
        neg_u[conflicts] = replacement_u
        neg_v[conflicts] = replacement_v
        neg_idx = neg_u[conflicts] * num_nodes + neg_v[conflicts]

        conflict_mask = torch.isin(neg_idx, exclude_idx)
        conflicts = conflicts[conflict_mask]

    return torch.stack([neg_u, neg_v], dim=0)


class DataLoader(TorchDataLoader):
    """
    A custom DataLoader that forbids multiprocessing to avoid negative sampling inconsistency.
    """

    def __new__(cls, *args, **kwargs):
        if 'num_workers' in kwargs and kwargs['num_workers'] > 0:
            raise ValueError('Multiprocessing is forbidden to avoid negative sampling inconsistency.')
        return super().__new__(cls)

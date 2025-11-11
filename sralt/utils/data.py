import torch


def leave_one_out_split(edge_index: torch.Tensor, num_users: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits user-item interactions into training, validation, and test sets using random leave-one-out strategy.
    For each user, one random interaction is assigned to the validation set and another to the test set, while the rest are used for training.

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

import hashlib

import torch
import torch.nn as nn


def calculate_model_fingerprint(model: nn.Module) -> str:
    """
    Calculate a fingerprint for the given model based on its parameters.

    :param model: The PyTorch model.
    :return: The hexadecimal fingerprint string.
    """
    hasher = hashlib.md5()
    hasher.update(str(model.__class__).encode())

    for name, param in model.state_dict().items():
        param: torch.Tensor
        hasher.update(name.encode('utf-8'))
        hasher.update(param.detach().cpu().numpy().tobytes())

    return hasher.hexdigest()


def build_sparse_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.sparse.Tensor:
    """
    Build a sparse adjacency matrix for mean aggregation.

    :param edge_index: Tensor of shape (2, num_edges).
    :param num_nodes: Total number of nodes.
    :return: Sparse adjacency matrix of shape (num_nodes, num_nodes).
    """
    src, dst = edge_index
    deg = torch.bincount(dst, minlength=num_nodes).float()
    deg_inv = 1.0 / deg.clamp(min=1)
    weights = deg_inv[dst]

    return torch.sparse_coo_tensor(
        torch.stack([dst, src]), weights, (num_nodes, num_nodes)
    ).coalesce().to_sparse_csr()
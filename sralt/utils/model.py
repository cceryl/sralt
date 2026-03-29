import typing
import hashlib
from typing import Literal

import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree, add_self_loops as add_self_loops_fn


def build_normalized_adj(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
    add_self_loops: bool = False,
    normalization: Literal['none', 'symmetric', 'source', 'target'] = 'symmetric'
) -> torch.sparse.Tensor:
    """
    Build a normalized CSR adjacency matrix from the given edge index.

    :param edge_index: Undirected edge index tensor of shape (2, num_edges).
    :param num_nodes: Total number of nodes.
    :param edge_weight: Optional edge weights tensor of shape (num_edges,).
    :param add_self_loops: Whether to add self-loops to the adjacency matrix.
    :param normalization: Type of normalization to apply ('none', 'symmetric', 'source', 'target').
    :return: Normalized adjacency matrix as a CSR sparse tensor.
    """
    if add_self_loops:
        edge_index, edge_weight = add_self_loops_fn(edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    match normalization:
        case 'none':
            pass

        case 'symmetric':
            edge_index, edge_weight = typing.cast(
                tuple[torch.Tensor, torch.Tensor],
                gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=False)
            )

        case 'source':
            src, dst = edge_index
            deg = degree(src, num_nodes=num_nodes)
            deg_inv = deg.clamp(min=1).pow(-1)
            edge_weight = deg_inv[src] * edge_weight

        case 'target':
            src, dst = edge_index
            deg = degree(dst, num_nodes=num_nodes)
            deg_inv = deg.clamp(min=1).pow(-1)
            edge_weight = edge_weight * deg_inv[dst]

    src, dst = edge_index
    row, col = dst, src

    return torch.sparse_coo_tensor(
        indices=torch.stack([row, col]),
        values=edge_weight,
        size=(num_nodes, num_nodes)
    ).coalesce().to_sparse_csr()


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """
    Calculate a fingerprint for the given tensor.

    :param tensor: The PyTorch tensor.
    :return: The hexadecimal fingerprint string.
    """
    hasher = hashlib.md5()
    hasher.update(str(tensor.shape).encode())
    hasher.update(str(tensor.dtype).encode())
    hasher.update(typing.cast(NDArray, tensor.detach().cpu().numpy()).tobytes())
    return hasher.hexdigest()


def model_fingerprint(model: nn.Module) -> str:
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
        hasher.update(typing.cast(NDArray, param.detach().cpu().numpy()).tobytes())

    return hasher.hexdigest()

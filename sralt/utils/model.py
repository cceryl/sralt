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
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())

    return hasher.hexdigest()

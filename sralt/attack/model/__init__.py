import os
import pkgutil
import importlib
from typing import Any, Callable

import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric

from .base import AttackModel
from sralt.attack.data import AttackData
from sralt.utils import disk_cache, set_seed

registry: dict[str, type[AttackModel]] = {}


def register_attack_model(name: str) -> Callable:
    """
    A decorator to register attack model classes.

    :param name: The name of the model.
    :return: The decorator function.
    """
    def decorator(cls: type[AttackModel]) -> type[AttackModel]:
        registry[name] = cls
        return cls
    return decorator


def create_attack_model(name: str, user_feature_dim: int, config: dict[str, Any]) -> AttackModel:
    """
    Create a attack model by name.

    :param name: The name of the model.
    :param user_feature_dim: The dimension of user features.
    :param config: The model configuration dictionary required by the specified model.
    :return: An instance of the model.
    """
    if name not in registry:
        raise NotImplementedError(f'Attack model "{name}" not found.')

    cls = registry[name]
    return cls(user_feature_dim=user_feature_dim, **config)


@disk_cache(
    cache_dir='.cache/attack/model/{name}',
    hit='Loading trained attack model "{name}" from cache at "{__cache__.path}".',
    miss='Training attack model "{name}" and caching to "{__cache__.path}".'
)
def load_attack_model(
    seed: int, deterministic: bool, device: torch.device,
    name: str, model_config: dict[str, Any],
    data: AttackData, tail_ratio: float, train_config: dict[str, Any],
    writer: SummaryWriter | None
) -> AttackModel:
    """
    Load a trained attack model.

    :param seed: The random seed for training.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param name: The name of the model.
    :param model_config: The model configuration dictionary required by the specified model.
    :param data: The attack dataset.
    :param tail_ratio: The ratio to be considered as tail users.
    :param train_config: The training configuration dictionary required by the specified model.
    :param writer: Optional TensorBoard writer for logging.
    :return: A trained instance of the model.
    """
    set_seed(seed, deterministic)

    model = create_attack_model(
        name=name,
        user_feature_dim=data.user_feature_dim,
        config=model_config
    ).to(device)

    model.train_model(data=data, tail_ratio=tail_ratio, writer=writer, **train_config)

    return model


def test_attack_model(
    seed: int, deterministic: bool, device: torch.device,
    model: AttackModel, data: AttackData, tail_ratio: float,
    metrics: list[Metric], test_config: dict[str, Any]
) -> tuple[list[float], list[float], list[float]]:
    """
    Test the attack model.

    :param seed: The random seed for testing.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to run the model on.
    :param model: The attack model to be tested.
    :param data: The attack dataset.
    :param tail_ratio: The ratio to be considered as tail users.
    :param metrics: A list of metric calculators.
    :param test_config: The testing configuration dictionary required by the specified model.
    :return: A tuple of lists of computed metric values for all, head, and tail groups.
    """
    set_seed(seed, deterministic)

    return model.to(device).test_model(data=data.to(device), metrics=metrics, tail_ratio=tail_ratio, **test_config)


for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name != '__init__':
        importlib.import_module(f'.{module_name}', package=__name__)

__all__ = [
    'register_attack_model', 'create_attack_model',
    'load_attack_model', 'test_attack_model',
    'AttackModel'
]

import os
import pkgutil
import importlib
from typing import Any, Callable

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric

from .base import TargetModel
from sralt.target.data import TargetData
from sralt.utils import disk_cache, set_seed


registry: dict[str, type[TargetModel]] = {}


def register_target_model(name: str) -> Callable:
    """
    A decorator to register target model classes.

    :param name: The name of the model.
    :return: The decorator function.
    """
    def decorator(cls: type[TargetModel]) -> type[TargetModel]:
        registry[name] = cls
        return cls
    return decorator


def create_target_model(name: str, num_users: int, num_items: int, config: dict[str, Any]) -> TargetModel:
    """
    Create a target model by name and configuration.

    :param name: The name of the model.
    :param num_users: The number of users.
    :param num_items: The number of items.
    :param config: The model configuration dictionary required by the specified model.
    :return: An instance of the model.
    """
    if name not in registry:
        raise NotImplementedError(f'Target model "{name}" not found.')

    cls = registry[name]
    return cls(num_users=num_users, num_items=num_items, **config)


@disk_cache(
    cache_dir='.cache/target/model/{name}',
    hit_log_format='Loading trained target model "{name}" from cache at "{__cache__.path}".',
    miss_log_format='Training target model "{name}" and caching to "{__cache__.path}".'
)
def load_target_model(
    seed: int, deterministic: bool, device: torch.device,
    name: str, model_config: dict[str, Any],
    data: TargetData, train_config: dict[str, Any], writer: SummaryWriter | None
) -> TargetModel:
    """
    Load a trained target model.

    :param seed: The random seed for training.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param name: The name of the model.
    :param model_config: The model configuration dictionary required by the specified model.
    :param data: The target dataset.
    :param train_config: The training configuration dictionary required by the specified model.
    :param writer: Optional TensorBoard writer for logging.
    :return: A trained instance of the model.
    """
    set_seed(seed, deterministic=deterministic)

    model = create_target_model(name, data.num_users, data.num_items, model_config).to(device)
    model.train_model(data=data.to(device), writer=writer, **train_config)

    return model


def test_target_model(
    seed: int, deterministic: bool, device: torch.device,
    model: TargetModel, data: TargetData,
    metrics: list[LinkPredMetric], test_config: dict[str, Any]
) -> list[float]:
    """
    Test a target model.

    :param seed: The random seed for testing.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param model: The target model to test.
    :param data: The target dataset.
    :param metrics: A list of link prediction metrics to compute.
    :param test_config: The testing configuration dictionary required by the specified model.
    :return: A list of metric values.
    """
    set_seed(seed, deterministic=deterministic)

    return model.to(device).test_model(data=data.to(device), metrics=metrics, **test_config)


for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name != '__init__':
        importlib.import_module(f'.{module_name}', package=__name__)

__all__ = [
    'register_target_model', 'create_target_model',
    'load_target_model', 'test_target_model',
    'TargetModel'
]

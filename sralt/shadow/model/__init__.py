import os
import pkgutil
import importlib
from typing import Any, Callable

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric

from .base import ShadowModel
from sralt.shadow.data import ShadowData
from sralt.utils import disk_cache, set_seed


registry: dict[str, type[ShadowModel]] = {}


def register_shadow_model(name: str) -> Callable:
    """
    A decorator to register shadow model classes.

    :param name: The name of the model.
    :return: The decorator function.
    """
    def decorator(cls: type[ShadowModel]) -> type[ShadowModel]:
        registry[name] = cls
        return cls
    return decorator


def create_shadow_model(name: str, num_users: int, num_items: int, config: dict[str, Any]) -> ShadowModel:
    """
    Create a shadow model by name.

    :param name: The name of the model.
    :param num_users: The number of users.
    :param num_items: The number of items.
    :param config: The model configuration dictionary required by the specified model.
    :return: An instance of the model.
    """
    if name not in registry:
        raise NotImplementedError(f'Shadow model "{name}" not found.')

    cls = registry[name]
    return cls(num_users=num_users, num_items=num_items, **config)


@disk_cache(
    cache_dir='.cache/shadow/model/{name}',
    hit_log_format='Loading trained shadow model "{name}" from cache at "{__cache__.path}".',
    miss_log_format='Training shadow model "{name}" and caching to "{__cache__.path}".'
)
def load_shadow_model(
    seed: int, deterministic: bool, device: torch.device,
    name: str, model_config: dict[str, Any],
    data: ShadowData, train_config: dict[str, Any], writer: SummaryWriter | None
) -> ShadowModel:
    """
    Load a trained shadow model.

    :param seed: The random seed for training.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param name: The name of the model.
    :param model_config: The model configuration dictionary required by the specified model.
    :param data: The shadow dataset.
    :param train_config: The training configuration dictionary required by the specified model.
    :param writer: Optional TensorBoard writer for logging.
    :return: A trained instance of the model.
    """
    set_seed(seed, deterministic)

    model = create_shadow_model(
        name=name,
        num_users=data.num_users,
        num_items=data.num_items,
        config=model_config
    ).to(device)

    model.train_model(data=data, writer=writer, **train_config)

    return model


def test_shadow_model(
    seed: int, deterministic: bool, device: torch.device,
    model: ShadowModel, data: ShadowData,
    metrics: list[LinkPredMetric], test_config: dict[str, Any]
) -> list[float]:
    """
    Test a shadow model.

    :param seed: The random seed for testing.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param model: The shadow model to test.
    :param data: The shadow dataset.
    :param metrics: A list of link prediction metrics to compute.
    :param test_config: The testing configuration dictionary required by the specified model.
    :return: A list of metric values.
    """
    set_seed(seed, deterministic)

    return model.to(device).test_model(data=data.to(device), metrics=metrics, **test_config)


for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name != '__init__':
        importlib.import_module(f'.{module_name}', package=__name__)

__all__ = [
    'register_shadow_model', 'create_shadow_model',
    'load_shadow_model', 'test_shadow_model',
    'ShadowModel'
]

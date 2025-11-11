import os
import pkgutil
import importlib
from typing import Callable


from .base import TargetModel


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


def create_target_model(name: str, num_users: int, num_items: int, config: dict[str, object]) -> TargetModel:
    """
    Create a target model by name.

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


for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name != '__init__':
        importlib.import_module(f'.{module_name}', package=__name__)

__all__ = [
    'register_target_model', 'create_target_model',
    'TargetModel'
]

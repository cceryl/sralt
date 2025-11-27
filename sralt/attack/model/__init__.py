import os
import pkgutil
import importlib
from typing import Any, Callable

from .base import AttackModel


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


for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name != '__init__':
        importlib.import_module(f'.{module_name}', package=__name__)

__all__ = [
    'register_attack_model', 'create_attack_model',
    'AttackModel'
]

import os
import pkgutil
import importlib
from typing import Callable

from .data import Data
from sralt.utils import disk_cache


registry: dict[str, type[Data]] = {}


def register_dataset(name: str) -> Callable:
    """
    A decorator to register dataset classes.

    :param name: The name of the dataset.
    :return: The decorator function.
    """
    def decorator(cls: type[Data]) -> type[Data]:
        registry[name] = cls
        return cls
    return decorator


@disk_cache('.cache')
def load_dataset(name: str, data_dir: str, config: dict[str, object]) -> Data:
    """
    Load a dataset by name and path.

    :param name: The name of the dataset.
    :param data_dir: The directory where the dataset is located.
    :param config: The data configuration dictionary required by the specified dataset.
    :return: An instance of the dataset.
    """
    if name not in registry:
        raise NotImplementedError(f'Dataset "{name}" not found.')

    cls = registry[name]
    return cls(cls.Config(name=name, data_dir=data_dir, **config))


for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module_name not in ('__init__', 'data'):
        importlib.import_module(f'.{module_name}', package=__name__)

__all__ = ['Data', 'register_dataset', 'load_dataset']

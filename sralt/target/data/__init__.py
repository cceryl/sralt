from typing import Any, Callable

from .data import TargetData
from .dataset import TrainTargetDataset, EvalTargetDataset


registry: dict[str, type[TargetData]] = {}


def register_dataset(name: str) -> Callable:
    """
    A decorator to register target data classes.

    :param name: The name of the dataset.
    :return: The decorator function.
    """
    def decorator(cls: type[TargetData]) -> type[TargetData]:
        registry[name] = cls
        return cls
    return decorator


def load_dataset(name: str, directory: str, config: dict[str, Any]) -> TargetData:
    """
    Load a dataset by name and path.

    :param name: The name of the dataset.
    :param directory: The directory where the dataset is stored.
    :param config: The data configuration dictionary required by the specified dataset.
    :return: An instance of the data.
    """
    if name not in registry:
        raise NotImplementedError(f'Dataset "{name}" not found.')

    cls = registry[name]
    return cls(name, directory, **config)


__all__ = [
    'register_dataset', 'load_dataset',
    'TargetData',
    'TrainTargetDataset', 'EvalTargetDataset'
]

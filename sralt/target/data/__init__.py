from typing import Any, Callable

from .data import TargetData
from .dataset import TrainTargetDataset, EvalTargetDataset

from sralt.utils import disk_cache, set_seed


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


@disk_cache(
    cache_dir='.cache/target/data/{name}',
    hit_log_format='Loading cached target data "{name}" from "{__cache__.path}".',
    miss_log_format='Loading target data "{name}" from directory "{directory}" and caching to "{__cache__.path}".'
)
def load_target_data(seed: int, deterministic: bool, name: str, directory: str, config: dict[str, Any]) -> TargetData:
    """
    Load a dataset by name and path.

    :param seed: The random seed for dataset initialization.
    :param deterministic: Whether to use deterministic behavior for dataset initialization.
    :param name: The name of the dataset.
    :param directory: The directory where the dataset is stored.
    :param config: The data configuration dictionary required by the specified dataset.
    :return: An instance of the data.
    """
    if name not in registry:
        raise NotImplementedError(f'Dataset "{name}" not found.')

    cls = registry[name]
    set_seed(seed, deterministic=deterministic)

    return cls(name, directory, **config)


__all__ = [
    'register_dataset', 'load_target_data',
    'TargetData', 'TrainTargetDataset', 'EvalTargetDataset'
]

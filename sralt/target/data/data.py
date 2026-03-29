from abc import ABC
from dataclasses import dataclass
from typing_extensions import Self

import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

from sralt.utils import disk_cache, set_seed


@dataclass(init=False)
class TargetData(ABC):
    """
    The graph data for training target models.
    """

    name: str
    directory: str
    device: torch.device

    num_users: int
    num_items: int

    # Undirected user-user social graph edge index of shape (2, 2 * num_social_edges)
    social_edge_index: torch.Tensor

    # Directed user-item behavioral training graph edge index of shape (2, num_behavioral_train_edges)
    behavioral_train_edge_index: torch.Tensor

    # Directed user-item behavioral validation graph edge index of shape (2, num_behavioral_val_edges)
    behavioral_val_edge_index: torch.Tensor

    # Directed user-item behavioral test graph edge index of shape (2, num_behavioral_test_edges)
    behavioral_test_edge_index: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return self.num_users + self.num_items

    @property
    def node_range(self) -> range:
        return range(0, self.num_users + self.num_items)

    @property
    def user_range(self) -> range:
        return range(0, self.num_users)

    @property
    def item_range(self) -> range:
        return range(self.num_users, self.num_users + self.num_items)

    @property
    def num_social_edges(self) -> int:
        return self.social_edge_index.size(1) // 2

    @property
    def behavioral_edge_index(self) -> torch.Tensor:
        return torch.cat([
            self.behavioral_train_edge_index,
            self.behavioral_val_edge_index,
            self.behavioral_test_edge_index
        ], dim=1)

    @property
    def num_behavioral_edges(self) -> int:
        return (
            self.behavioral_train_edge_index.size(1) +
            self.behavioral_val_edge_index.size(1) +
            self.behavioral_test_edge_index.size(1)
        )

    @property
    def num_behavioral_train_edges(self) -> int:
        return self.behavioral_train_edge_index.size(1)

    @property
    def num_behavioral_val_edges(self) -> int:
        return self.behavioral_val_edge_index.size(1)

    @property
    def num_behavioral_test_edges(self) -> int:
        return self.behavioral_test_edge_index.size(1)

    def to(self, device: torch.device) -> Self:
        """
        Move all tensors to the specified device in-place.

        :param device: The target device.
        :return: The TargetData object on the specified device.
        """
        self.device = device
        self.social_edge_index = self.social_edge_index.to(device)
        self.behavioral_train_edge_index = self.behavioral_train_edge_index.to(device)
        self.behavioral_val_edge_index = self.behavioral_val_edge_index.to(device)
        self.behavioral_test_edge_index = self.behavioral_test_edge_index.to(device)
        return self

    def __init__(self, name: str, directory: str):
        """
        Initialize the target data.

        :param name: The name of the dataset.
        :param directory: The directory where the dataset is stored.
        """
        self.name = name
        self.directory = directory
        self.device = torch.device('cpu')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, directory={self.directory})'


@disk_cache(
    cache_dir='.cache/target/data/{config.name}',
    hit='Load cached target data "{config.name}" from "{__cache__.path}".',
    miss='Generate target data "{config.name}" from {config.directory} and cache to "{__cache__.path}".'
)
def load_target_data(seed: int, deterministic: bool, config: DictConfig) -> TargetData:
    """
    Load a dataset by provided data loader.

    :param seed: The random seed for dataset initialization.
    :param deterministic: Whether to use deterministic behavior for dataset initialization.
    :param config: The instantiable configuration for the dataset.
    """
    set_seed(seed, deterministic)

    return instantiate(config)

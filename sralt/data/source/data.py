from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch


@dataclass(init=False)
class Data(ABC):
    """
    The graph data for a social recommendation dataset.
    """

    @dataclass
    class Config():
        """
        Base class for data configuration.
        """
        name: str
        data_dir: str

    config: Config
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

    def to(self, device: torch.device) -> 'Data':
        """
        Move all data tensors to the specified device.

        :param device: The target device.
        """
        self.device = device
        self.social_edge_index = self.social_edge_index.to(device)
        self.behavioral_train_edge_index = self.behavioral_train_edge_index.to(device)
        self.behavioral_val_edge_index = self.behavioral_val_edge_index.to(device)
        self.behavioral_test_edge_index = self.behavioral_test_edge_index.to(device)
        return self

    @abstractmethod
    def load(self, config: Config):
        """
        Load data according to the given configuration.

        :param config: The data configuration.
        """
        pass

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cpu')
        self.load(config)

        for member_name in self.__dataclass_fields__.keys():
            if getattr(self, member_name, None) is None:
                raise ValueError(f'Data member "{member_name}" is not initialized in dataset {self.config.name}.')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(config={self.config})'

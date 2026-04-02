import logging
from dataclasses import dataclass
from typing_extensions import Self

import torch
from omegaconf import DictConfig

from sralt.target import TargetData
from sralt.shadow import ShadowData, ShadowModel
from sralt.utils import disk_cache, set_seed


@dataclass
class AttackData:
    """
    The binary classification data for training social recommendation attacks.
    """

    name: str
    directory: str

    target_data_repr: str
    shadow_data_repr: str
    val_ratio: float

    device: torch.device
    num_users: int

    # Undirected target social graph edge index of shape (2, 2 * num_target_social_edges)
    target_social_edge_index: torch.Tensor

    # Undirected shadow social graph edge index of shape (2, 2 * num_shadow_social_edges)
    shadow_social_edge_index: torch.Tensor

    # Directed training, validation and test social graph edge indices
    train_social_edge_index: torch.Tensor
    val_social_edge_index: torch.Tensor
    test_social_edge_index: torch.Tensor

    # User features of shape (num_users, feature_dim)
    user_features: torch.Tensor

    @property
    def num_shadow_social_edges(self) -> int:
        return self.shadow_social_edge_index.size(1) // 2

    @property
    def num_target_social_edges(self) -> int:
        return self.target_social_edge_index.size(1) // 2

    @property
    def num_train_social_edges(self) -> int:
        return self.train_social_edge_index.size(1)

    @property
    def num_val_social_edges(self) -> int:
        return self.val_social_edge_index.size(1)

    @property
    def num_test_social_edges(self) -> int:
        return self.test_social_edge_index.size(1)

    @property
    def user_feature_dim(self) -> int:
        return self.user_features.size(1)

    def to(self, device: torch.device) -> Self:
        """
        Move all tensors to the specified device in-place.

        :param device: The target device.
        :return: The AttackData object on the specified device.
        """
        self.device = device

        self.target_social_edge_index = self.target_social_edge_index.to(device)
        self.shadow_social_edge_index = self.shadow_social_edge_index.to(device)

        self.train_social_edge_index = self.train_social_edge_index.to(device)
        self.val_social_edge_index = self.val_social_edge_index.to(device)
        self.test_social_edge_index = self.test_social_edge_index.to(device)

        self.user_features = self.user_features.to(device)
        return self

    def __init__(self, target_data: TargetData, shadow_data: ShadowData, shadow_model: ShadowModel, val_ratio: float):
        """
        Initialize the attack data.

        :param target_data: The target data.
        :param shadow_data: The shadow data.
        :param shadow_model: The shadow model.
        :param val_ratio: The ratio of validation edges in the shadow social graph.
        """
        if repr(target_data) != shadow_data.target_data_repr:
            raise ValueError('The shadow data must be generated from the same target data.')

        self.name = target_data.name
        self.directory = target_data.directory

        self.target_data_repr = repr(target_data)
        self.shadow_data_repr = repr(shadow_data)
        self.shadow_model_repr = repr(shadow_model)
        self.val_ratio = val_ratio

        self.device = target_data.device
        self.num_users = target_data.num_users

        self.target_social_edge_index = target_data.social_edge_index
        self.shadow_social_edge_index = shadow_data.social_edge_index

        def to_directed(edge_index: torch.Tensor) -> torch.Tensor:
            return edge_index[:, edge_index[0] < edge_index[1]]

        # Split shadow social edges into training and validation sets
        directed_shadow_edges = to_directed(self.shadow_social_edge_index)
        perm = torch.randperm(directed_shadow_edges.size(1), device=self.device)
        num_val_edges = int(self.val_ratio * directed_shadow_edges.size(1))

        self.train_social_edge_index = directed_shadow_edges[:, perm[num_val_edges:]]
        self.val_social_edge_index = directed_shadow_edges[:, perm[:num_val_edges]]

        # Test edges are unseen edges in the target social graph
        directed_target_edges = to_directed(self.target_social_edge_index)

        target_edge_ids = directed_target_edges[0] * self.num_users + directed_target_edges[1]
        shadow_edge_ids = directed_shadow_edges[0] * self.num_users + directed_shadow_edges[1]

        unseen_mask = torch.isin(target_edge_ids, shadow_edge_ids, invert=True)
        self.test_social_edge_index = directed_target_edges[:, unseen_mask]

        # Extract user features from the shadow model
        self.user_features = shadow_model.get_user_features(shadow_data)

        logging.info(
            f'Attack data generated with '
            f'{self.num_users} users, '
            f'{self.num_train_social_edges} training edges, '
            f'{self.num_val_social_edges} validation edges, '
            f'{self.num_test_social_edges} test edges.'
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'target_data={self.target_data_repr}, '
            f'shadow_data={self.shadow_data_repr}, '
            f'shadow_model={self.shadow_model_repr}, '
            f'val_ratio={self.val_ratio})'
        )


@disk_cache(
    cache_dir='.cache/attack/data/{target_data.name}/{shadow_data.target_model_name}/{shadow_model.__class__.__name__}',
    hit='Load cached attack data "{target_data.name}" from "{__cache__.path}".',
    miss='Generate attack data from dataset "{target_data.name}", shadow model "{shadow_model.__class__.__name__}" and cache to "{__cache__.path}".'
)
def load_attack_data(
    seed: int, deterministic: bool,
    target_data: TargetData, shadow_data: ShadowData, shadow_model: ShadowModel, config: DictConfig
) -> AttackData:
    """
    Load attack data from target data, shadow data and shadow model.

    :param seed: The random seed for attack data generation.
    :param deterministic: Whether to use deterministic behavior for attack data generation.
    :param target_data: The target data.
    :param shadow_data: The shadow data.
    :param shadow_model: The shadow model.
    :param config: The configuration for attack data generation.
    :return: An instance of the attack data.
    """
    set_seed(seed, deterministic=deterministic)

    return AttackData(
        target_data=target_data,
        shadow_data=shadow_data,
        shadow_model=shadow_model,
        val_ratio=config.val_ratio
    )

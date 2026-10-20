import logging

import torch
from omegaconf import DictConfig
from typing_extensions import Self
from torch_geometric.utils import to_undirected

from sralt.target import TargetData, TargetModel
from sralt.utils import disk_cache, set_seed, leave_one_out_split


class ShadowData(TargetData):
    """
    The graph data for training shadow models.
    """

    def __init__(
        self,
        target_data: TargetData, target_model: TargetModel,
        shadow_social_ratio: float, num_recommendations: int,
        inference_batch_size: int, attack_val_ratio: float
    ):
        """
        Initialize the shadow data.

        :param target_data: The target data.
        :param target_model: The target recommendation model.
        :param shadow_social_ratio: The ratio of social edges to include in the shadow data.
        :param num_recommendations: The number of recommendations to generate for each user.
        :param inference_batch_size: The batch size for generating recommendations.
        :param attack_val_ratio: Fraction of public relations reserved for attack validation.
        """
        # Store metadata
        super().__init__(name=target_data.name, directory=target_data.directory)

        self.target_model_name = target_model.__class__.__name__
        self.target_data_repr = repr(target_data)
        self.target_model_repr = repr(target_model)

        self.shadow_social_ratio = shadow_social_ratio
        self.num_recommendations = num_recommendations
        self.attack_val_ratio = attack_val_ratio

        self.device = target_data.device
        self.num_users = target_data.num_users
        self.num_items = target_data.num_items

        # Sample social connections from the target data
        directed_mask = target_data.social_edge_index[0] <= target_data.social_edge_index[1]
        directed_social_edge_index = target_data.social_edge_index[:, directed_mask]

        num_shadow_social_edges = int(directed_social_edge_index.size(1) * shadow_social_ratio)
        perm = torch.randperm(directed_social_edge_index.size(1), device=target_data.device)
        selected_indices = perm[:num_shadow_social_edges]

        public_edges = directed_social_edge_index[:, selected_indices]
        self.public_social_edge_index = to_undirected(public_edges)
        split_perm = torch.randperm(public_edges.size(1), device=self.device)
        num_val_edges = int(attack_val_ratio * public_edges.size(1))
        self.attack_train_social_edge_index = public_edges[:, split_perm[num_val_edges:]]
        self.attack_val_social_edge_index = public_edges[:, split_perm[:num_val_edges]]
        self.social_edge_index = to_undirected(self.attack_train_social_edge_index)

        # Generate recommendations for each user using the target model
        recommendations_edge_index = target_model.recommend(
            data=target_data,
            num_recommendations=num_recommendations,
            batch_size=inference_batch_size
        )

        train_edge_index, val_edge_index, test_edge_index = leave_one_out_split(
            recommendations_edge_index, self.num_users
        )

        self.behavioral_train_edge_index = train_edge_index
        self.behavioral_val_edge_index = val_edge_index
        self.behavioral_test_edge_index = test_edge_index

        self.to(torch.device('cpu'))
        logging.info(
            f'Shadow data generated with '
            f'{self.num_users} users, '
            f'{self.num_items} items, '
            f'{self.num_social_edges} social edges, '
            f'{self.num_behavioral_edges} behavioral edges.'
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'target_data={self.target_data_repr}, '
            f'target_model={self.target_model_repr}, '
            f'shadow_social_ratio={self.shadow_social_ratio}, '
            f'attack_val_ratio={self.attack_val_ratio}, '
            f'social_split=attack-train-only-v1, '
            f'num_recommendations={self.num_recommendations})'
        )

    def to(self, device: torch.device) -> Self:
        super().to(device)
        self.public_social_edge_index = self.public_social_edge_index.to(device)
        self.attack_train_social_edge_index = self.attack_train_social_edge_index.to(device)
        self.attack_val_social_edge_index = self.attack_val_social_edge_index.to(device)
        return self

    def cache_key(self) -> tuple:
        return (super().cache_key(), self.public_social_edge_index,
                self.attack_train_social_edge_index, self.attack_val_social_edge_index)


@disk_cache(
    cache_dir='.cache/shadow/data/{target_data.name}/{target_model.__class__.__name__}',
    hit='Load cached shadow data "{target_data.name}" from "{__cache__.path}".',
    miss='Generate shadow data from target data "{target_data.name}" and cache to "{__cache__.path}".',
    dependencies=lambda args: args['config'].attack_val_ratio
)
def load_shadow_data(
    seed: int, deterministic: bool,
    target_data: TargetData, target_model: TargetModel, config: DictConfig
) -> ShadowData:
    """
    Load shadow data from target data and target model.

    :param seed: The random seed for shadow data generation.
    :param deterministic: Whether to use deterministic behavior for shadow data generation.
    :param target_data: The target data.
    :param target_model: The target recommendation model.
    :param config: The configuration for shadow data generation.
    :return: An instance of the shadow data.
    """
    set_seed(seed, deterministic=deterministic)

    return ShadowData(
        target_data=target_data,
        target_model=target_model,
        shadow_social_ratio=config.shadow_social_ratio,
        num_recommendations=config.num_recommendations,
        inference_batch_size=config.inference_batch_size,
        attack_val_ratio=config.attack_val_ratio
    )

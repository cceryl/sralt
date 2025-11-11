import logging
from dataclasses import dataclass

import torch
from torch_geometric.utils import coalesce, to_undirected

from sralt.target.data import TargetData
from sralt.target.model import TargetModel
from sralt.utils import leave_one_out_split


@dataclass(init=False)
class ShadowData():
    """
    The graph data for training shadow social recommendation models.
    """
    target_data_repr: str
    target_model_repr: str
    shadow_social_ratio: float
    num_recommendations: int
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

    def to(self, device: torch.device) -> 'ShadowData':
        """
        Move all tensors to the specified device.

        :param device: The target device.
        """
        self.device = device
        self.social_edge_index = self.social_edge_index.to(device)
        self.behavioral_train_edge_index = self.behavioral_train_edge_index.to(device)
        self.behavioral_val_edge_index = self.behavioral_val_edge_index.to(device)
        self.behavioral_test_edge_index = self.behavioral_test_edge_index.to(device)
        return self

    @torch.no_grad()
    def __init__(
        self,
        target_data: TargetData, target_model: TargetModel,
        shadow_social_ratio: float, num_recommendations: int,
        inference_batch_size: int = 2048
    ):
        """
        Initialize the shadow data.

        :param target_data: The target data.
        :param target_model: The target recommendation model.
        :param shadow_social_ratio: The ratio of social edges to include in the shadow data.
        :param num_recommendations: The number of recommendations to generate for each user.
        :param inference_batch_size: The batch size for generating recommendations.
        """
        # Store metadata
        self.target_data_repr = repr(target_data)
        self.target_model_repr = repr(target_model)

        self.shadow_social_ratio = shadow_social_ratio
        self.num_recommendations = num_recommendations

        self.device = target_data.device
        self.num_users = target_data.num_users
        self.num_items = target_data.num_items

        # Sample social connections from the target data
        num_shadow_social_edges = int(target_data.num_social_edges * shadow_social_ratio)
        perm = torch.randperm(target_data.num_social_edges, device=target_data.device)
        selected_indices = perm[:num_shadow_social_edges]

        directed_mask = target_data.social_edge_index[0] < target_data.social_edge_index[1]
        directed_social_edge_index = target_data.social_edge_index[:, directed_mask]
        self.social_edge_index = to_undirected(directed_social_edge_index[:, selected_indices])

        # Generate recommendations for each user using the target model
        target_model = target_model.to(target_data.device)
        target_model.eval()

        user_indices, item_indices = target_data.behavioral_edge_index
        item_indices = item_indices - target_data.item_range.start

        embeds = target_model.get_embedding(
            target_data.social_edge_index,
            to_undirected(target_data.behavioral_edge_index)
        )
        user_embeds, item_embeds = torch.split(embeds, [target_data.num_users, target_data.num_items], dim=0)

        recommendations = []
        for start in range(0, target_data.num_users, inference_batch_size):
            end = min(start + inference_batch_size, target_data.num_users)

            scores = user_embeds[start:end] @ item_embeds.t()
            mask = (start <= user_indices) & (user_indices < end)
            scores[user_indices[mask] - start, item_indices[mask]] = float('-inf')

            _, top_items = torch.topk(scores, k=num_recommendations, sorted=True)

            users = torch.arange(start, end, device=target_data.device).repeat_interleave(num_recommendations)
            items = top_items.flatten() + target_data.item_range.start

            recommendations.append(torch.stack([users, items], dim=0))

        recommendations_edge_index = coalesce(torch.cat(recommendations, dim=1))
        train_edge_index, val_edge_index, test_edge_index = leave_one_out_split(
            recommendations_edge_index, self.num_users
        )

        self.behavioral_train_edge_index = train_edge_index
        self.behavioral_val_edge_index = val_edge_index
        self.behavioral_test_edge_index = test_edge_index

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
            f'num_recommendations={self.num_recommendations})'
        )

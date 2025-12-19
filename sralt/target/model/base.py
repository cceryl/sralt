from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric

from sralt.target.data import TargetData
from sralt.utils import timer


class TargetModel(nn.Module, ABC):
    """
    Base class for target social recommendation models.
    """

    def __init__(self, num_users: int, num_items: int):
        """
        :param num_users: Total number of users.
        :param num_items: Total number of items.
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items

    @abstractmethod
    @timer('Training model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_model(self, *, data: TargetData, writer: SummaryWriter | None, **kwargs):
        """
        Train the target model.

        :param data: The target dataset.
        :param writer: Optional TensorBoard writer for logging.
        :param kwargs: Additional training parameters specific to the model.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def test_model(self, *, data: TargetData, metrics: list[LinkPredMetric], **kwarg) -> list[float]:
        """
        Test the target model.

        :param data: The target dataset.
        :param metrics: A list of link prediction metrics to compute.
        :param kwargs: Additional testing parameters specific to the model.
        :return: A list of metric values.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def recommend(self, *, data: TargetData, num_recommendations: int, batch_size: int = 1024) -> torch.Tensor:
        """
        Generate recommendations for the given users, avoiding items they have already interacted with.

        :param data: The target dataset.
        :param num_recommendations: The number of recommendations to generate per user.
        :param batch_size: The batch size for generating recommendations.
        :return: Edge index tensor of shape (2, num_users * num_recommendations) containing recommendations.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation and fingerprint of the model.
        """
        pass

from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric

from sralt.shadow.data import ShadowData
from sralt.utils import timer


class ShadowModel(nn.Module, ABC):
    """
    Base class for shadow social recommendation models.
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
    def train_model(self, *, data: ShadowData, writer: SummaryWriter | None, **kwargs):
        """
        Train the shadow model.

        :param data: The shadow dataset.
        :param writer: Optional TensorBoard writer for logging.
        :param kwargs: Additional training parameters specific to the model.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def test_model(self, *, data: ShadowData, metrics: list[LinkPredMetric], **kwarg) -> list[float]:
        """
        Test the shadow model.

        :param data: The shadow dataset.
        :param metrics: A list of link prediction metrics to compute.
        :param kwargs: Additional testing parameters specific to the model.
        :return: A list of metric values.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def get_user_features(self, data: ShadowData) -> torch.Tensor:
        """
        Get the user features learned by the shadow model for attack purposes.

        :param data: The shadow dataset.
        :return: A tensor of shape (num_users, feature_dim) representing user features.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation and fingerprint of the model.
        """
        pass

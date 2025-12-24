from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric

from sralt.attack.data import AttackData
from sralt.utils import timer


class AttackModel(nn.Module, ABC):
    """
    Base class for social connection attack models.
    """

    def __init__(self, user_feature_dim: int):
        """
        :param user_feature_dim: Dimension of user features.
        """
        super().__init__()

        self.user_feature_dim = user_feature_dim

    @abstractmethod
    @timer('Training model "{self.__class__.__name__}" took {__timer__.elapsed:.2f} seconds.')
    def train_model(self, *, data: AttackData, tail_ratio: float, writer: SummaryWriter | None, **kwargs):
        """
        Train the attack model.

        :param data: The attack dataset.
        :param tail_ratio: The ratio to be considered as tail users.
        :param writer: Optional TensorBoard writer for logging.
        :param kwargs: Additional training parameters specific to the model.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def test_model(self, *, data: AttackData, tail_ratio: float, metrics: list[Metric], **kwargs) -> tuple[list[float], list[float], list[float]]:
        """
        Test the attack model.

        :param data: The attack dataset.
        :param tail_ratio: The ratio to be considered as tail users.
        :param metrics: A list of metric calculators.
        :param kwargs: Additional testing parameters specific to the model.
        :return: A tuple of lists of computed metric values for all, head, and tail groups.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation and fingerprint of the model.
        """
        pass

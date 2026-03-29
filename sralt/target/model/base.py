from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric

from sralt.target.data import TargetData
from sralt.utils import disk_cache, timer, set_seed


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
    def train_model(self, *, data: TargetData, writer: SummaryWriter | None, **kwargs) -> None:
        """
        Train the target model.

        :param data: The target dataset.
        :param writer: Optional TensorBoard writer for logging.
        :param kwargs: Training config specific to the model.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def test_model(self, *, data: TargetData, metrics: list[LinkPredMetric], **kwargs) -> list[float]:
        """
        Test the target model.

        :param data: The target dataset.
        :param metrics: A list of link prediction metrics to compute.
        :param kwargs: Testing config specific to the model.
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


def create_target_model(
    seed: int, deterministic: bool, device: torch.device,
    config: DictConfig, num_users: int, num_items: int
) -> TargetModel:
    """
    Create a target model without training.

    :param seed: The random seed for model creation.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param config: The model instantiation config.
    :param num_users: Total number of users in the dataset.
    :param num_items: Total number of items in the dataset.
    :return: An untrained instance of the model.
    """
    set_seed(seed, deterministic=deterministic)

    model: TargetModel = instantiate(config, num_users=num_users, num_items=num_items)
    model = model.to(device)

    return model


@disk_cache(
    cache_dir='.cache/target/model/{config.name}',
    hit='Load trained target model "{config.name}" from cache at "{__cache__.path}".',
    miss='Train target model "{config.name}" and cache to "{__cache__.path}".'
)
def load_target_model(
    seed: int, deterministic: bool, device: torch.device,
    config: DictConfig, data: TargetData, writer: SummaryWriter | None = None
) -> TargetModel:
    """
    Load a trained target model.
    The config must contain the following fields:
    - name: The name of the model.
    - attribute: The model instantiation config.
    - train: The training config specific to the model.

    :param seed: The random seed for training.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param name: The name of the model.
    :param config: The model intantiation and training configs.
    :param data: The target dataset.
    :param writer: Optional TensorBoard writer for logging.
    :return: A trained instance of the model.
    """
    model = create_target_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        config=config.attribute,
        num_users=data.num_users,
        num_items=data.num_items
    )

    set_seed(seed, deterministic=deterministic)
    model.train_model(data=data, writer=writer, **config.train)

    return model


def test_target_model(
    seed: int, deterministic: bool, device: torch.device,
    model: TargetModel, data: TargetData, config: DictConfig
) -> list[float]:
    """
    Test a target model.
    The config must contain the following fields:
    - metrics: A list of link prediction metric instantiation configs.
    - evaluation: The testing config specific to the model.

    :param seed: The random seed for testing.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param model: The target model to test.
    :param data: The target dataset.
    :param config: The test configuration.
    :return: A list of metric values.
    """
    set_seed(seed, deterministic=deterministic)

    model = model.to(device)
    metrics: list[LinkPredMetric] = instantiate(config.metrics)
    metric_values = model.test_model(data=data, metrics=metrics, **config.evaluation)

    return metric_values

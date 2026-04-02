from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics.link_pred import LinkPredMetric

from sralt.shadow.data import ShadowData
from sralt.utils import disk_cache, timer, set_seed


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
        :param kwargs: Training config specific to the model.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def test_model(self, *, data: ShadowData, metrics: list[LinkPredMetric], **kwargs) -> list[float]:
        """
        Test the shadow model.

        :param data: The shadow dataset.
        :param metrics: A list of link prediction metrics to compute.
        :param kwargs: Testing config specific to the model.
        :return: A list of metric values.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def get_user_features(self, data: ShadowData) -> torch.Tensor:
        """
        Get the user features learned by the shadow model.

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


def create_shadow_model(
    seed: int, deterministic: bool, device: torch.device,
    config: DictConfig, num_users: int, num_items: int
) -> ShadowModel:
    """
    Create a shadow model without training.

    :param seed: The random seed for model creation.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param config: The model instantiation config.
    :param num_users: Total number of users in the dataset.
    :param num_items: Total number of items in the dataset.
    :return: An untrained instance of the model.    
    """
    set_seed(seed, deterministic=deterministic)

    model: ShadowModel = instantiate(config, num_users=num_users, num_items=num_items)
    model = model.to(device)

    return model


@disk_cache(
    cache_dir='.cache/shadow/model/{config.name}',
    hit='Load trained shadow model "{config.name}" from cache at "{__cache__.path}".',
    miss='Train shadow model "{config.name}" and cache to "{__cache__.path}".'
)
def load_shadow_model(
    seed: int, deterministic: bool, device: torch.device,
    config: DictConfig, data: ShadowData, writer: SummaryWriter | None = None
) -> ShadowModel:
    """
    Load a trained shadow model.
    The config must contain the following fields:
    - name: The name of the model.
    - attribute: The model instantiation config.
    - train: The training config specific to the model.

    :param seed: The random seed for training.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param config: The model intantiation and training configs.
    :param data: The shadow dataset.
    :param writer: Optional TensorBoard writer for logging.
    :return: A trained instance of the model.
    """
    model = create_shadow_model(
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


def test_shadow_model(
    seed: int, deterministic: bool, device: torch.device,
    model: ShadowModel, data: ShadowData, config: DictConfig
) -> list[float]:
    """
    Test a shadow model.
    The config must contain the following fields:
    - metrics: A list of link prediction metric instantiation configs.
    - evaluation: The testing config specific to the model.

    :param seed: The random seed for testing.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param model: The shadow model to test.
    :param data: The shadow dataset.
    :param config: The test configuration.
    :return: A list of metric values.
    """
    set_seed(seed, deterministic)

    model = model.to(device)
    metrics: list[LinkPredMetric] = instantiate(config.metrics)
    metric_values = model.test_model(data=data, metrics=metrics, **config.evaluation)

    return metric_values

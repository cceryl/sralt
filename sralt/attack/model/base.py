import typing
from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from torchmetrics import Metric
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter

from sralt.attack.data import AttackData
from sralt.utils import disk_cache, timer, set_seed


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
        :param tail_ratio: Proportion of users classified as long-tail users based on social connection counts.
        :param writer: Optional TensorBoard writer for logging.
        :param kwargs: Training config specific to the model.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def test_model(self, *, data: AttackData, tail_ratio: float, metrics: list[Metric], **kwargs) -> tuple[list[float], list[float], list[float]]:
        """
        Test the attack model.

        :param data: The attack dataset.
        :param tail_ratio: Proportion of users classified as long-tail users based on social connection counts.
        :param metrics: A list of metric calculators.
        :param kwargs: Testing config specific to the model.
        Final evaluation uses one public-validation-All-optimal F1 threshold for every group.
        Five-group metrics and threshold metadata are stored on data.subgroup_metrics.
        :return: Metric lists for All/Head/Tail, retained for existing callers.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def predict(self, user_edge_index: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Predict the attack outcome.

        :param user_edge_index: Edge index tensor of shape (2, batch_size).
        :param user_features: Tensor of shape (num_users, user_feature_dim) representing user features.
        :return: The predicted probability tensor of shape (batch_size,).
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation and fingerprint of the model.
        """
        pass


def create_attack_model(
    seed: int, deterministic: bool, device: torch.device,
    config: DictConfig, user_feature_dim: int
) -> AttackModel:
    """
    Create a attack model by name.

    :param name: The name of the model.
    :param user_feature_dim: The dimension of user features.
    :param config: The model configuration dictionary required by the specified model.
    :return: An instance of the model.
    """
    set_seed(seed, deterministic=deterministic)

    model: AttackModel = instantiate(config, user_feature_dim=user_feature_dim)
    model = model.to(device)

    return model


@disk_cache(
    cache_dir='.cache/attack/model/{config.name}',
    hit='Load trained attack model "{config.name}" from cache at "{__cache__.path}".',
    miss='Train attack model "{config.name}" and cache to "{__cache__.path}".',
    ignore=['device', 'writer'],
    hook=lambda model, args, _: typing.cast(AttackModel, model).to(args['device'])
)
def load_attack_model(
    seed: int, deterministic: bool, device: torch.device,
    config: DictConfig, data: AttackData, tail_ratio: float,
    writer: SummaryWriter | None
) -> AttackModel:
    """
    Load a trained attack model.
    The config must contain the following fields:
    - name: The name of the model.
    - attribute: The model instantiation config.
    - train: The training config specific to the model.

    :param seed: The random seed for training.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param config: The model intantiation and training configs.
    :param data: The attack dataset.
    :param tail_ratio: Proportion of users classified as long-tail users based on social connection counts. 
    :param writer: Optional TensorBoard writer for logging.
    :return: A trained instance of the model.
    """
    model = create_attack_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        config=config.attribute,
        user_feature_dim=data.user_features.size(1)
    )

    set_seed(seed, deterministic=deterministic)
    model.train_model(data=data, tail_ratio=tail_ratio, writer=writer, **config.train)

    return model.cpu()


def test_attack_model(
    seed: int, deterministic: bool, device: torch.device,
    model: AttackModel, data: AttackData, config: DictConfig, tail_ratio: float
) -> tuple[list[float], list[float], list[float]]:
    """
    Test a attack model.
    The config must contain the following fields:
    - metrics: A list of binary classification metric instantiation configs.
    - evaluation: The testing config specific to the model.

    :param seed: The random seed for testing.
    :param deterministic: Whether to use deterministic algorithms.
    :param device: The device to load the model onto.
    :param model: The shadow model to test.
    :param data: The shadow dataset.
    :param config: The test configuration.
    :param tail_ratio: Proportion of users classified as long-tail users based on social connection counts.
    :return: A tuple of lists of computed metric values for all, head, and tail groups.
    """
    set_seed(seed, deterministic)

    model = model.to(device)
    metrics: list[Metric] = instantiate(config.metrics)
    metric_values = model.test_model(data=data, metrics=metrics, tail_ratio=tail_ratio, **config.evaluation)

    return metric_values

import os
import copy
import random
import logging

import torch
import torch.nn as nn
import numpy as np


def set_seed(seed: int, deterministic: bool = True):
    """
    Set the random seed for reproducibility.

    :param seed: The seed value to set for random number generation.
    :param deterministic: Whether to set deterministic options for CUDA.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logging.info(f'Random seed={seed}, deterministic={deterministic}.')


class EarlyStopping:
    """
    Early stopping utility to stop training when metric does not improve.
    """

    def __init__(self, model: nn.Module, patience: int, greater_is_better: bool = False):
        """
        :param model: The model being trained.
        :param patience: Number of epochs to wait for improvement before stopping.
        :param greater_is_better: If True, higher metric values are better (e.g., accuracy).
        """
        self.model = model
        self.patience = patience
        self.greater_is_better = greater_is_better

        self.counter = 0
        self.best_metric = -float('inf') if greater_is_better else float('inf')
        self.best_state = copy.deepcopy(model.state_dict())

    def step(self, metric: float) -> bool:
        """
        Check if early stopping condition is met.

        :param metric: The current validation metric (e.g., loss or accuracy).
        :return: True if training should be stopped, False otherwise.
        """
        if (self.greater_is_better and metric > self.best_metric) or (not self.greater_is_better and metric < self.best_metric):
            self.counter = 0
            self.best_metric = metric
            self.best_state = copy.deepcopy(self.model.state_dict())
            return False

        self.counter += 1
        if self.counter < self.patience:
            logging.debug(f'No improvement for {self.counter} epochs (best metric={self.best_metric:.4f}).')
            return False

        logging.info(
            f'Early stop after {self.patience} epochs without improvement (best metric={self.best_metric:.4f}).'
        )
        return True

    def load_best_state(self):
        """
        Load the best model state saved during training.
        """
        self.model.load_state_dict(self.best_state)
        logging.info(f'Loaded best model state with metric={self.best_metric:.4f}.')

    def reset(self):
        """
        Reset the early stopping state.
        """
        self.counter = 0
        self.best_metric = -float('inf') if self.greater_is_better else float('inf')
        self.best_state = copy.deepcopy(self.model.state_dict())
        logging.info('Early stopping state is reset.')

# This file is an example script for training and evaluating a target recommendation model.

import typing
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from sralt.target import load_target_data, load_target_model, test_target_model


@hydra.main(config_path='pkg://sralt.configs', config_name='config', version_base=None)
def main(config: DictConfig):
    instantiate(config.filter_warnings)

    device: torch.device = instantiate(config.device)
    writer: SummaryWriter = instantiate(config.writer)

    target_data = load_target_data(
        seed=config.seed,
        deterministic=config.deterministic,
        config=config.target.data
    ).to(device)

    target_model = load_target_model(
        seed=config.seed,
        deterministic=config.deterministic,
        device=device,
        config=config.target.model,
        data=target_data,
        writer=writer
    )

    test_metrics = test_target_model(
        seed=config.seed,
        deterministic=config.deterministic,
        device=device,
        model=target_model,
        data=target_data,
        config=config.target.test
    )

    metric_names = [
        f'{typing.cast(str, metric._target_).split(".")[-1].removeprefix("LinkPred")}@{metric.k}'
        for metric in config.target.test.metrics
    ]

    for name, value in zip(metric_names, test_metrics):
        logging.info(f'Test {name}: {value:.4f}')

    logging.info(f'Model representation: {target_model}')
    torch.save(target_model.state_dict(), 'model.pth')
    writer.close()


if __name__ == '__main__':
    main()

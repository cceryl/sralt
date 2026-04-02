# This file is an example script for training and evaluating target and shadow recommendation models.

import typing
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

import hydra
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from sralt.target import TargetData, TargetModel, load_target_data, load_target_model, test_target_model
from sralt.shadow import ShadowData, ShadowModel, load_shadow_data, load_shadow_model, test_shadow_model


def target_stage(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter, config: DictConfig
) -> tuple[TargetData, TargetModel]:
    target_data = load_target_data(
        seed=seed,
        deterministic=deterministic,
        config=config.data
    ).to(device)

    target_model = load_target_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        config=config.model,
        data=target_data,
        writer=writer
    )

    test_metrics = test_target_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        model=target_model,
        data=target_data,
        config=config.test
    )

    metric_names = [
        f'{typing.cast(str, metric._target_).split(".")[-1].removeprefix("LinkPred")}@{metric.k}'
        for metric in config.test.metrics
    ]

    logging.info('Target model evaluation results:')

    for name, value in zip(metric_names, test_metrics):
        logging.info(f'Test {name}: {value:.4f}')

    logging.info(f'Model representation: {target_model}')

    return target_data, target_model


def shadow_stage(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter, config: DictConfig,
    target_data: TargetData, target_model: TargetModel
) -> tuple[ShadowData, ShadowModel]:
    shadow_data = load_shadow_data(
        seed=seed,
        deterministic=deterministic,
        target_data=target_data,
        target_model=target_model,
        config=config.data
    ).to(device)

    shadow_model = load_shadow_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        config=config.model,
        data=shadow_data,
        writer=writer
    )

    test_metrics = test_shadow_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        model=shadow_model,
        data=shadow_data,
        config=config.test
    )

    metric_names = [
        f'{typing.cast(str, metric._target_).split(".")[-1].removeprefix("LinkPred")}@{metric.k}'
        for metric in config.test.metrics
    ]

    logging.info('Shadow model evaluation results:')

    for name, value in zip(metric_names, test_metrics):
        logging.info(f'Test {name}: {value:.4f}')

    logging.info(f'Model representation: {shadow_model}')

    return shadow_data, shadow_model


@hydra.main(config_path='pkg://sralt.configs', config_name='config', version_base=None)
def main(config: DictConfig):
    instantiate(config.filter_warnings)

    device: torch.device = instantiate(config.device)
    writer: SummaryWriter = instantiate(config.writer)

    target_data, target_model = target_stage(
        seed=config.seed,
        deterministic=config.deterministic,
        device=device,
        writer=writer,
        config=config.target
    )

    shadow_data, shadow_model = shadow_stage(
        seed=config.seed,
        deterministic=config.deterministic,
        device=device,
        writer=writer,
        config=config.shadow,
        target_data=target_data,
        target_model=target_model
    )

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    torch.save(target_data, output_dir / 'target_data.pth')
    torch.save(shadow_data, output_dir / 'shadow_data.pth')
    torch.save(target_model.state_dict(), output_dir / 'target_model.pth')
    torch.save(shadow_model.state_dict(), output_dir / 'shadow_model.pth')

    writer.close()


if __name__ == '__main__':
    main()

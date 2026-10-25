"""Run the SRALT pipeline with prompt model disposal and no run checkpoints."""

import gc
import json
import logging
import math
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from sralt.target import load_target_data, load_target_model, test_target_model
from sralt.shadow import load_shadow_data, load_shadow_model, test_shadow_model
from sralt.attack import load_attack_data, load_attack_model, test_attack_model
from sralt.attack.report import save_attack_report


def metric_values(metrics, values):
    result = {}
    for metric, value in zip(metrics, values, strict=True):
        name = metric._target_.split('.')[-1].removeprefix('Binary').removeprefix('LinkPred')
        if name == 'TPRatFPR':
            name = f'TPR@FPR={metric.target_fpr:.0%}'
        elif 'k' in metric:
            name = f'{name}@{metric.k}'
        result[name] = float(value) if math.isfinite(value) else None
    return result


def release_memory(device: torch.device) -> None:
    """Collect training reference cycles before returning unused CUDA blocks."""
    gc.collect()
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def log_recommendation_results(stage: str, config: DictConfig, values: list[float], model) -> None:
    logging.info(f'{stage} model evaluation results:')
    for metric, value in zip(config.metrics, values):
        name = metric._target_.split('.')[-1].removeprefix('LinkPred')
        logging.info(f'Test {name}@{metric.k}: {value:.4f}')
    logging.info(f'Model representation: {model}')


def run_pipeline(config: DictConfig) -> None:
    instantiate(config.filter_warnings)
    logging.info(
        f'Dataset: {config.target.data.name}; '
        f'Target: {config.target.model.name}; '
        f'Shadow: {config.shadow.model.name}; '
        f'Attack: {config.attack.model.name}.'
    )
    device: torch.device = instantiate(config.device)
    writer = instantiate(config.writer)
    common = dict(seed=config.seed, deterministic=config.deterministic)
    recorded_metrics = {}

    try:
        target_data = load_target_data(**common, config=config.target.data).to(device)
        target_model = load_target_model(
            **common, device=device, writer=writer,
            config=config.target.model, data=target_data
        )
        values = test_target_model(
            **common, device=device, model=target_model,
            data=target_data, config=config.target.test
        )
        log_recommendation_results('Target', config.target.test, values, target_model)
        recorded_metrics['target'] = {'all': metric_values(config.target.test.metrics, values)}

        shadow_data = load_shadow_data(
            **common, target_data=target_data, target_model=target_model,
            config=config.shadow.data
        )
        # Only the target data is needed to construct the attack split later.
        del target_model
        release_memory(device)
        logging.info('Released target model before shadow training.')

        shadow_data = shadow_data.to(device)
        if config.shadow.model.name == 'None':
            shadow_model = None
            logging.info('No shadow model: use queried recommendations and public split directly.')
        else:
            shadow_model = load_shadow_model(
                **common, device=device, writer=writer,
                config=config.shadow.model, data=shadow_data
            )
            values = test_shadow_model(
                **common, device=device, model=shadow_model,
                data=shadow_data, config=config.shadow.test
            )
            log_recommendation_results('Shadow', config.shadow.test, values, shadow_model)
            recorded_metrics['shadow'] = {'all': metric_values(config.shadow.test.metrics, values)}

        attack_data = load_attack_data(
            **common, target_data=target_data, shadow_data=shadow_data,
            shadow_model=shadow_model, config=config.attack.data
        )
        # AttackData owns the extracted features and graph tensors it needs.
        del shadow_model, shadow_data, target_data
        release_memory(device)
        logging.info('Released shadow model and upstream data before attack training.')

        attack_data = attack_data.to(device)
        attack_model = load_attack_model(
            **common, device=device, writer=writer, config=config.attack.model,
            data=attack_data, tail_ratio=config.attack.tail_ratio
        )
        results = test_attack_model(
            **common, device=device, model=attack_model, data=attack_data,
            config=config.attack.test, tail_ratio=config.attack.tail_ratio
        )
        logging.info('Attack model evaluation results:')
        for group, metrics in zip(['All', 'Head', 'Tail'], results):
            for metric, value in zip(config.attack.test.metrics, metrics):
                name = metric._target_.split('.')[-1].removeprefix('Binary')
                logging.info(f'Test {group} {name}: {value:.4f}')
        logging.info(f'Model representation: {attack_model}')
        recorded_metrics['attack'] = {
            group: metric_values(config.attack.test.metrics, values)
            for group, values in zip(('all', 'head', 'tail'), results, strict=True)
        }
        if HydraConfig.initialized():
            output_dir = Path(HydraConfig.get().runtime.output_dir)
            save_attack_report(attack_data, attack_model, output_dir)
            record = {
                'schema': 1, 'dataset': config.target.data.name.lower(), 'seed': config.seed,
                'models': {stage: config[stage].model.name for stage in ('target', 'shadow', 'attack')},
                'revisions': {stage: config[stage].model.get('revision') for stage in ('target', 'shadow', 'attack')},
                'metrics': recorded_metrics,
            }
            temporary = output_dir / 'metrics.json.tmp'
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
            temporary.replace(output_dir / 'metrics.json')

        del attack_model, attack_data
        release_memory(device)
        logging.info('Released attack model and data.')
    finally:
        writer.close()


@hydra.main(config_path='pkg://sralt.configs', config_name='config', version_base=None)
def main(config: DictConfig) -> None:
    run_pipeline(config)


if __name__ == '__main__':
    main()

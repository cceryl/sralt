# This file compares the 'random' ablation with 'base' setting and baseline.

import os
import logging
import warnings
import itertools

import torch
import torch.optim as optim
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torchmetrics import AUROC, F1Score, Precision, Recall

from sralt.target import load_target_data, load_target_model, TargetData
from sralt.shadow import load_shadow_data, load_shadow_model, ShadowData, ShadowModel
from sralt.attack import load_attack_data, load_attack_model, test_attack_model
from sralt.utils import setup_logging, get_device, disk_cache
from sralt.utils import AdamWithGradientClipping, TPRatFPR


def prepare_data(
    name: str, directory: str,
    target_seed: int, shadow_seed: int, device: torch.device
) -> tuple[TargetData, ShadowData]:
    config = {
        'min_social_connections': 1,
        'min_user_interactions': 5,
        'min_item_interactions': 1
    }

    if name in ['Ciao', 'Epinions']:
        config['min_rating'] = 2

    target_data = load_target_data(
        seed=target_seed,
        deterministic=True,
        name=name,
        directory=directory,
        config=config
    ).to(device)

    target_model = load_target_model(
        seed=target_seed,
        deterministic=True,
        device=device,
        name='DESIGN',
        model_config={
            'embed_dim': 64,
            'user_fusion_layers': [0, 1, 2],
            'item_fusion_layers': [0, 1, 2],
            'decay': 5e-3,
            'distill_weight': 1e-2
        },
        data=target_data,
        train_config={
            'train_batch_size': 32768,
            'val_batch_size': 4096,
            'val_num_negatives': 999,
            'num_epochs': 300,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': 1e-2},
            'metrics': [LinkPredRecall(k=10), LinkPredNDCG(k=10)],
            'metric_aggregator': 'lambda loss, metrics: sum(metrics)',
            'early_stopping_kwargs': {'patience': 10, 'greater_is_better': True}
        },
        writer=None
    )

    shadow_data = load_shadow_data(
        seed=shadow_seed,
        deterministic=True,
        target_data=target_data,
        target_model=target_model,
        shadow_social_ratio=0.1,
        num_recommendations=30,
        inference_batch_size=1024
    ).to(device)

    return target_data, shadow_data


def shadow_baseline(
    shadow_seed: int, device: torch.device, shadow_data: ShadowData
) -> ShadowModel:
    shadow_model = load_shadow_model(
        seed=shadow_seed,
        deterministic=True,
        device=device,
        name='SMIA',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-2
        },
        data=shadow_data,
        train_config={
            'train_batch_size': 8192,
            'val_batch_size': 2048,
            'val_num_negatives': 999,
            'num_epochs': 100,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': 1e-2},
            'metrics': [LinkPredRecall(k=10), LinkPredNDCG(k=10)],
            'metric_aggregator': 'lambda loss, metrics: sum(metrics)',
            'early_stopping_kwargs': {'patience': 10, 'greater_is_better': True}
        },
        writer=None
    )

    return shadow_model


def shadow_semantic_augmentation(
    shadow_seed: int, device: torch.device, shadow_data: ShadowData
) -> ShadowModel:
    shadow_model = load_shadow_model(
        seed=shadow_seed,
        deterministic=True,
        device=device,
        name='SemanticAugmentation',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-3,
            'rvq_num_quantizers': 8,
            'rvq_codebook_size': 256,
            'commitment_weight': 0.0,
            'without_quantization': False,
            'ablation_code': 'base',
            'ablation_feature': 'base'
        },
        data=shadow_data,
        train_config={
            'train_batch_size': 8192,
            'val_batch_size': 2048,
            'val_num_negatives': 999,
            'num_epochs': 300,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': 1e-2},
            'metrics': [LinkPredRecall(k=10), LinkPredNDCG(k=10)],
            'metric_aggregator': 'lambda loss, metrics: sum(metrics)',
            'early_stopping_kwargs': {'patience': 10, 'greater_is_better': True}
        },
        writer=None
    )

    return shadow_model


def shadow_random_augmentation(
    shadow_seed: int, device: torch.device, shadow_data: ShadowData
) -> ShadowModel:
    shadow_model = load_shadow_model(
        seed=shadow_seed,
        deterministic=True,
        device=device,
        name='SemanticAugmentation',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-3,
            'rvq_num_quantizers': 8,
            'rvq_codebook_size': 256,
            'commitment_weight': 0.0,
            'without_quantization': False,
            'ablation_code': 'random',
            'ablation_feature': 'base'
        },
        data=shadow_data,
        train_config={
            'train_batch_size': 8192,
            'val_batch_size': 2048,
            'val_num_negatives': 999,
            'num_epochs': 300,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': 1e-2},
            'metrics': [LinkPredRecall(k=10), LinkPredNDCG(k=10)],
            'metric_aggregator': 'lambda loss, metrics: sum(metrics)',
            'early_stopping_kwargs': {'patience': 10, 'greater_is_better': True}
        },
        writer=None
    )

    return shadow_model


def attack(
    attack_base_seed: int, num_seeds: int, device: torch.device,
    target_data: TargetData, shadow_data: ShadowData, shadow_model: ShadowModel
) -> dict[str, dict[str, tuple[float, float]]]:
    def attack_single_seed(seed: int) -> dict[str, dict[str, float]]:
        attack_data = load_attack_data(
            seed=seed,
            deterministic=True,
            target_data=target_data,
            shadow_data=shadow_data,
            shadow_model=shadow_model,
            val_ratio=0.1
        ).to(device)

        attack_model = load_attack_model(
            seed=seed,
            deterministic=True,
            device=device,
            name='SMIA',
            model_config={
                'dropout': 0.5,
                'leaky_relu_slope': 0.2,
                'pos_weight': 10.0
            },
            data=attack_data,
            tail_ratio=0.8,
            train_config={
                'train_batch_size': 512,
                'val_batch_size': 2048,
                'negative_ratio': 1.0,
                'alpha': 0.0,
                'num_epochs': 50,
                'optimizer_cls': AdamWithGradientClipping,
                'optimizer_kwargs': {'lr': 1e-3, 'weight_decay': 1e-4, 'grad_clip_norm': 1.0},
                'metrics': [
                    AUROC(task='binary'),
                    F1Score(task='binary'),
                    TPRatFPR(target_fpr=0.01),
                    Precision(task='binary'),
                    Recall(task='binary')
                ],
                'metric_aggregator': 'lambda loss, metrics: metrics[0] + metrics[1]',
                'early_stopping_kwargs': {'patience': 5, 'greater_is_better': True}
            },
            writer=None
        )

        test_metrics = test_attack_model(
            seed=seed,
            deterministic=True,
            device=device,
            model=attack_model,
            data=attack_data,
            tail_ratio=0.8,
            metrics=[
                AUROC(task='binary'),
                F1Score(task='binary'),
                TPRatFPR(target_fpr=0.01),
                Precision(task='binary'),
                Recall(task='binary')
            ],
            test_config={
                'test_batch_size': 2048
            }
        )

        results_dict = {}
        for group, metrics in zip(['All', 'Head', 'Tail'], test_metrics):
            results_dict[group] = {
                name: value for name, value in
                zip(['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall'], metrics)
            }

        return results_dict

    aggregated_results: dict[str, dict[str, tuple[float, float]]] = {}
    all_seed_results = [attack_single_seed(attack_base_seed + i) for i in range(num_seeds)]
    for group in ['All', 'Head', 'Tail']:
        aggregated_results[group] = {}
        for metric in ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']:
            values = [res[group][metric] for res in all_seed_results]
            mean_val = sum(values) / len(values)
            std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            aggregated_results[group][metric] = (mean_val, std_val)

    return aggregated_results


@disk_cache('.cache/random_{lower_name}_design')
def pipeline_single_set(
    name: str, directory: str, lower_name: str,
    target_seed: int, shadow_seed: int, attack_base_seed: int, attack_num_seeds: int,
    device: torch.device
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    target_data, shadow_data = prepare_data(
        name=name,
        directory=directory,
        target_seed=target_seed,
        shadow_seed=shadow_seed,
        device=device
    )

    results: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}

    funcs = [shadow_baseline, shadow_semantic_augmentation, shadow_random_augmentation]
    names = ['Baseline', 'Semantic', 'Random']

    for func, name in zip(funcs, names):
        shadow_model = func(
            shadow_seed=shadow_seed,
            device=device,
            shadow_data=shadow_data
        )

        attack_metrics = attack(
            attack_base_seed=attack_base_seed,
            num_seeds=attack_num_seeds,
            device=device,
            target_data=target_data,
            shadow_data=shadow_data,
            shadow_model=shadow_model
        )

        del shadow_model
        results[name] = attack_metrics

    return results


def logs(
    results: list[tuple[dict[str, int], dict[str, dict[str, dict[str, tuple[float, float]]]]]]
):
    def single_result_log(
        baseline_name: str, method_name: str,
        baseline_result: dict[str, dict[str, tuple[float, float]]],
        method_result: dict[str, dict[str, tuple[float, float]]]
    ) -> str:
        log_lines = [f'Results for {method_name} vs {baseline_name}:']
        for group in ['All', 'Head', 'Tail']:
            line = f'    {group} -> '
            metrics = []

            for metric in ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']:
                base_mean, _ = baseline_result[group][metric]
                method_mean, method_std = method_result[group][metric]
                metrics.append(
                    f'{metric}: {method_mean:.4f} ± {method_std:.4f} ({(method_mean - base_mean) / base_mean:+.2%})'
                )

            line += ', '.join(metrics)
            log_lines.append(line)

        return '\n'.join(log_lines) + '\n\n'

    for config, result in results:
        config_str = ', '.join(f'{k}={v}' for k, v in config.items())
        log_text = ''

        baseline_results = result['Baseline']
        semantic_results = result['Semantic']
        random_results = result['Random']

        # Semantic vs Baseline
        log_text += single_result_log(
            baseline_name='Baseline',
            method_name='Semantic',
            baseline_result=baseline_results,
            method_result=semantic_results
        )

        # Random vs Baseline
        log_text += single_result_log(
            baseline_name='Baseline',
            method_name='Random',
            baseline_result=baseline_results,
            method_result=random_results
        )

        # Random vs Semantic
        log_text += single_result_log(
            baseline_name='Semantic',
            method_name='Random',
            baseline_result=semantic_results,
            method_result=random_results
        )

        logging.info(
            f'\n'
            f'Configuration: {config_str}\n'
            f'{log_text}'
        )


def main():
    name = 'Flickr'
    directory = './data/flickr'

    root_path = f'runs/random/{name.lower()}-design/target_seed_test'
    os.makedirs(root_path)
    setup_logging(log_file=os.path.join(root_path, 'search.log'), tz='Asia/Shanghai')
    device = get_device(allow_share=True)

    target_seeds = [42, 43, 44, 45, 46]
    shadow_seeds = [42]
    attack_base_seed = 42
    attack_num_seeds = 10

    all_results: list[tuple[dict[str, int], dict[str, dict[str, dict[str, tuple[float, float]]]]]] = []
    for target_seed, shadow_seed in itertools.product(target_seeds, shadow_seeds):
        config = {
            'target_seed': target_seed,
            'shadow_seed': shadow_seed,
            'attack_base_seed': attack_base_seed,
            'attack_num_seeds': attack_num_seeds
        }

        results = pipeline_single_set(
            name=name,
            directory=directory,
            lower_name=name.lower(),
            target_seed=target_seed,
            shadow_seed=shadow_seed,
            attack_base_seed=attack_base_seed,
            attack_num_seeds=attack_num_seeds,
            device=device
        )

        all_results.append((config, results))

    logs(all_results)


if __name__ == '__main__':
    warnings.filterwarnings(
        'ignore',
        message=''
        'You are trying to use a metric in deterministic mode on GPU that uses `torch.cumsum`, which is currently not supported. '
        'The tensor will be copied to the CPU memory to compute it and then copied back to GPU. Expect some slowdowns.',
        module='torchmetrics.utilities.prints'
    )
    warnings.filterwarnings(
        'ignore',
        message='Sparse CSR tensor support is in beta state.'
    )

    main()

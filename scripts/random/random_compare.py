# This file provides comparison helpers for semantic models against baselines and random ablations.

import os
import logging
import warnings

import torch
import torch.optim as optim
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torchmetrics import AUROC, F1Score, Precision, Recall

import pickle
import pandas as pd

from sralt.target import load_target_data, load_target_model, TargetData
from sralt.shadow import load_shadow_data, load_shadow_model, ShadowData, ShadowModel
from sralt.attack import load_attack_data, load_attack_model, test_attack_model
from sralt.utils import setup_logging, get_device
from sralt.utils import AdamWithGradientClipping, TPRatFPR


def prepare_data(
    seed: int, deterministic: bool, device: torch.device
) -> tuple[TargetData, ShadowData]:
    target_data = load_target_data(
        seed=seed,
        deterministic=deterministic,
        name='Ciao',
        directory='./data/ciao',
        config={
            'min_rating': 2,
            'min_social_connections': 1,
            'min_user_interactions': 5,
            'min_item_interactions': 1
        }
    ).to(device)

    target_model = load_target_model(
        seed=seed,
        deterministic=deterministic,
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
        seed=seed,
        deterministic=deterministic,
        target_data=target_data,
        target_model=target_model,
        shadow_social_ratio=0.1,
        num_recommendations=30,
        inference_batch_size=1024
    ).to(device)

    return target_data, shadow_data


def shadow_stage(
    seed: int, deterministic: bool, device: torch.device, shadow_data: ShadowData, config: dict
) -> ShadowModel:
    shadow_model = load_shadow_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        name=config['name'],
        model_config=config['model'],
        data=shadow_data,
        train_config=config['train'],
        writer=None
    )

    return shadow_model


def attack_stage(
    base_seed: int, num_seeds: int, deterministic: bool, device: torch.device,
    target_data: TargetData, shadow_data: ShadowData, shadow_model: ShadowModel
) -> dict[str, dict[str, tuple[float, float]]]:
    def attack_single_seed(seed: int):
        attack_data = load_attack_data(
            seed=seed,
            deterministic=deterministic,
            target_data=target_data,
            shadow_data=shadow_data,
            shadow_model=shadow_model,
            val_ratio=0.1
        ).to(device)

        attack_model = load_attack_model(
            seed=seed,
            deterministic=deterministic,
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
            deterministic=deterministic,
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

    # Aggregate results over multiple seeds as (mean, std)
    aggregated_results = {}
    all_seed_results = [attack_single_seed(base_seed + i) for i in range(num_seeds)]
    for group in ['All', 'Head', 'Tail']:
        aggregated_results[group] = {}
        for metric in ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']:
            values = [res[group][metric] for res in all_seed_results]
            mean_val = sum(values) / len(values)
            std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            aggregated_results[group][metric] = (mean_val, std_val)

    return aggregated_results


def test_config(config: dict, save_path: str):
    """
    :param config: A dictionary containing 'baseline', 'semantic', and 'random' configurations. In each one, there should be a 'name' key
                   for the shadow model name, a 'model' key for the shadow model configuration, and a 'train' key for the shadow model
                   training configuration.
    :param save_path: The path where the results should be saved.
    """
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

    os.makedirs(save_path, exist_ok=True)
    setup_logging(log_file=os.path.join(save_path, 'test.log'), tz='Asia/Shanghai')
    device = get_device(allow_share=True)
    data_kwargs = {'seed': 42, 'deterministic': True, 'device': device}

    # Prepare data
    target_data, shadow_data = prepare_data(**data_kwargs)
    shadow_kwargs = {'seed': 42, 'deterministic': True, 'device': device, 'shadow_data': shadow_data}
    attack_kwargs = {
        'base_seed': 42, 'num_seeds': 10, 'deterministic': True, 'device': device, 'target_data': target_data, 'shadow_data': shadow_data
    }

    # Test each configuration
    def test_single_config(config_name: str):
        shadow_model = shadow_stage(**shadow_kwargs, config=config[config_name])
        results = attack_stage(**attack_kwargs, shadow_model=shadow_model)
        return results

    baseline_res = test_single_config('baseline')
    semantic_res = test_single_config('semantic')
    random_res = test_single_config('random')

    # Pickle
    with open(os.path.join(save_path, 'results.pkl'), 'wb') as f:
        pickle.dump({'baseline': baseline_res, 'semantic': semantic_res, 'random': random_res}, f)

    # Sheet data
    groups = ['All', 'Head', 'Tail']
    all_metrics = ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']
    concise_metrics = ['AUROC', 'F1', 'TPR@FPR=0.01']

    full_data = []
    concise_data = []

    for group in groups:
        for i, metric in enumerate(all_metrics):
            s_mean, s_std = semantic_res[group][metric]
            b_mean, b_std = baseline_res[group][metric]
            r_mean, r_std = random_res[group][metric]

            vs_baseline = ((s_mean - b_mean) / b_mean * 100) if b_mean != 0 else 0
            vs_random = ((s_mean - r_mean) / r_mean * 100) if r_mean != 0 else 0

            group_str = group if i == 0 else ''
            vs_baseline_str = f"{vs_baseline:+.2f}%"
            vs_random_str = f"{vs_random:+.2f}%"

            full_row = {
                'Group': group_str,
                'Metric': metric,
                'Semantic Avg': round(s_mean, 4),
                'Semantic Std': round(s_std, 4),
                'Baseline Avg': round(b_mean, 4),
                'Baseline Std': round(b_std, 4),
                'Δ Baseline (%)': vs_baseline_str,
                'Random Avg': round(r_mean, 4),
                'Random Std': round(r_std, 4),
                'Δ Random (%)': vs_random_str
            }
            full_data.append(full_row)

            if metric in concise_metrics:
                concise_row = {
                    'Group': group_str,
                    'Metric': metric,
                    'Semantic': round(s_mean, 4),
                    'Baseline': round(b_mean, 4),
                    'Δ Baseline (%)': vs_baseline_str,
                    'Random': round(r_mean, 4),
                    'Δ Random (%)': vs_random_str
                }
                concise_data.append(concise_row)

    # Create DataFrames
    df_full = pd.DataFrame(full_data)
    df_concise = pd.DataFrame(concise_data)

    # Save to Excel
    filename = os.path.join(save_path, 'comparison.xlsx')
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df_full.to_excel(writer, sheet_name='Full', index=False)
        df_concise.to_excel(writer, sheet_name='Concise', index=False)

        for sheet_name in ['Full', 'Concise']:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:B', 15)
            worksheet.set_column('C:K', 15)

    logging.info(f"Results saved to {filename}")

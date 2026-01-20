# This tests a balanced attack model experiment script that performs grid search over different balance loss weights.

from time import time
import os
import logging
import warnings

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torchmetrics import AUROC, F1Score, Precision, Recall

from sralt.target import load_target_data, load_target_model
from sralt.shadow import load_shadow_data, load_shadow_model
from sralt.attack import load_attack_data, load_attack_model, test_attack_model, visualize_attack_probs, AttackData
from sralt.utils import setup_logging, get_device, AdamWithGradientClipping, TPRatFPR


def run_attack_model_experiment(
    runs_path: str, seed: int, deterministic: bool, device: torch.device, data: AttackData,
    balance_loss_weight: float = 1.0
) -> tuple[list[float], list[float], list[float]]:
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

    attack_model = load_attack_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        name='Balance',
        model_config={
            'dropout': 0.5,
            'leaky_relu_slope': 0.2,
            'pos_weight': 10.0
        },
        data=data,
        tail_ratio=0.8,
        train_config={
            'train_batch_size': 4096,
            'val_batch_size': 2048,
            'negative_ratio': 5.0,
            'alpha': 0.0,
            'num_epochs': 300,
            'optimizer_cls': AdamWithGradientClipping,
            'optimizer_kwargs': {'lr': 1e-4, 'weight_decay': 1e-4, 'grad_clip_norm': 1.0},
            'balance_loss_weight': balance_loss_weight,
            'metrics': [
                AUROC(task='binary'),
                F1Score(task='binary'),
                TPRatFPR(target_fpr=0.01),
                Precision(task='binary'),
                Recall(task='binary')
            ],
            'metric_aggregator': 'lambda loss, metrics: metrics[0] + metrics[1]',
            'early_stopping_kwargs': {'patience': 30, 'greater_is_better': True}
        },
        writer=writer
    )

    test_metrics = test_attack_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        model=attack_model,
        data=data,
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

    for group, metrics in zip(['All', 'Head', 'Tail'], test_metrics):
        for name, value in zip(['AUROC', 'F1 Score', 'TPR@FPR=0.01', 'Precision', 'Recall'], metrics):
            logging.info(f'Test {group} {name}: {value:.4f}')

    visualize_attack_probs(
        attack_model=attack_model,
        attack_data=data,
        tail_ratio=0.8,
        seed=seed,
        save_path=f'{runs_path}/attack_prob_distribution.png'
    )

    writer.close()

    return test_metrics


def prepare() -> tuple[str, int, bool, torch.device, AttackData]:
    runs_path = f'runs/explore/balance/grid_search/{int(time())}'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'prepare.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

    seed = 42
    deterministic = True
    device = get_device(allow_share=True)

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
        writer=writer
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

    shadow_model = load_shadow_model(
        seed=seed,
        deterministic=deterministic,
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
        writer=writer
    )

    attack_data = load_attack_data(
        seed=seed,
        deterministic=deterministic,
        target_data=target_data,
        shadow_data=shadow_data,
        shadow_model=shadow_model,
        val_ratio=0.1
    ).to(device)

    return runs_path, seed, deterministic, device, attack_data


def main():
    base_runs_path, seed, deterministic, device, attack_data = prepare()

    all_test_metrics: list[tuple[float, tuple[list[float], list[float], list[float]]]] = []

    for idx, balance_loss_weight in enumerate([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0]):
        logging.info(
            f'Starting experiment {idx + 1} with balance_loss_weight: {balance_loss_weight}')
        runs_path = os.path.join(base_runs_path, f'experiment_{idx + 1}')

        test_metrics = run_attack_model_experiment(
            runs_path=runs_path,
            seed=seed,
            deterministic=deterministic,
            device=device,
            data=attack_data,
            balance_loss_weight=balance_loss_weight
        )
        all_test_metrics.append((balance_loss_weight, test_metrics))

    def get_total_score(metrics: tuple[list[float], list[float], list[float]]) -> float:
        all_metrics, head_metrics, tail_metrics = metrics

        all_auroc, all_f1, all_tpr, _, _ = all_metrics
        head_auroc, head_f1, head_tpr, _, _ = head_metrics
        tail_auroc, tail_f1, tail_tpr, _, _ = tail_metrics

        # we give all metrics a weight of 1.0, head metrics a weight of 0.5, tail metrics a weight of 1.5
        # and among these, tpr is considered twice as important as auroc and f1
        total_score = (
            1.0 * (all_auroc + all_f1 + 2.0 * all_tpr) +
            0.5 * (head_auroc + head_f1 + 2.0 * head_tpr) +
            1.5 * (tail_auroc + tail_f1 + 2.0 * tail_tpr)
        )
        return total_score

    all_test_metrics.sort(key=lambda x: get_total_score(x[1]), reverse=True)
    for idx, (params, metrics) in enumerate(all_test_metrics):
        # we only what to log AUROC, F1 Score, TPR@FPR=0.01 for three groups: All, Head, Tail
        filtered_metrics = (
            [metrics[0][0], metrics[0][1], metrics[0][2]],  # All
            [metrics[1][0], metrics[1][1], metrics[1][2]],  # Head
            [metrics[2][0], metrics[2][1], metrics[2][2]]   # Tail
        )
        filtered_metrics_str = [[f'{m:.4f}' for m in group] for group in filtered_metrics]
        logging.info(f'Rank {idx + 1}: Params: {params}, Metrics: {filtered_metrics_str}')


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

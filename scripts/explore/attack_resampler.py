# This file tests the variations of sampler.

import os
import logging
import warnings

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torchmetrics import AUROC, F1Score, Precision, Recall

from sralt.target import load_target_data, load_target_model, test_target_model, TargetData, TargetModel
from sralt.shadow import load_shadow_data, load_shadow_model, test_shadow_model, ShadowData, ShadowModel
from sralt.attack import load_attack_data, load_attack_model, test_attack_model
from sralt.utils import setup_logging, get_device, AdamWithGradientClipping, TPRatFPR


def target_stage(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter | None
) -> tuple[TargetData, TargetModel]:
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

    test_metrics = test_target_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        model=target_model,
        data=target_data,
        metrics=[LinkPredRecall(k=10), LinkPredNDCG(k=10)],
        test_config={
            'test_batch_size': 4096,
            'test_num_negatives': 999
        }
    )

    for name, value in zip(['Recall@10', 'NDCG@10'], test_metrics):
        logging.info(f'Target model test {name}: {value:.4f}')

    logging.info(f'Target model representation: {target_model}')
    return target_data, target_model


def shadow_stage(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter | None,
    target_data: TargetData, target_model: TargetModel
) -> tuple[ShadowData, ShadowModel]:
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

    test_metrics = test_shadow_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        model=shadow_model,
        data=shadow_data,
        metrics=[LinkPredRecall(k=10), LinkPredNDCG(k=10)],
        test_config={
            'test_batch_size': 2048,
            'test_num_negatives': 999
        }
    )

    for name, value in zip(['Recall@10', 'NDCG@10'], test_metrics):
        logging.info(f'Shadow model test {name}: {value:.4f}')

    logging.info(f'Shadow model representation: {shadow_model}')
    return shadow_data, shadow_model


def attack_stage(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter | None,
    target_data: TargetData, shadow_data: ShadowData, shadow_model: ShadowModel,
    negative_ratio: float, alpha: float
) -> tuple[list[float], list[float], list[float]]:
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
            'train_batch_size': 256,
            'val_batch_size': 2048,
            'negative_ratio': negative_ratio,
            'alpha': alpha,
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
        writer=writer
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

    logging.info(f'Using negative_ratio={negative_ratio:.2f}, alpha={alpha:.2f}')
    for group, metrics in zip(['All', 'Head', 'Tail'], test_metrics):
        for name, value in zip(['AUROC', 'F1 Score', 'TPR@FPR=0.01', 'Precision', 'Recall'], metrics):
            logging.info(f'Test {group} {name}: {value:.4f}')

    logging.info(f'Attack model representation: {attack_model}')
    return test_metrics


def main():
    runs_path = 'runs/explore/attack_resampler'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = None

    seed = 42
    deterministic = True
    device = get_device(allow_share=True)

    target_data, target_model = target_stage(
        seed=seed,
        deterministic=deterministic,
        device=device,
        writer=writer
    )

    shadow_data, shadow_model = shadow_stage(
        seed=seed,
        deterministic=deterministic,
        device=device,
        writer=writer,
        target_data=target_data,
        target_model=target_model
    )

    common_attack_args = {
        'seed': seed,
        'deterministic': deterministic,
        'device': device,
        'writer': writer,
        'target_data': target_data,
        'shadow_data': shadow_data,
        'shadow_model': shadow_model,

        'negative_ratio': 1.0,
        'alpha': 0.0
    }

    test_metrics: dict[tuple[float, float], tuple[list[float], list[float], list[float]]] = {}

    for ratio in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        test_metrics[ratio, 0.0] = attack_stage(**{**common_attack_args, 'negative_ratio': ratio})

    for alpha in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        test_metrics[1.0, alpha] = attack_stage(**{**common_attack_args, 'alpha': alpha})

    def metric_score(metrics: tuple[list[float], list[float], list[float]]) -> float:
        def weighted_sum(values: list[float]) -> float:
            auc, f1, tpr, _, _ = values
            return auc + f1 + tpr * 2

        all, head, tail = metrics
        return weighted_sum(all) + 0.5 * weighted_sum(head) + 2.0 * weighted_sum(tail)

    sorted_metrics = sorted(
        test_metrics.items(),
        key=lambda item: metric_score(item[1]),
        reverse=True
    )

    for (ratio, alpha), metrics in sorted_metrics:
        all, head, tail = metrics
        logging.info(f'Negative Ratio: {ratio:.2f}, Alpha: {alpha:.2f}')
        logging.info(
            f'  All  - AUROC: {all[0]:.4f}, F1: {all[1]:.4f}, TPR@FPR=0.01: {all[2]:.4f}, Precision: {all[3]:.4f}, Recall: {all[4]:.4f}'
        )
        logging.info(
            f'  Head - AUROC: {head[0]:.4f}, F1: {head[1]:.4f}, TPR@FPR=0.01: {head[2]:.4f}, Precision: {head[3]:.4f}, Recall: {head[4]:.4f}'
        )
        logging.info(
            f'  Tail - AUROC: {tail[0]:.4f}, F1: {tail[1]:.4f}, TPR@FPR=0.01: {tail[2]:.4f}, Precision: {tail[3]:.4f}, Recall: {tail[4]:.4f}'
        )


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

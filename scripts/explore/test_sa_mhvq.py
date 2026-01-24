# This file is an example script for training and evaluating a pipeline of target, shadow, and attack models.

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
from sralt.attack import load_attack_data, load_attack_model, test_attack_model, AttackData, AttackModel
from sralt.utils import setup_logging, get_device, AdamWithGradientClipping, TPRatFPR


def target_stage(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter
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
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter,
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
        name='SemanticAugmentationMHVQ',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-3,
            'mhvq_num_quantizers': 8,
            'mhvq_codebook_size': 256,
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
            'test_num_negatives': 999,
            'stage': 'social'
        }
    )

    for name, value in zip(['Recall@10', 'NDCG@10'], test_metrics):
        logging.info(f'Shadow model test {name}: {value:.4f}')

    logging.info(f'Shadow model representation: {shadow_model}')
    return shadow_data, shadow_model


def attack_stage(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter,
    target_data: TargetData, shadow_data: ShadowData, shadow_model: ShadowModel
) -> tuple[AttackData, AttackModel]:
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

    for group, metrics in zip(['All', 'Head', 'Tail'], test_metrics):
        for name, value in zip(['AUROC', 'F1 Score', 'TPR@FPR=0.01', 'Precision', 'Recall'], metrics):
            logging.info(f'Test {group} {name}: {value:.4f}')

    logging.info(f'Attack model representation: {attack_model}')
    return attack_data, attack_model


def main():
    runs_path = 'runs/explore/test_sa_mhvq'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

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

    attack_data, attack_model = attack_stage(
        seed=seed,
        deterministic=deterministic,
        device=device,
        writer=writer,
        target_data=target_data,
        shadow_data=shadow_data,
        shadow_model=shadow_model
    )

    pipeline = {
        'target_data': target_data,
        'target_model': target_model.state_dict(),
        'shadow_data': shadow_data,
        'shadow_model': shadow_model.state_dict(),
        'attack_data': attack_data,
        'attack_model': attack_model.state_dict()
    }
    torch.save(pipeline, os.path.join(runs_path, 'pipeline.pth'))
    writer.close()


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

# This tests the Semantic Augmentation (SA) based hybrid attack model with hyperparameter search.

from time import time
import os
import logging
import warnings
import itertools

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torchmetrics import AUROC, F1Score, Precision, Recall

from sralt.target.data import load_target_data, TargetData
from sralt.target.model import load_target_model, test_target_model, TargetModel
from sralt.shadow.data import load_shadow_data, ShadowData
from sralt.shadow.model import load_shadow_model, test_shadow_model, ShadowModel
from sralt.attack.data import load_attack_data, AttackData
from sralt.attack.model import load_attack_model, test_attack_model, AttackModel
from sralt.utils import setup_logging, get_device
from sralt.utils import AdamWithGradientClipping, TPRatFPR


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


def shadow_stage_baseline(
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


def shadow_stage_sa(
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
        name='SemanticAugmentation',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-2,
            'rvq_num_quantizers': 4,
            'rvq_codebook_size': 16,
            'commitment_weight': 0.25
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
            'test_num_negatives': 999,
            'stage': 'social'
        }
    )

    for name, value in zip(['Recall@10', 'NDCG@10'], test_metrics):
        logging.info(f'Shadow model test {name}: {value:.4f}')

    logging.info(f'Shadow model representation: {shadow_model}')
    return shadow_data, shadow_model


def shadow_stage_sa_search(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter,
    target_data: TargetData, target_model: TargetModel, sa_config: dict
) -> tuple[ShadowData, ShadowModel, list[float]]:
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
        name='SemanticAugmentation',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': sa_config['decay'],
            'rvq_num_quantizers': sa_config['rvq_num_quantizers'],
            'rvq_codebook_size': sa_config['rvq_codebook_size'],
            'commitment_weight': sa_config['commitment_weight'],
            'without_quantization': sa_config['without_quantization'],
            'ablation_code': sa_config['ablation_code'],
            'ablation_feature': sa_config['ablation_feature']
        },
        data=shadow_data,
        train_config={
            'train_batch_size': 8192,
            'val_batch_size': 2048,
            'val_num_negatives': 999,
            'num_epochs': 300,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': sa_config['lr']},
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
        test_config={'test_batch_size': 2048, 'test_num_negatives': 999, 'stage': 'social'}
    )
    return shadow_data, shadow_model, test_metrics


def attack_stage(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter,
    target_data: TargetData, shadow_data: ShadowData, shadow_model: ShadowModel
) -> tuple[AttackData, AttackModel, tuple[list[float], list[float], list[float]]]:
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
        name='Hybrid',
        model_config={
            'fusion': ['sum', 'diff'],
            'low_rank': 16,
            'attention_heads': 1,
            'num_experts': 2,
            'dropout': 0.5,
            'noise_multiplier': 1e-2,
            'pos_weight': 10.0,
            'degree_weight_factor': 5.0,
            'feature_weight_factor': 2.0
        },
        data=attack_data,
        tail_ratio=0.8,
        train_config={
            'train_batch_size': 256,
            'val_batch_size': 2048,
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
    return attack_data, attack_model, test_metrics


def main():
    root_path = f'runs/explore/sa/grid_hybrid_{int(time())}'
    os.makedirs(root_path, exist_ok=True)
    setup_logging(log_file=os.path.join(root_path, 'search.log'), tz='Asia/Shanghai')

    seed = 42
    deterministic = True
    device = get_device(allow_share=True)

    target_writer = SummaryWriter(log_dir=os.path.join(root_path, 'target'))
    target_data, target_model = target_stage(seed, deterministic, device, target_writer)

    param_grid = {
        'decay': [1e-2],
        'rvq_num_quantizers': [1, 2, 3, 4, 6, 8],
        'rvq_codebook_size': [16, 32, 64, 256],
        'commitment_weight': [0.0],
        'lr': [1e-2],
        'without_quantization': [False],
        'ablation_code': ['base'],
        'ablation_feature': ['base']
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    logging.info(f"Starting Hyperparameter Search. Total combinations: {len(combinations)}")

    results = []

    for i, config in enumerate(combinations):
        run_name = f"run_{i}_" + "_".join([f"{k}={v}" for k, v in config.items()])
        logging.info(f"\n{'='*20} [{i+1}/{len(combinations)}] Testing: {run_name} {'='*20}")

        run_path = os.path.join(root_path, run_name)
        writer = SummaryWriter(log_dir=run_path)

        try:
            shadow_data, shadow_model, s_metrics = shadow_stage_sa_search(
                seed, deterministic, device, writer, target_data, target_model, config
            )

            _, _, a_metrics = attack_stage(seed, deterministic, device, writer, target_data, shadow_data, shadow_model)

            s_dict = {n: v for n, v in zip(['Recall@10', 'NDCG@10'], s_metrics)}
            a_dict = {g: {n: v for n, v in zip(
                ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall'], a_metrics[idx]
            )} for idx, g in enumerate(['All', 'Head', 'Tail'])}

            res = {'config': config, 'shadow': s_dict, 'attack': a_dict}
            results.append(res)
            logging.info(
                f"Result for this run: AUROC={a_dict['All']['AUROC']:.4f}, F1={a_dict['All']['F1']:.4f}, "
                f"TPR@FPR=0.01={a_dict['All']['TPR@FPR=0.01']:.4f}"
            )

            # Save intermediate results
            results_path = os.path.join(root_path, 'intermediate_results.txt')
            with open(results_path, 'w') as f:
                for r in results:
                    f.write(str(r) + '\n')

        except Exception as e:
            logging.error(f"Run {i} failed with error: {e}")

        writer.close()

    logging.info("\n" + "#"*30 + " SEARCH SUMMARY " + "#"*30)

    def weighted_result(result: dict[str, dict[str, float]]):
        all = result['All']
        head = result['Head']
        tail = result['Tail']

        all_val = all['AUROC'] + all['F1'] + all['TPR@FPR=0.01'] * 2.0
        head_val = head['AUROC'] + head['F1'] + head['TPR@FPR=0.01'] * 2.0
        tail_val = tail['AUROC'] + tail['F1'] + tail['TPR@FPR=0.01'] * 2.0

        return all_val + head_val * 0.5 + tail_val * 2.0

    sorted_results = sorted(results, key=lambda x: weighted_result(x['attack']), reverse=True)

    for rank, res in enumerate(sorted_results[:10]):
        attack = res['attack']

        logging.info(
            f"\n"
            f"Rank {rank+1}: config={res['config']}\n"
            f"   All:  {', '.join([f'{k}={v:.4f}' for k, v in attack['All'].items()])}\n"
            f"   Head: {', '.join([f'{k}={v:.4f}' for k, v in attack['Head'].items()])}\n"
            f"   Tail: {', '.join([f'{k}={v:.4f}' for k, v in attack['Tail'].items()])}\n"
        )

    logging.info(f"Full results saved to {root_path}")


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

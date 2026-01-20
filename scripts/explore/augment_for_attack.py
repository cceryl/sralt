# This file explores the sa shadow model and compares its performance with a baseline shadow model.

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
from sralt.shadow.model.sa import SemanticAugmentation
from sralt.attack.data import load_attack_data, AttackData
from sralt.attack.model import load_attack_model, test_attack_model, AttackModel
from sralt.utils import setup_logging, get_device
from sralt.utils import AdamWithGradientClipping, TPRatFPR


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


def shadow_stage_baseline(
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


def shadow_stage_sa(
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
        name='SemanticAugmentation',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-2,
            'rvq_num_quantizers': 4,
            'rvq_codebook_size': 64,
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


def attack_stage(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter | None,
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
            'early_stopping_kwargs': {'patience': 5, 'greater_is_better': True},
            '_prevent_cache_hit': time()
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


def attack_stage_with_augment(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter | None,
    target_data: TargetData, shadow_data: ShadowData, shadow_model: ShadowModel, max_augments: int = 5
) -> tuple[AttackData, AttackModel, tuple[list[float], list[float], list[float]]]:
    attack_data = load_attack_data(
        seed=seed,
        deterministic=deterministic,
        target_data=target_data,
        shadow_data=shadow_data,
        shadow_model=shadow_model,
        val_ratio=0.1
    ).to(device)

    # Monkey patching: augment the shadow_social_edge_index with semantic augmentation given by the shadow model
    with torch.no_grad():
        from torch_geometric.utils import to_undirected, coalesce
        from sralt.utils import set_seed

        # get semantic augmentation user-code edges
        assert isinstance(shadow_model, SemanticAugmentation)
        sa_edges = shadow_model.augment_social_graph(shadow_data.social_edge_index)

        # we convert user-code edges to user-user edges by making the bipartite graph to a complete graph with only users
        # say u1, u2, u3 are connected to code c1, then we add edges (u1, u2), (u1, u3), (u2, u3)
        # however, this may create a large number of edges, so we only a part of them, which is controlled by `max_augments`
        user_code_sa_edges = sa_edges[:, sa_edges[0] < shadow_data.num_users]
        code_to_users: dict[int, list[int]] = {}
        for user_node, code_node in user_code_sa_edges.t().tolist():
            if code_node not in code_to_users:
                code_to_users[code_node] = []
            code_to_users[code_node].append(user_node)

        total_augments = []
        for users in code_to_users.values():
            if len(users) < 2:
                continue
            user_pairs = list(itertools.combinations(users, 2))
            total_augments.extend(user_pairs)
        set_seed(seed, deterministic=deterministic)
        total_augments = torch.tensor(total_augments, device=attack_data.device).t()
        perm = torch.randperm(total_augments.size(1), device=attack_data.device)
        augments = total_augments[:, perm[:max_augments]]

        # modify shadow_social_edge_index (this will affect train/val edges, but not test edges)
        attack_data.shadow_social_edge_index = coalesce(torch.cat(
            [attack_data.shadow_social_edge_index, augments], dim=1
        ))

        # Re-split train and val edges
        def to_directed(edge_index: torch.Tensor) -> torch.Tensor:
            return edge_index[:, edge_index[0] < edge_index[1]]

        set_seed(seed, deterministic=deterministic)
        directed_shadow_edges = to_directed(attack_data.shadow_social_edge_index)
        perm = torch.randperm(directed_shadow_edges.size(1), device=attack_data.device)
        num_val_edges = int(attack_data.val_ratio * directed_shadow_edges.size(1))

        attack_data.train_social_edge_index = directed_shadow_edges[:, perm[num_val_edges:]]
        attack_data.val_social_edge_index = directed_shadow_edges[:, perm[:num_val_edges]]

    attack_model = load_attack_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        name='SMIA',
        model_config={
            'dropout': 0.5,
            'leaky_relu_slope': 0.2,
            'pos_weight': 1.0
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
            'early_stopping_kwargs': {'patience': 5, 'greater_is_better': True},
            '_prevent_cache_hit': time()
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


def metric_to_dict(metrics: tuple[list[float], list[float], list[float]]) -> dict[str, dict[str, float]]:
    result = {'All': {}, 'Head': {}, 'Tail': {}}
    groups = ['All', 'Head', 'Tail']
    metric_names = ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']

    for group, group_metrics in zip(groups, metrics):
        for name, value in zip(metric_names, group_metrics):
            result[group][name] = value

    return result


def cal_mean_std(results: list) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    mean_metrics = {'All': {}, 'Head': {}, 'Tail': {}}
    std_metrics = {'All': {}, 'Head': {}, 'Tail': {}}
    for group in ['All', 'Head', 'Tail']:
        for metric in ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']:
            values = [res['attack'][group][metric] for res in results]
            mean_metrics[group][metric] = sum(values) / len(values)
            std_metrics[group][metric] = (sum((x - mean_metrics[group][metric]) **
                                          2 for x in values) / len(values)) ** 0.5
    return mean_metrics, std_metrics


def test_baseline_shadow_and_attack(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter | None,
    target_data: TargetData, target_model: TargetModel, num_attack_seeds: int = 10
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    shadow_data, shadow_model = shadow_stage_baseline(
        seed=seed,
        deterministic=deterministic,
        device=device,
        writer=writer,
        target_data=target_data,
        target_model=target_model
    )

    results = []

    for attack_seed in range(seed, seed + num_attack_seeds):
        _, _, attack_metrics = attack_stage(
            seed=attack_seed,
            deterministic=deterministic,
            device=device,
            writer=writer,
            target_data=target_data,
            shadow_data=shadow_data,
            shadow_model=shadow_model
        )

        results.append({
            'seed': attack_seed,
            'attack': metric_to_dict(attack_metrics)
        })

    return cal_mean_std(results)


def test_sa_shadow_and_attack(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter | None,
    target_data: TargetData, target_model: TargetModel, num_attack_seeds: int = 10
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    shadow_data, shadow_model = shadow_stage_sa(
        seed=seed,
        deterministic=deterministic,
        device=device,
        writer=writer,
        target_data=target_data,
        target_model=target_model
    )

    results = []

    for attack_seed in range(seed, seed + num_attack_seeds):
        _, _, attack_metrics = attack_stage(
            seed=attack_seed,
            deterministic=deterministic,
            device=device,
            writer=writer,
            target_data=target_data,
            shadow_data=shadow_data,
            shadow_model=shadow_model
        )

        results.append({
            'seed': attack_seed,
            'attack': metric_to_dict(attack_metrics)
        })

    return cal_mean_std(results)


def test_sa_shadow_and_augmented_attack(
    seed: int, deterministic: bool, device: torch.device, writer: SummaryWriter | None,
    target_data: TargetData, target_model: TargetModel, num_attack_seeds: int = 10, augment_ratio: float = 5.0
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    shadow_data, shadow_model = shadow_stage_sa(
        seed=seed,
        deterministic=deterministic,
        device=device,
        writer=writer,
        target_data=target_data,
        target_model=target_model
    )

    results = []

    for attack_seed in range(seed, seed + num_attack_seeds):
        _, _, attack_metrics = attack_stage_with_augment(
            seed=attack_seed,
            deterministic=deterministic,
            device=device,
            writer=writer,
            target_data=target_data,
            shadow_data=shadow_data,
            shadow_model=shadow_model,
            max_augments=int(shadow_data.num_social_edges * augment_ratio)
        )

        results.append({
            'seed': attack_seed,
            'attack': metric_to_dict(attack_metrics)
        })

    return cal_mean_std(results)


def main():
    root_path = f'runs/explore/sa/try_augment_social_{int(time())}'
    os.makedirs(root_path, exist_ok=True)
    setup_logging(log_file=os.path.join(root_path, 'search.log'), tz='Asia/Shanghai')

    seed = 42
    deterministic = True
    device = get_device(allow_share=True)

    target_data, target_model = target_stage(seed, deterministic, device, None)

    baseline_mean, baseline_std = test_baseline_shadow_and_attack(
        seed, deterministic, device, None, target_data, target_model
    )


    sa_mean, sa_std = test_sa_shadow_and_attack(
        seed, deterministic, device, None, target_data, target_model
    )

    sa_aug_mean, sa_aug_std = test_sa_shadow_and_augmented_attack(
        seed, deterministic, device, None, target_data, target_model, augment_ratio=1.0
    )

    data = {
        'baseline': {
            'mean': baseline_mean,
            'std': baseline_std
        },
        'sa': {
            'mean': sa_mean,
            'std': sa_std
        },
        'sa aug': {
            'mean': sa_aug_mean,
            'std': sa_aug_std
        }
    }

    for method, metrics in data.items():
        logging.info(f'============= {method} ===============')
        for group in ['All', 'Head', 'Tail']:
            for metric in ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']:
                baseline_mean_value = baseline_mean[group][metric]
                mean_value = metrics['mean'][group][metric]
                std_value = metrics['std'][group][metric]
                logging.info(
                    f'{group} {metric}: {mean_value:.4f} ± {std_value:.4f} (CV: {std_value / mean_value:.4f}, '
                    f'improvement {(mean_value - baseline_mean_value) / baseline_mean_value * 100:.2f}%)')


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

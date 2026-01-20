# This file tests the randomness from TrainAttackDataset (negative sampling during training).
# We fix all other random sources and only vary the seed for training negative sampling.

import os
import logging
import warnings

import torch
import torch.optim as optim
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torchmetrics import AUROC, F1Score, Precision, Recall

from sralt.target import load_target_data, load_target_model
from sralt.shadow import load_shadow_data, load_shadow_model
from sralt.attack import load_attack_data, load_attack_model, test_attack_model
from sralt.attack.data.dataset import TrainAttackDataset
from sralt.utils import setup_logging, get_device, set_seed, AdamWithGradientClipping, TPRatFPR


def prepare_attack_data(
    seed: int,
    deterministic: bool,
    device: torch.device
):
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
        writer=None
    )

    attack_data = load_attack_data(
        seed=seed,
        deterministic=deterministic,
        target_data=target_data,
        shadow_data=shadow_data,
        shadow_model=shadow_model,
        val_ratio=0.1
    ).to(device)

    return attack_data


def attack_with_controlled_train_negsampling(
    attack_data,
    base_seed: int,
    deterministic: bool,
    device: torch.device
):
    """
    Train attack model with controlled negative sampling seed.
    Only training negative sampling uses train_negsample_seed.
    """
    from torch_geometric.utils import to_undirected, negative_sampling
    from sralt.utils import rebalanced_negative_sampling
    
    # Monkey patch TrainAttackDataset.__getitem__ to remove per-epoch resampling
    original_getitem = TrainAttackDataset.__getitem__
    
    def patched_getitem(self, idx: int):
        if self.access_counter == 0 and not hasattr(self, '_flag'):
            self._flag = True  # Ensure resampling only once per epoch

            sample_args = {
                'edge_index': to_undirected(self.pos_edge_index),
                'num_nodes': self.attack_data.num_users,
                'num_neg_samples': self.num_negatives
            }

            if self.alpha == 0.0:
                self.neg_edge_index = negative_sampling(**sample_args, method='dense')
            else:
                self.neg_edge_index = rebalanced_negative_sampling(**sample_args, alpha=self.alpha)

            self.edge_index = torch.cat([self.pos_edge_index, self.neg_edge_index], dim=1)

        self.access_counter = (self.access_counter + 1) % len(self)

        return self.edge_index[0, idx], self.edge_index[1, idx], self.labels[idx]
    
    TrainAttackDataset.__getitem__ = patched_getitem

    attack_model = load_attack_model(
        seed=base_seed,
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
            'negative_ratio': 10.0,
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
            '_var_to_prevent_cache_hit': base_seed + 1000  # Ensure different seed for negative sampling
        },
        writer=None
    )

    # Restore original method
    TrainAttackDataset.__getitem__ = original_getitem

    test_metrics = test_attack_model(
        seed=base_seed,
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
            logging.info(f'[Train Seed {base_seed}] Test {group} {name}: {value:.4f}')

    return test_metrics


def compute_comprehensive_randomness_score(test_metrics):
    """
    Compute comprehensive randomness score based on variance of key metrics.
    
    Considers AUC, F1, TPR@FPR=0.01 with group weights:
    - All: 1.0
    - Head: 0.5
    - Tail: 2.0
    
    Returns:
        float: Weighted average of coefficient of variation (CV) across metrics and groups
    """
    metric_indices = {
        'AUROC': 0,
        'F1 Score': 1,
        'TPR@FPR=0.01': 2
    }
    group_weights = {'All': 1.0, 'Head': 0.5, 'Tail': 2.0}
    
    total_cv = 0.0
    total_weight = 0.0
    
    for group_idx, group in enumerate(['All', 'Head', 'Tail']):
        for metric_name, metric_idx in metric_indices.items():
            values = [test_metrics[seed][group_idx][metric_idx] for seed in test_metrics]
            mean_value = sum(values) / len(values)
            std_value = (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5
            
            # Coefficient of variation (CV) = std / mean
            # Avoid division by zero
            cv = std_value / mean_value if mean_value > 0 else 0.0
            
            weight = group_weights[group]
            total_cv += cv * weight
            total_weight += weight
    
    comprehensive_score = total_cv / total_weight if total_weight > 0 else 0.0
    return comprehensive_score


def main():
    runs_path = 'runs/stable/rand_train_dataset'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')

    base_seed = 42
    deterministic = True
    device = get_device(allow_share=True)

    # Prepare attack data once with fixed seed
    attack_data = prepare_attack_data(
        seed=base_seed,
        deterministic=deterministic,
        device=device
    )

    test_metrics = {}
    for seed in range(base_seed, base_seed + 5):
        metrics = attack_with_controlled_train_negsampling(
            attack_data=attack_data,
            base_seed=seed,
            deterministic=deterministic,
            device=device
        )

        test_metrics[seed] = metrics

    logging.info('\n' + '='*80)
    logging.info('Summary: Randomness from TrainAttackDataset (Training Negative Sampling)')
    logging.info('='*80)
    
    for group_idx, group in enumerate(['All', 'Head', 'Tail']):
        for metric_idx, name in enumerate(['AUROC', 'F1 Score', 'TPR@FPR=0.01', 'Precision', 'Recall']):
            values = [test_metrics[seed][group_idx][metric_idx] for seed in test_metrics]
            mean_value = sum(values) / len(values)
            std_value = (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5
            cv = std_value / mean_value if mean_value > 0 else 0.0
            logging.info(f'Test {group} {name}: {mean_value:.4f} ± {std_value:.4f} (CV: {cv:.4f})')
    
    # Compute and log comprehensive randomness score
    randomness_score = compute_comprehensive_randomness_score(test_metrics)
    logging.info('\n' + '='*80)
    logging.info(f'Comprehensive Randomness Score: {randomness_score:.6f}')
    logging.info('(Lower score indicates more stable results)')
    logging.info('='*80)


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

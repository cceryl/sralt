# This file compares the performance of SA model (RPQ) with baseline (SMIA) and random codebook ablation.

import torch.optim as optim
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG

from scripts.random.random_compare import test_config

configs = {
    'baseline': {
        'name': 'SMIA',
        'model': {
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-2
        },
        'train': {
            'train_batch_size': 8192,
            'val_batch_size': 2048,
            'val_num_negatives': 999,
            'num_epochs': 100,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': 1e-2},
            'metrics': [LinkPredRecall(k=10), LinkPredNDCG(k=10)],
            'metric_aggregator': 'lambda loss, metrics: sum(metrics)',
            'early_stopping_kwargs': {'patience': 10, 'greater_is_better': True}
        }
    },
    'semantic': {
        'name': 'SemanticAugmentationRPQ',
        'model': {
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-3,
            'rpq_num_quantizers': 4,
            'rpq_codebook_size': 64,
            'ablation_code': 'base',
            'ablation_feature': 'base'
        },
        'train': {
            'train_batch_size': 8192,
            'val_batch_size': 2048,
            'val_num_negatives': 999,
            'num_epochs': 300,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': 1e-2},
            'metrics': [LinkPredRecall(k=10), LinkPredNDCG(k=10)],
            'metric_aggregator': 'lambda loss, metrics: sum(metrics)',
            'early_stopping_kwargs': {'patience': 10, 'greater_is_better': True}
        }
    },
    'random': {
        'name': 'SemanticAugmentationRPQ',
        'model': {
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-3,
            'rpq_num_quantizers': 4,
            'rpq_codebook_size': 64,
            'ablation_code': 'random',
            'ablation_feature': 'base'
        },
        'train': {
            'train_batch_size': 8192,
            'val_batch_size': 2048,
            'val_num_negatives': 999,
            'num_epochs': 300,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': 1e-2},
            'metrics': [LinkPredRecall(k=10), LinkPredNDCG(k=10)],
            'metric_aggregator': 'lambda loss, metrics: sum(metrics)',
            'early_stopping_kwargs': {'patience': 10, 'greater_is_better': True}
        }
    }
}

test_config(
    config=configs,
    save_path='runs/random/compare/rpq'
)

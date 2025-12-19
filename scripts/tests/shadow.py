# This file is an example script for training and evaluating a shadow model.

import os
import logging
import warnings

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG

from sralt.target.data import load_target_data
from sralt.target.model import load_target_model
from sralt.shadow.data import load_shadow_data
from sralt.shadow.model import load_shadow_model, test_shadow_model
from sralt.utils import setup_logging, get_device


def main():
    runs_path = 'runs/tests/shadow_model'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
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
        logging.info(f'Test {name}: {value:.4f}')

    logging.info(f'Model representation: {shadow_model}')
    torch.save(shadow_model.state_dict(), os.path.join(runs_path, 'model.pth'))
    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings(
        'ignore',
        message='Sparse CSR tensor support is in beta state.'
    )

    main()

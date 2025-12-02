# This file is an example script for training and evaluating a shadow model.

import os
import logging
import warnings

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG

from sralt.target.data import load_dataset
from sralt.target.model import create_target_model
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.shadow.model import create_shadow_model
from sralt.utils import setup_logging, get_device, set_seed, DataLoader


def main():
    runs_path = 'runs/tests/shadow_model_test'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

    set_seed(42)
    device = get_device()

    target_model_path = './runs/tests/target_model_test/model.pth'
    target_data = load_dataset(
        name='Ciao',
        directory='./data/ciao',
        config={
            'min_rating': 2,
            'min_social_connections': 1,
            'min_user_interactions': 5,
            'min_item_interactions': 0
        }
    ).to(device)
    target_model = create_target_model(
        name='DESIGN',
        num_users=target_data.num_users,
        num_items=target_data.num_items,
        config={
            'embed_dim': 64,
            'user_fusion_layers': [0, 1, 2],
            'item_fusion_layers': [0, 1, 2],
            'decay': 5e-3,
            'distill_weight': 0.1
        }
    ).to(device)
    target_model.load_state_dict(torch.load(target_model_path, map_location=device, weights_only=True))

    shadow_data = ShadowData(
        target_data=target_data,
        target_model=target_model,
        shadow_social_ratio=0.1,
        num_recommendations=30,
        inference_batch_size=2048
    ).to(device)

    train_dataset = TrainShadowDataset(shadow_data)
    val_dataset = EvalShadowDataset(shadow_data, split='val', num_negative_samples=999)
    test_dataset = EvalShadowDataset(shadow_data, split='test', num_negative_samples=999)

    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    shadow_model = create_shadow_model(
        name='SMIA',
        num_users=shadow_data.num_users,
        num_items=shadow_data.num_items,
        config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-3
        }
    ).to(device)

    test_metrics = shadow_model.train_model(
        data=shadow_data,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=100,
        optimizer_cls=optim.Adam,
        optimizer_kwargs={'lr': 1e-2},
        metrics=[LinkPredRecall(k=10), LinkPredNDCG(k=10)],
        metric_aggregator=lambda loss, metrics: sum(metrics),
        early_stopping_kwargs={'patience': 10, 'greater_is_better': True},
        writer=writer
    )

    for name, value in test_metrics.items():
        logging.info(f'Test {name}: {value:.4f}')

    torch.save(shadow_model.state_dict(), os.path.join(runs_path, 'model.pth'))
    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings(
        'ignore',
        message='Sparse CSR tensor support is in beta state.'
    )

    main()

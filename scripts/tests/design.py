# This file tests the DESIGN model on the Ciao dataset.

import os
import logging

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG

from sralt.target.model import create_target_model
from sralt.target.data import load_dataset, TrainTargetDataset, EvalTargetDataset
from sralt.utils import setup_logging, get_device, set_seed, DataLoader


def main():
    runs_path = 'runs/design_ciao_test'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

    set_seed(42)
    device = get_device()

    data = load_dataset(
        name='Ciao',
        directory='./data/ciao',
        config={
            'min_rating': 2,
            'min_social_connections': 1,
            'min_user_interactions': 5,
            'min_item_interactions': 0
        }
    ).to(device)

    train_dataset = TrainTargetDataset(data)
    val_dataset = EvalTargetDataset(data, split='val', num_negative_samples=999)
    test_dataset = EvalTargetDataset(data, split='test', num_negative_samples=999)

    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    model = create_target_model(
        name='DESIGN',
        num_users=data.num_users,
        num_items=data.num_items,
        config={
            'embed_dim': 64,
            'user_fusion_layers': [0, 1, 2],
            'item_fusion_layers': [0, 1, 2],
            'decay': 5e-3,
            'distill_weight': 0.1
        }
    ).to(device)

    test_metrics = model.train_model(
        data=data,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=200,
        optimizer_cls=optim.Adam,
        optimizer_kwargs={'lr': 1e-2},
        metrics=[LinkPredRecall(k=10), LinkPredNDCG(k=10)],
        metric_aggregator=lambda loss, metrics: sum(metrics),
        early_stopping_kwargs={'patience': 20, 'greater_is_better': True},
        writer=writer
    )

    for name, value in test_metrics.items():
        logging.info(f'Test {name}: {value:.4f}')

    torch.save(model.state_dict(), os.path.join(runs_path, 'model.pth'))
    writer.close()


if __name__ == '__main__':
    main()

# This file is an example script for training and evaluating a membership inference attack model.

import os
import logging
import warnings

import torch
from torchmetrics import Metric, AUROC, F1Score, Precision, Recall
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected

from sralt.target.data import load_dataset
from sralt.target.model import create_target_model
from sralt.shadow.data import ShadowData
from sralt.shadow.model import create_shadow_model
from sralt.attack.data import AttackData, TrainAttackDataset, EvalAttackDataset
from sralt.attack.model import create_attack_model
from sralt.utils import setup_logging, get_device, set_seed, DataLoader, TPRatFPR, AdamWithGradientClipping


def main():
    runs_path = 'runs/tests/attack_model_test'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

    set_seed(42)
    device = get_device()

    target_model_path = './runs/tests/target_model_test/model.pth'
    shadow_model_path = './runs/tests/shadow_model_test/model.pth'

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
    shadow_model.load_state_dict(torch.load(shadow_model_path, map_location=device, weights_only=True))

    assert shadow_data.target_data_repr == repr(target_data)
    assert shadow_data.target_model_repr == repr(target_model)

    attack_data = AttackData(
        target_data=target_data,
        shadow_data=shadow_data,
        val_ratio=0.1
    ).to(device)
    user_features = shadow_model.get_user_features(
        social_edge_index=shadow_data.social_edge_index,
        behavioral_edge_index=to_undirected(shadow_data.behavioral_edge_index)
    ).detach()

    train_dataset = TrainAttackDataset(attack_data)
    val_dataset = EvalAttackDataset(attack_data, split='val')
    test_dataset = EvalAttackDataset(attack_data, split='test')

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    attack_model = create_attack_model(
        name='SMIA',
        user_feature_dim=user_features.size(1),
        config={
            'dropout': 0.5,
            'leaky_relu_slope': 0.2
        }
    ).to(device)

    metrics: list[Metric] = [
        AUROC(task='binary'),
        TPRatFPR(target_fpr=0.01),
        F1Score(task='binary'),
        Precision(task='binary'),
        Recall(task='binary')
    ]
    metrics_weights = [1.0, 1.0, 1.0, 0.0, 0.0]

    test_metrics = attack_model.train_model(
        user_features=user_features,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=50,
        loss_fn=torch.nn.BCEWithLogitsLoss().to(device),
        optimizer_cls=AdamWithGradientClipping,
        optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-4, 'grad_clip_norm': 5.0},
        metrics=metrics,
        metric_aggregator=lambda loss, metrics: sum(w * m for w, m in zip(metrics_weights, metrics)),
        early_stopping_kwargs={'patience': 10, 'greater_is_better': True},
        writer=writer
    )

    for name, value in test_metrics.items():
        logging.info(f'Test {name}: {value:.4f}')

    torch.save(attack_model.state_dict(), os.path.join(runs_path, 'model.pth'))
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

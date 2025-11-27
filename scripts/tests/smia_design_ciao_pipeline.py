# This file tests the complete pipeline of SMIA on the Ciao dataset and the DESIGN target model.

import os
import logging
import warnings

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torchmetrics import Metric, AUROC, F1Score, Precision, Recall

from sralt.target.model import create_target_model
from sralt.target.data import load_dataset, TrainTargetDataset, EvalTargetDataset
from sralt.shadow.model import create_shadow_model
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.attack.model import create_attack_model
from sralt.attack.data import AttackData, TrainAttackDataset, EvalAttackDataset
from sralt.utils import setup_logging, timer, get_device, set_seed, DataLoader


class TPRatFPR(Metric):
    """
    True Positive Rate (TPR) at a specified False Positive Rate (FPR).
    """
    full_state_update = True

    def __init__(self, target_fpr: float):
        """
        Initialize the TPRatFPR metric.

        :param target_fpr: The target false positive rate (between 0 and 1).
        """
        super().__init__(dist_sync_on_step=False)
        assert 0 <= target_fpr <= 1

        self.target_fpr = target_fpr
        self.preds: list[torch.Tensor]
        self.target: list[torch.Tensor]
        self.add_state('preds', default=[], dist_reduce_fx='cat')
        self.add_state('target', default=[], dist_reduce_fx='cat')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric state with new predictions and targets.

        :param preds: Predicted probabilities of shape (batch_size,).
        :param target: Ground truth binary labels of shape (batch_size,).
        """
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        """
        Compute the TPR at the specified FPR.

        :return: The TPR at the specified FPR.
        """
        # Sort predictions and targets by predicted scores in descending order
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)

        sorted_idx = torch.argsort(preds, descending=True)
        preds = preds[sorted_idx]
        target = target[sorted_idx]

        # Calculate TPR and FPR
        pos = torch.sum(target == 1).item()
        neg = torch.sum(target == 0).item()

        if pos == 0 or neg == 0:
            return torch.tensor(float('nan'), device=preds.device)

        tp = torch.cumsum(target == 1, dim=0)
        fp = torch.cumsum(target == 0, dim=0)

        tpr = tp / pos
        fpr = fp / neg

        # Corner cases
        if fpr[-1] < self.target_fpr:
            return tpr[-1]

        if fpr[0] >= self.target_fpr:
            return tpr[0]

        # Linear interpolation to find TPR at target FPR
        idx = torch.searchsorted(fpr, torch.tensor(self.target_fpr, device=preds.device), right=False)

        fpr0, fpr1 = fpr[idx-1], fpr[idx]
        tpr0, tpr1 = tpr[idx-1], tpr[idx]

        if fpr1 == fpr0:
            return tpr1

        alpha = (self.target_fpr - fpr0) / (fpr1 - fpr0)
        result = tpr0 + alpha * (tpr1 - tpr0)
        return result


class AdamWithGradientClipping(optim.Adam):
    """
    Adam optimizer with gradient clipping.
    """

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.0, grad_clip_norm: float = 1.0):
        """
        Initialize the AdamWithGradientClipping optimizer.

        :param params: Parameters to optimize.
        :param lr: Learning rate.
        :param weight_decay: Weight decay (L2 regularization).
        :param grad_clip_norm: Maximum norm for gradient clipping.
        """
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.grad_clip_norm = grad_clip_norm

    def step(self, closure=None):
        """
        Perform a single optimization step with gradient clipping.
        """
        torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], self.grad_clip_norm)
        super().step(closure)


@timer()
def main():
    runs_path = 'runs/smia_design_ciao_pipeline_test'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

    seed = 42
    device = get_device()

    # Target model stage
    logging.info('Starting target model training...')
    set_seed(seed)

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

    train_dataset = TrainTargetDataset(target_data)
    val_dataset = EvalTargetDataset(target_data, split='val', num_negative_samples=999)
    test_dataset = EvalTargetDataset(target_data, split='test', num_negative_samples=999)

    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

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

    target_test_metrics = target_model.train_model(
        data=target_data,
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

    for name, value in target_test_metrics.items():
        logging.info(f'Target model test {name}: {value:.4f}')

    torch.save(target_model.state_dict(), os.path.join(runs_path, 'target_model.pth'))
    logging.info('Target model training completed.')

    # Shadow model stage
    logging.info('Starting shadow model training...')
    set_seed(seed)

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

    shadow_test_metrics = shadow_model.train_model(
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

    for name, value in shadow_test_metrics.items():
        logging.info(f'Shadow model test {name}: {value:.4f}')

    torch.save(shadow_model.state_dict(), os.path.join(runs_path, 'shadow_model.pth'))
    logging.info('Shadow model training completed.')

    # Attack model stage
    logging.info('Starting attack model training...')
    set_seed(seed)

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

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
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

    attack_metrics: list[Metric] = [
        AUROC(task='binary'),
        TPRatFPR(target_fpr=0.01),
        F1Score(task='binary'),
        Precision(task='binary'),
        Recall(task='binary')
    ]
    metrics_weights = [1.0, 1.0, 1.0, 0.0, 0.0]

    attack_test_metrics = attack_model.train_model(
        user_features=user_features,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=50,
        loss_fn=torch.nn.BCEWithLogitsLoss().to(device),
        optimizer_cls=AdamWithGradientClipping,
        optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-4, 'grad_clip_norm': 5.0},
        metrics=attack_metrics,
        metric_aggregator=lambda loss, metrics: sum(w * m for w, m in zip(metrics_weights, metrics)),
        early_stopping_kwargs={'patience': 5, 'greater_is_better': True},
        writer=writer
    )

    for name, value in attack_test_metrics.items():
        logging.info(f'Attack model test {name}: {value:.4f}')

    torch.save(attack_model.state_dict(), os.path.join(runs_path, 'attack_model.pth'))
    logging.info('Attack model training completed.')
    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings(
        'ignore',
        message=''
        'You are trying to use a metric in deterministic mode on GPU that uses `torch.cumsum`, which is currently not supported. '
        'The tensor will be copied to the CPU memory to compute it and then copied back to GPU. Expect some slowdowns.',
        module="torchmetrics.utilities.prints"
    )

    main()

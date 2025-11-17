# This file tests the SMIA attacker on the pre-trained SMIA shadow model.

import os
import typing
import logging
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Metric, AUROC, F1Score, Precision, Recall
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected

from sralt.target.data import load_dataset
from sralt.target.model import create_target_model
from sralt.shadow.data import ShadowData
from sralt.shadow.model import create_shadow_model
from sralt.attack.data import AttackData, TrainAttackDataset, EvalAttackDataset
from sralt.attack.model import create_attack_model, AttackModel
from sralt.utils import setup_logging, timer, get_device, set_seed, DataLoader, EarlyStopping


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


@torch.no_grad()
def compute_binary_metrics(model: AttackModel, user_features: torch.Tensor, eval_loader: DataLoader, metrics: list[Metric]) -> list[float]:
    """
    Compute binary classification metrics.

    :param model: The attack model.
    :param user_features: The user features tensor of shape (num_users, user_feature_dim).
    :param eval_loader: DataLoader for evaluation data.
    :param metrics: A list of metric calculators.
    :return: A list of computed metric values.
    """
    model.eval()

    metrics = [metric.to(user_features.device) for metric in metrics]
    for metric in metrics:
        metric.reset()

    for user1, user2, labels in eval_loader:
        probs = model.predict(torch.stack([user1, user2]), user_features)

        for metric in metrics:
            metric.update(probs, labels)

    return [typing.cast(torch.Tensor, metric.compute()).item() for metric in metrics]


@timer()
def train(
    model: AttackModel, optimizer: optim.Optimizer, loss_fn: nn.Module,
    user_features: torch.Tensor, train_loader: DataLoader, eval_loader: DataLoader,
    num_epochs: int, early_stopping: EarlyStopping,
    metrics: list[Metric], metrics_weights: list[float],
    writer: SummaryWriter | None = None
):
    """
    Train the attack model.

    :param model: The attack model.
    :param optimizer: The optimizer for training.
    :param loss_fn: The loss function.
    :param user_features: The user features tensor of shape (num_users, user_feature_dim).
    :param train_loader: DataLoader for training data.
    :param eval_loader: DataLoader for evaluation data.
    :param num_epochs: The number of training epochs.
    :param early_stopping: Early stopping mechanism.
    :param metrics: A list of metric calculators.
    :param metrics_weights: A list of weights for each metric to compute a weighted sum for early stopping.
    :param writer: Optional TensorBoard SummaryWriter for logging.
    """
    metrics_names = [metric.__class__.__name__ for metric in metrics]

    for epoch in range(1, num_epochs + 1):
        model.train()

        # Training steps
        total_loss = 0.0
        total_samples = 0

        for user1, user2, labels in train_loader:
            optimizer.zero_grad()

            logits = model(torch.stack([user1, user2]), user_features)

            labels: torch.Tensor
            loss: torch.Tensor = loss_fn(logits, labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        if writer is not None:
            writer.add_scalar('Loss/train', average_loss, epoch)

        # Evaluation
        metrics_results = compute_binary_metrics(model, user_features, eval_loader, metrics)
        metrics_str = ', '.join([f'{name}={value:.4f}' for name, value in zip(metrics_names, metrics_results)])
        logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metrics_str}')

        if writer is not None:
            for name, value in zip(metrics_names, metrics_results):
                writer.add_scalar(f'{name}/val', value, epoch)

        weighted_metric = sum([weight * value for weight, value in zip(metrics_weights, metrics_results)])
        if early_stopping.step(weighted_metric):
            break

    early_stopping.load_best_state()


def main():
    runs_path = 'runs/smia_attack_on_design_ciao_test'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

    set_seed(42)
    device = get_device()

    target_model_path = './runs/design_ciao_test/model.pth'
    shadow_model_path = './runs/shadow_smia_on_design_ciao_test/model.pth'

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

    optimizer = AdamWithGradientClipping(attack_model.parameters(), lr=5e-3, weight_decay=1e-2, grad_clip_norm=5.0)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    early_stopping = EarlyStopping(shadow_model, patience=10, greater_is_better=True)
    metrics: list[Metric] = [
        AUROC(task='binary'),
        F1Score(task='binary'),
        Precision(task='binary'),
        Recall(task='binary')
    ]
    metrics_weights = [1.0, 1.0, 0.0, 0.0]

    train(
        model=attack_model, optimizer=optimizer, loss_fn=loss_fn,
        user_features=user_features, train_loader=train_loader, eval_loader=val_loader,
        num_epochs=100, early_stopping=early_stopping,
        metrics=metrics, metrics_weights=metrics_weights, writer=writer
    )

    final_metrics = compute_binary_metrics(attack_model, user_features, test_loader, metrics)
    for metric, value in zip(metrics, final_metrics):
        logging.info(f'Final {metric.__class__.__name__}: {value:.4f}')

    torch.save(shadow_model.state_dict(), os.path.join(runs_path, 'model.pth'))
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

# This file tests the complete pipeline of SMIA on the Ciao dataset and the DESIGN target model.

import os
import typing
import logging
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torch_geometric.metrics.link_pred import LinkPredMetric
from torchmetrics import Metric, AUROC, F1Score, Precision, Recall

from sralt.target.model import create_target_model, TargetModel
from sralt.target.data import load_dataset, TargetData, TrainTargetDataset, EvalTargetDataset
from sralt.shadow.model import create_shadow_model, ShadowModel
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.attack.model import create_attack_model, AttackModel
from sralt.attack.data import AttackData, TrainAttackDataset, EvalAttackDataset
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
def compute_metrics_at_k(
    model: TargetModel | ShadowModel, data: TargetData | ShadowData, metrics: list[LinkPredMetric], eval_loader: DataLoader
) -> list[float]:
    """
    Compute multiple metrics (e.g., Recall, NDCG) at given k(s).

    :param model: The target or shadow model.
    :param data: The data on which to compute the metrics.
    :param metrics: A list of metric calculators.
    :param eval_loader: DataLoader for evaluation data.
    :return: The computed metrics as a list of floats.
    """
    model.eval()

    split = getattr(eval_loader.dataset, 'split')
    if split == 'val':
        behavioral_edge_index = data.behavioral_train_edge_index
    elif split == 'test':
        behavioral_edge_index = torch.cat([data.behavioral_train_edge_index, data.behavioral_val_edge_index], dim=1)
    else:
        raise ValueError(f'Unknown split: {split}')

    # Extract user and item embeddings
    embeds = model.get_embedding(data.social_edge_index, to_undirected(behavioral_edge_index))
    user_embeds, item_embeds = torch.split(embeds, [data.num_users, data.num_items], dim=0)

    # Prepare metrics calculators
    metrics = [metric.to(data.device) for metric in metrics]
    max_k = max([metric.k for metric in metrics])
    for metric in metrics:
        metric.reset()

    for users, pos_items, neg_items in eval_loader:
        users: torch.Tensor
        pos_items: torch.Tensor
        neg_items: torch.Tensor

        # Calculate scores
        user_emb_batch = user_embeds[users - data.user_range.start]
        candidate_items = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        scores = user_emb_batch @ item_embeds.t()

        # Mask out items not in the candidate set
        users_range = torch.arange(users.size(0), device=data.device)
        candidate_mask = torch.zeros_like(scores, dtype=torch.bool)
        candidate_mask[users_range.unsqueeze(1), candidate_items - data.item_range.start] = True
        scores[~candidate_mask] = float('-inf')

        # Get top-k item indices for each user
        _, top_items = torch.topk(scores, k=max_k, sorted=True)

        # Build edge_label_index
        edge_label_index = torch.stack([users_range, pos_items - data.item_range.start])

        # Update all calculators
        for metric in metrics:
            metric.update(top_items[:, :metric.k], edge_label_index)

    return [metric.compute().item() for metric in metrics]


@torch.no_grad()
def compute_attack_metrics(model: AttackModel, user_features: torch.Tensor, eval_loader: DataLoader, metrics: list[Metric]) -> list[float]:
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
def train_recommendation_model(
    model: TargetModel | ShadowModel, data: TargetData | ShadowData,
    train_loader: DataLoader, eval_loader: DataLoader,
    optimizer: optim.Optimizer, num_epochs: int, early_stopping: EarlyStopping,
    metrics: list[LinkPredMetric], writer: SummaryWriter | None = None
):
    """
    Train the target or shadow model.

    :param model: The target or shadow model.
    :param data: The corresponding dataset.
    :param train_loader: DataLoader for training data.
    :param eval_loader: DataLoader for evaluation data.
    :param optimizer: The optimizer for training.
    :param num_epochs: The number of training epochs.
    :param early_stopping: Early stopping mechanism.
    :param metrics: A list of metric calculators.
    :param writer: Optional TensorBoard SummaryWriter for logging.
    """
    undirected_behavioral_train_edge_index = to_undirected(data.behavioral_train_edge_index)
    model_name = model.__class__.__name__
    metric_names = [
        typing.cast(str, metric.__class__.__name__).removeprefix('LinkPred') + f'@{metric.k}' for metric in metrics
    ]

    for epoch in range(1, num_epochs + 1):
        model.train()

        # Training steps
        total_loss = 0.0
        total_samples = 0

        for users, pos_items, neg_items in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
            optimizer.zero_grad()

            pos_edges = torch.stack([users, pos_items], dim=0)
            neg_edges = torch.stack([users, neg_items], dim=0)
            node_ids = torch.cat([users, pos_items, neg_items])

            pos_scores: torch.Tensor = model(data.social_edge_index, undirected_behavioral_train_edge_index, pos_edges)
            neg_scores: torch.Tensor = model(data.social_edge_index, undirected_behavioral_train_edge_index, neg_edges)

            loss = model.calculate_loss(pos_scores, neg_scores, node_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * users.size(0)
            total_samples += users.size(0)

        loss = total_loss / total_samples
        if writer is not None:
            writer.add_scalar(f'{model_name}/Loss/train', loss, epoch)

        # Evaluation
        metrics_results = compute_metrics_at_k(model, data, metrics, eval_loader)
        metrics_str = ', '.join([f'{name}={value:.4f}' for name, value in zip(metric_names, metrics_results)])
        logging.info(f'Evaluation at epoch {epoch:04d}: loss={loss:.4f}, {metrics_str}')

        if writer is not None:
            for name, value in zip(metric_names, metrics_results):
                writer.add_scalar(f'{model_name}/{name}/val', value, epoch)

        if early_stopping.step(sum(metrics_results)):
            break

    early_stopping.load_best_state()


@timer()
def train_attack_model(
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
    model_name = model.__class__.__name__
    metrics_names = [metric.__class__.__name__.removeprefix('Binary') for metric in metrics]

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
            writer.add_scalar(f'{model_name}/Loss/train', average_loss, epoch)

        # Evaluation
        metrics_results = compute_attack_metrics(model, user_features, eval_loader, metrics)
        metrics_str = ', '.join([f'{name}={value:.4f}' for name, value in zip(metrics_names, metrics_results)])
        logging.info(f'Evaluation at epoch {epoch:04d}: loss={average_loss:.4f}, {metrics_str}')

        if writer is not None:
            for name, value in zip(metrics_names, metrics_results):
                writer.add_scalar(f'{model_name}/{name}/val', value, epoch)

        weighted_metric = sum([weight * value for weight, value in zip(metrics_weights, metrics_results)])
        if early_stopping.step(weighted_metric):
            break

    early_stopping.load_best_state()


@timer()
def main():
    runs_path = 'runs/smia_design_ciao_pipeline_test'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

    seed = 42
    device = get_device(allow_share=True)

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
    optimizer = optim.Adam(target_model.parameters(), lr=1e-2)
    early_stopping = EarlyStopping(target_model, patience=20, greater_is_better=True)
    target_metrics = [LinkPredRecall(k=10), LinkPredNDCG(k=10)]

    train_recommendation_model(
        model=target_model, data=target_data, train_loader=train_loader, eval_loader=val_loader,
        optimizer=optimizer, num_epochs=200, early_stopping=early_stopping,
        metrics=target_metrics, writer=writer
    )

    final_metrics = compute_metrics_at_k(target_model, target_data, target_metrics, test_loader)
    metrics_names = [
        typing.cast(str, metric.__class__.__name__).removeprefix('LinkPred') + f'@{metric.k}' for metric in target_metrics
    ]
    for name, value in zip(metrics_names, final_metrics):
        logging.info(f'Target model final {name}: {value:.4f}')

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

    optimizer = optim.Adam(shadow_model.parameters(), lr=1e-2)
    early_stopping = EarlyStopping(shadow_model, patience=10, greater_is_better=True)
    shadow_metrics = [LinkPredRecall(k=10), LinkPredNDCG(k=10)]

    train_recommendation_model(
        model=shadow_model, data=shadow_data, train_loader=train_loader, eval_loader=val_loader,
        optimizer=optimizer, num_epochs=100, early_stopping=early_stopping,
        metrics=shadow_metrics, writer=writer
    )

    final_metrics = compute_metrics_at_k(shadow_model, shadow_data, shadow_metrics, test_loader)
    metrics_names = [
        typing.cast(str, metric.__class__.__name__).removeprefix('LinkPred') + f'@{metric.k}' for metric in shadow_metrics
    ]
    for name, value in zip(metrics_names, final_metrics):
        logging.info(f'Shadow model final {name}: {value:.4f}')

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
    early_stopping = EarlyStopping(attack_model, patience=5, greater_is_better=True)
    attack_metrics: list[Metric] = [
        AUROC(task='binary'),
        F1Score(task='binary'),
        Precision(task='binary'),
        Recall(task='binary')
    ]
    metrics_weights = [1.0, 1.0, 0.0, 0.0]

    train_attack_model(
        model=attack_model, optimizer=optimizer, loss_fn=loss_fn,
        user_features=user_features, train_loader=train_loader, eval_loader=val_loader,
        num_epochs=50, early_stopping=early_stopping,
        metrics=attack_metrics, metrics_weights=metrics_weights,
        writer=writer
    )

    final_metrics = compute_attack_metrics(attack_model, user_features, test_loader, attack_metrics)
    for metric, value in zip(attack_metrics, final_metrics):
        logging.info(f'Final {metric.__class__.__name__.removeprefix('Binary')}: {value:.4f}')

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

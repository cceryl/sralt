# This file tests the SMIA shadow model on the pre-trained target DESIGN model.

import os
import typing
import logging

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torch_geometric.metrics.link_pred import LinkPredMetric

from sralt.target.data import load_dataset
from sralt.target.model import create_target_model
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.shadow.model import create_shadow_model, ShadowModel
from sralt.utils import setup_logging, timer, get_device, set_seed, DataLoader, EarlyStopping


@torch.no_grad()
def compute_metrics_at_k(model: ShadowModel, data: ShadowData, metrics: list[LinkPredMetric], eval_loader: DataLoader) -> list[float]:
    """
    Compute multiple metrics (e.g., Recall, NDCG) at given k(s).

    :param model: The shadow model.
    :param data: The data on which to compute the metrics.
    :param metrics: A list of metric calculators.
    :param eval_loader: DataLoader for evaluation data.
    :return: The computed metrics as a list of floats.
    """
    model.eval()

    split = typing.cast(EvalShadowDataset, eval_loader.dataset).split
    behavioral_edge_index = [data.behavioral_train_edge_index]
    if split == 'test':
        behavioral_edge_index.append(data.behavioral_val_edge_index)
    behavioral_edge_index = torch.cat(behavioral_edge_index, dim=1)

    # Extract user and item embeddings
    embeds = model.get_embedding(data.social_edge_index, to_undirected(behavioral_edge_index))
    user_embeds, item_embeds = torch.split(embeds, [data.num_users, data.num_items], dim=0)

    # Prepare metrics calculators
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


@timer()
def train(
    model: ShadowModel, data: ShadowData, train_loader: DataLoader, eval_loader: DataLoader,
    optimizer: optim.Optimizer, num_epochs: int, early_stopping: EarlyStopping,
    metrics: list[LinkPredMetric], writer: SummaryWriter | None = None
):
    """
    Train the shadow model.

    :param model: The shadow model.
    :param data: The shadow dataset.
    :param train_loader: DataLoader for training data.
    :param eval_loader: DataLoader for evaluation data.
    :param optimizer: The optimizer for training.
    :param num_epochs: The number of training epochs.
    :param early_stopping: Early stopping mechanism.
    :param metrics: A list of metric calculators.
    :param writer: Optional TensorBoard SummaryWriter for logging.
    """
    undirected_behavioral_train_edge_index = to_undirected(data.behavioral_train_edge_index)
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

            users: torch.Tensor
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
            writer.add_scalar('Loss/train', loss, epoch)

        # Evaluation
        metrics_results = compute_metrics_at_k(model, data, metrics, eval_loader)
        metrics_str = ', '.join([f'{name}={value:.4f}' for name, value in zip(metric_names, metrics_results)])
        logging.info(f'Evaluation at epoch {epoch:04d}: loss={loss:.4f}, {metrics_str}')

        if writer is not None:
            for name, value in zip(metric_names, metrics_results):
                writer.add_scalar(f'{name}/val', value, epoch)

        if early_stopping.step(sum(metrics_results)):
            break

    early_stopping.load_best_state()


def main():
    runs_path = 'runs/shadow_smia_on_design_ciao_test'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

    set_seed(42)
    device = get_device()

    target_model_path = './runs/design_ciao_test/model.pth'
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

    optimizer = optim.Adam(shadow_model.parameters(), lr=1e-2)
    early_stopping = EarlyStopping(shadow_model, patience=10, greater_is_better=True)
    metrics = [LinkPredRecall(k=10), LinkPredNDCG(k=10)]

    train(
        model=shadow_model, data=shadow_data, train_loader=train_loader, eval_loader=val_loader,
        optimizer=optimizer, num_epochs=100, early_stopping=early_stopping,
        metrics=metrics, writer=writer
    )

    final_metrics = compute_metrics_at_k(shadow_model, shadow_data, metrics, test_loader)
    metrics_names = [
        typing.cast(str, metric.__class__.__name__).removeprefix('LinkPred') + f'@{metric.k}' for metric in metrics
    ]
    for name, value in zip(metrics_names, final_metrics):
        logging.info(f'Final {name}: {value:.4f}')

    torch.save(shadow_model.state_dict(), os.path.join(runs_path, 'model.pth'))
    writer.close()


if __name__ == '__main__':
    main()

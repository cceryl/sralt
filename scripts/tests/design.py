# This file tests the DESIGN model on the Ciao dataset.

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

from sralt.target.model import create_target_model, TargetModel
from sralt.target.data import load_dataset, TargetData, TrainTargetDataset, EvalTargetDataset
from sralt.utils import setup_logging, timer, get_device, set_seed, DataLoader, EarlyStopping


@torch.no_grad()
def compute_metrics_at_k(model: TargetModel, data: TargetData, metrics: list[LinkPredMetric], eval_loader: DataLoader) -> list[float]:
    """
    Compute multiple metrics (e.g., Recall, NDCG) at given k(s).

    :param model: The DESIGN model.
    :param data: The data on which to compute the metrics.
    :param metrics: A list of metric calculators.
    :param eval_loader: DataLoader for evaluation data.
    :return: The computed metrics as a list of floats.
    """
    model.eval()

    split = typing.cast(EvalTargetDataset, eval_loader.dataset).split
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
        assert max_k <= neg_items.size(1) + 1, 'Number of candidate items must be at least k.'

        # Adjust indices to 0-based
        users: torch.Tensor = users - data.user_range.start
        pos_items: torch.Tensor = pos_items - data.item_range.start
        neg_items: torch.Tensor = neg_items - data.item_range.start

        # Calculate scores for candidate items
        candidate_items = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        user_embeds_batch = user_embeds[users].unsqueeze(1)
        item_embeds_batch = item_embeds[candidate_items]
        scores = (user_embeds_batch * item_embeds_batch).sum(dim=-1)

        # Get top-k item indices among candidate items for each user
        _, topk_local = torch.topk(scores, k=max_k, sorted=True)
        topk_global = torch.gather(candidate_items, 1, topk_local)

        # Update all calculators
        users_range = torch.arange(users.size(0), device=data.device)
        edge_label_index = torch.stack([users_range, pos_items])

        for metric in metrics:
            metric.update(topk_global[:, :metric.k], edge_label_index)

    return [metric.compute().item() for metric in metrics]


@timer()
def train(
    model: TargetModel, data: TargetData, train_loader: DataLoader, eval_loader: DataLoader,
    optimizer: optim.Optimizer, num_epochs: int, early_stopping: EarlyStopping,
    metrics: list[LinkPredMetric], writer: SummaryWriter | None = None
):
    """
    Train the DESIGN model.

    :param model: The DESIGN model.
    :param data: The dataset.
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
    runs_path = 'runs/design_ciao_test'
    os.makedirs(runs_path, exist_ok=True)
    setup_logging(log_file=os.path.join(runs_path, 'train.log'), tz='Asia/Shanghai')
    writer = SummaryWriter(log_dir=runs_path)

    set_seed(42)
    device = get_device(allow_share=True)

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
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    early_stopping = EarlyStopping(model, patience=20, greater_is_better=True)
    metrics = [LinkPredRecall(k=10), LinkPredNDCG(k=10)]

    train(
        model=model, data=data, train_loader=train_loader, eval_loader=val_loader,
        optimizer=optimizer, num_epochs=200, early_stopping=early_stopping,
        metrics=metrics, writer=writer
    )

    final_metrics = compute_metrics_at_k(model, data, metrics, test_loader)
    metrics_names = [
        typing.cast(str, metric.__class__.__name__).removeprefix('LinkPred') + f'@{metric.k}' for metric in metrics
    ]
    for name, value in zip(metrics_names, final_metrics):
        logging.info(f'Final {name}: {value:.4f}')

    torch.save(model.state_dict(), os.path.join(runs_path, 'model.pth'))
    writer.close()


if __name__ == '__main__':
    main()

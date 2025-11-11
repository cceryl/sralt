# This file tests the DESIGN model on the Ciao dataset.

import os
import typing
import logging

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torch_geometric.metrics.link_pred import LinkPredMetric

from sralt.target.model import create_target_model, TargetModel
from sralt.target.data import load_dataset, TargetData, TrainTargetDataset, EvalTargetDataset
from sralt.utils import setup_logging, timer, get_device, set_seed, EarlyStopping


@torch.no_grad()
def compute_metrics_at_k(
    model: TargetModel, data: TargetData, metrics: dict[type[LinkPredMetric], list[int]], eval_loader: DataLoader
) -> dict[type[LinkPredMetric], list[float]]:
    """
    Compute multiple metrics (e.g., Recall, NDCG) at given k(s).

    :param model: The DESIGN model.
    :param data: The data on which to compute the metrics.
    :param metrics: A dictionary where keys are metric classes (subclasses of LinkPredMetric) and values are lists of k values.
    :param eval_loader: DataLoader for evaluation data.
    :return: The computed metrics as a dictionary where keys are metric classes and values are lists of metric values corresponding to the k values.
    """
    model.eval()

    split = typing.cast(EvalTargetDataset, eval_loader.dataset).split
    behavioral_edge_index = [data.behavioral_train_edge_index]
    if split == 'test':
        behavioral_edge_index.append(data.behavioral_val_edge_index)
    behavioral_edge_index = torch.cat(behavioral_edge_index, dim=1)

    # Extract user and item embeddings
    embeds = model.get_embedding(data.social_edge_index, to_undirected(behavioral_edge_index))
    user_embeds, item_embeds = torch.split(embeds, [data.num_users, data.num_items], dim=0)

    # Prepare metrics calculators
    max_k = max([max(ks) for ks in metrics.values()])
    calculators: dict[type[LinkPredMetric], list[LinkPredMetric]] = {
        metric_cls: [metric_cls(k).to(data.device) for k in ks] for metric_cls, ks in metrics.items()
    }

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
        for calculator_list in calculators.values():
            for calculator in calculator_list:
                calculator.update(top_items[:, :calculator.k], edge_label_index)

    return {
        metric_cls: [calculator.compute().item() for calculator in calculator_list]
        for metric_cls, calculator_list in calculators.items()
    }


@timer()
def train(
    model: TargetModel, data: TargetData, train_loader: DataLoader, eval_loader: DataLoader,
    optimizer: optim.Optimizer, num_epochs: int, early_stopping: EarlyStopping,
    metrics: dict[type[LinkPredMetric], list[int]], writer: SummaryWriter | None = None,
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
    :param metrics: A dictionary where keys are metric classes (subclasses of LinkPredMetric) and values are lists of k values.
    :param writer: Optional TensorBoard SummaryWriter for logging.
    """
    undirected_behavioral_train_edge_index = to_undirected(data.behavioral_train_edge_index)
    metric_names = {
        metric_cls: [f'{metric_cls.__name__.replace('LinkPred', '')}@{k}' for k in ks]
        for metric_cls, ks in metrics.items()
    }

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
        metrics_results = {
            name: value
            for metric_cls, values in compute_metrics_at_k(model, data, metrics, eval_loader).items()
            for name, value in zip(metric_names[metric_cls], values)
        }

        metrics_str = ', '.join([f'{name}={value:.4f}' for name, value in metrics_results.items()])
        logging.info(f'Evaluation at epoch {epoch:04d}: loss={loss:.4f}, {metrics_str}')

        if writer is not None:
            for name, value in metrics_results.items():
                writer.add_scalar(f'{name}/val', value, epoch)

        if early_stopping.step(sum(metrics_results.values())):
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
    early_stopping = EarlyStopping(model, patience=30, greater_is_better=True)
    metrics = {LinkPredRecall: [10], LinkPredNDCG: [10]}

    train(
        model=model, data=data, train_loader=train_loader, eval_loader=val_loader,
        optimizer=optimizer, num_epochs=300, early_stopping=early_stopping,
        metrics=metrics, writer=writer
    )

    final_metrics = compute_metrics_at_k(model, data, metrics, test_loader)
    for metric_cls, values in final_metrics.items():
        for k, value in zip(metrics[metric_cls], values):
            logging.info(f'Final {metric_cls.__name__.replace('LinkPred', '')}@{k}: {value:.4f}')

    torch.save(model.state_dict(), os.path.join(runs_path, 'model.pth'))
    writer.close()


if __name__ == '__main__':
    main()

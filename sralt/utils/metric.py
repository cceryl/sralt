import typing
from typing import Callable

import torch
from torchmetrics import Metric
from torch_geometric.utils import coalesce
from torch_geometric.metrics.link_pred import LinkPredMetric

from .data import DataLoader


@torch.no_grad()
def recommend_items(
    embeds: torch.Tensor,
    num_users: int,
    num_recommendations: int,
    mask_edge_index: torch.Tensor | None,
    batch_size: int = 1024
) -> torch.Tensor:
    """
    Generate recommendations for the users based on dot product scores.
    All indices must be global, with users in [0, num_users) and items in [num_users, num_users + num_items).

    :param embeds: Combined user and item embeddings tensor of shape (num_users + num_items, embed_dim).
    :param num_users: Number of users.
    :param num_recommendations: The number of recommendations to generate per user.
    :param mask_edge_index: Edge index tensor of shape (2, num_edges) representing user-item interactions to be masked (excluded) from recommendations.
    :param batch_size: Batch size for processing users.
    :return: Edge index tensor of shape (2, num_users * num_recommendations) containing recommendations.
    """
    item_embeds = embeds[num_users:]
    device = embeds.device

    mask_users = torch.tensor([], dtype=torch.long, device=device)
    mask_items = torch.tensor([], dtype=torch.long, device=device)
    if mask_edge_index is not None:
        mask_users, mask_items = mask_edge_index
        mask_items = mask_items - num_users

    recommendations = []
    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)

        scores = embeds[start:end] @ item_embeds.t()
        batch_mask = (start <= mask_users) & (mask_users < end)
        scores[mask_users[batch_mask] - start, mask_items[batch_mask]] = float('-inf')
        _, top_items = torch.topk(scores, k=num_recommendations, sorted=True)

        top_items: torch.Tensor
        users = torch.arange(start, end, device=device).repeat_interleave(num_recommendations)
        items = top_items.flatten() + num_users
        recommendations.append(torch.stack([users, items], dim=0))

    return coalesce(torch.cat(recommendations, dim=1))


@torch.no_grad()
def compute_recommendation_metrics(
    embeds: torch.Tensor,
    metrics: list[LinkPredMetric],
    eval_loader: DataLoader
) -> list[float]:
    """
    Compute multiple recommendation metrics (e.g., Recall, NDCG) at given k(s) using dot product scores.
    All indices must be global, with users in [0, num_users) and items in [num_users, num_users + num_items).

    :param embeds: The combined user and item embeddings.
    :param metrics: A list of metric calculators.
    :param eval_loader: DataLoader for evaluation data yielding (users, pos_items, neg_items), 
                        where users & pos_items of shape (batch_size,), neg_items of shape (batch_size, num_neg).
    :return: The computed metrics as a list of floats.
    """
    device = embeds.device
    max_k = max([metric.k for metric in metrics])
    calculators = [typing.cast(Metric, metric).to(device) for metric in metrics]
    for calculator in calculators:
        calculator.reset()

    for users, pos_items, neg_items in eval_loader:
        if max_k > neg_items.size(1) + 1:
            raise ValueError('Number of candidate items must be at least k.')

        users: torch.Tensor
        pos_items: torch.Tensor
        neg_items: torch.Tensor

        # Score candidate items
        candidate_items = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        user_embeds_batch = embeds[users].unsqueeze(1)
        item_embeds_batch = embeds[candidate_items]
        scores = (user_embeds_batch * item_embeds_batch).sum(dim=-1)

        # Get top-k item indices among candidate items for each user
        _, topk_local = torch.topk(scores, k=max_k, sorted=True)
        topk_global = torch.gather(candidate_items, 1, topk_local)

        # Update all calculators
        users_range = torch.arange(users.size(0), device=device)
        edge_label_index = torch.stack([users_range, pos_items])

        for calculator in calculators:
            calculator.update(topk_global[:, :calculator.k], edge_label_index)

    return [typing.cast(torch.Tensor, calculator.compute()).item() for calculator in calculators]


@torch.no_grad()
def compute_attack_metrics(
    device: torch.device,
    predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metrics: list[Metric],
    eval_loader: DataLoader,
    head_indices: torch.Tensor,
    tail_indices: torch.Tensor
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute binary classification metrics for attack models.
    Note that for head/tail users, samples with 1 head/tail node are counted once, and samples with 2 head/tail nodes are counted twice.

    :param device: The device of the evaluation data loader.
    :param predictor: A callable that takes two tensors of user indices and returns predicted probabilities.
    :param metrics: A list of metric calculators.
    :param eval_loader: DataLoader for evaluation data yielding (user1, user2, labels).
    :param head_indices: Tensor of head user indices.
    :param tail_indices: Tensor of tail user indices.
    :return: A tuple of lists of computed metric values for all users, head users, and tail users respectively.
    """
    head_indices = head_indices.to(device)
    tail_indices = tail_indices.to(device)

    for metric in metrics:
        metric.reset()

    groups = ['all', 'head', 'tail']
    calculators = {group: [metric.clone().to(device) for metric in metrics] for group in groups}

    for user1, user2, labels in eval_loader:
        probs = predictor(user1, user2)

        for calculator in calculators['all']:
            calculator.update(probs, labels)

        for group, indices in [('head', head_indices), ('tail', tail_indices)]:
            mask1 = torch.isin(user1, indices)
            mask2 = torch.isin(user2, indices)

            combined_probs = torch.cat([probs[mask1], probs[mask2]], dim=0)
            combined_labels = torch.cat([labels[mask1], labels[mask2]], dim=0)

            if combined_labels.numel() > 0:
                for calculator in calculators[group]:
                    calculator.update(combined_probs, combined_labels)

    def compute_group_metrics(group: str) -> list[float]:
        return [typing.cast(torch.Tensor, calculator.compute()).item() for calculator in calculators[group]]

    return compute_group_metrics('all'), compute_group_metrics('head'), compute_group_metrics('tail')


class TPRatFPR(Metric):
    """
    TPR at a specified FPR, with tied scores grouped and linear ROC interpolation.
    """
    full_state_update = True

    def __init__(self, target_fpr: float):
        """
        Initialize the TPRatFPR metric.

        :param target_fpr: The target false positive rate.
        """
        super().__init__(dist_sync_on_step=False)

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

        # A threshold includes every sample with the same score at once
        threshold_ends = torch.cat([
            preds[:-1] != preds[1:],
            torch.ones(1, dtype=torch.bool, device=preds.device)
        ])
        tp = torch.cumsum(target == 1, dim=0)[threshold_ends]
        fp = torch.cumsum(target == 0, dim=0)[threshold_ends]
        origin = torch.zeros(1, device=preds.device)
        tpr = torch.cat([origin, tp / pos])
        fpr = torch.cat([origin, fp / neg])

        # Use the last point on an FPR plateau, including at an exact match
        idx = torch.searchsorted(fpr, fpr.new_tensor(self.target_fpr), right=True)
        if idx == fpr.numel():
            return tpr[-1]

        fpr0, fpr1 = fpr[idx-1], fpr[idx]
        tpr0, tpr1 = tpr[idx-1], tpr[idx]

        alpha = (self.target_fpr - fpr0) / (fpr1 - fpr0)
        result = tpr0 + alpha * (tpr1 - tpr0)
        return result

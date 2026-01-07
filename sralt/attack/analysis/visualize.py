import typing
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from sralt.attack.data import AttackData, EvalAttackDataset
from sralt.attack.model import AttackModel
from sralt.utils import set_seed, head_tail_split, DataLoader


@torch.no_grad()
def visualize_attack_probs(attack_model: AttackModel, attack_data: AttackData, *, tail_ratio: float, seed: int, save_path: str):
    """
    Visualize attack model predictions as smooth KDE curves for head/tail groups and member/non-member classes.

    :param attack_model: The trained attack model.
    :param attack_data: The attack dataset.
    :param tail_ratio: The ratio to be considered as tail users.
    :param device: The device to run the model on.
    :param save_path: Path to save the visualization figure.
    """
    attack_model.eval()
    set_seed(seed)

    # Load test dataset
    test_dataset = EvalAttackDataset(attack_data, split='test')
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    # Get head/tail split
    head_indices, tail_indices = head_tail_split(
        attack_data.num_users,
        attack_data.target_social_edge_index,
        tail_ratio=tail_ratio
    )
    head_indices_set = set(head_indices.cpu().tolist())
    tail_indices_set = set(tail_indices.cpu().tolist())
    if head_indices_set | tail_indices_set != set(range(attack_data.num_users)):
        raise ValueError('Head and tail indices do not cover all users.')

    # Collect predictions
    all_probs = []
    all_labels = []
    all_u1 = []
    all_u2 = []

    for u1_batch, u2_batch, edge_label in test_loader:
        u1_batch: torch.Tensor
        u2_batch: torch.Tensor
        edge_label: torch.Tensor

        edge_index_batch = torch.stack([u1_batch, u2_batch])
        probs = attack_model.predict(edge_index_batch, attack_data.user_features)

        all_probs.append(probs.cpu())
        all_labels.append(edge_label.cpu())
        all_u1.append(u1_batch.cpu())
        all_u2.append(u2_batch.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_u1 = torch.cat(all_u1).numpy()
    all_u2 = torch.cat(all_u2).numpy()

    # Separate into head/tail based on user indices
    head_probs_pos, head_probs_neg = [], []
    tail_probs_pos, tail_probs_neg = [], []
    probs_list_map = {
        (True, 1): head_probs_pos,
        (True, 0): head_probs_neg,
        (False, 1): tail_probs_pos,
        (False, 0): tail_probs_neg
    }

    for prob, label, *users in zip(all_probs, all_labels, all_u1, all_u2):
        for user in users:
            probs_list_map[(user in head_indices_set, label)].append(prob)

    if any(len(prob_list) == 0 for prob_list in probs_list_map.values()):
        raise ValueError('Some groups have no samples for visualization.')

    # Create smooth curves using KDE
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = typing.cast(tuple[plt.Axes, ...], (ax1, ax2, ax3, ax4))

    x_range = np.linspace(0, 1, 500)
    style: dict[str, typing.Any] = {'linewidth': 2.5, 'alpha': 0.8}

    kde_head_pos = gaussian_kde(np.array(head_probs_pos), bw_method=0.05)
    kde_tail_pos = gaussian_kde(np.array(tail_probs_pos), bw_method=0.05)
    kde_head_neg = gaussian_kde(np.array(head_probs_neg), bw_method=0.05)
    kde_tail_neg = gaussian_kde(np.array(tail_probs_neg), bw_method=0.05)

    def set_annotations(ax: plt.Axes, title: str):
        ax.set_xlabel('Attack Probability', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)

    # Top-left: Member (label=1)
    ax1.plot(x_range, kde_head_pos(x_range), label=f'Head (n={len(head_probs_pos)})', color='#E74C3C', **style)
    ax1.plot(x_range, kde_tail_pos(x_range), label=f'Tail (n={len(tail_probs_pos)})', color='#3498DB', **style)
    set_annotations(ax1, 'Member (Label=1)')

    # Top-right: Non-member (label=0)
    ax2.plot(x_range, kde_head_neg(x_range), label=f'Head (n={len(head_probs_neg)})', color='#E74C3C', **style)
    ax2.plot(x_range, kde_tail_neg(x_range), label=f'Tail (n={len(tail_probs_neg)})', color='#3498DB', **style)
    set_annotations(ax2, 'Non-member (Label=0)')

    # Bottom-left: Head users
    ax3.plot(x_range, kde_head_pos(x_range), label=f'Member (n={len(head_probs_pos)})', color='#27AE60', **style)
    ax3.plot(x_range, kde_head_neg(x_range), label=f'Non-member (n={len(head_probs_neg)})', color='#E67E22', **style)
    set_annotations(ax3, 'Head Users')

    # Bottom-right: Tail users
    ax4.plot(x_range, kde_tail_pos(x_range), label=f'Member (n={len(tail_probs_pos)})', color='#27AE60', **style)
    ax4.plot(x_range, kde_tail_neg(x_range), label=f'Non-member (n={len(tail_probs_neg)})', color='#E67E22', **style)
    set_annotations(ax4, 'Tail Users')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f'Attack probability visualization saved to {save_path}')

    # Log statistics
    logging.info(f'Head - Member mean: {np.mean(head_probs_pos):.4f}, Non-member mean: {np.mean(head_probs_neg):.4f}')
    logging.info(f'Tail - Member mean: {np.mean(tail_probs_pos):.4f}, Non-member mean: {np.mean(tail_probs_neg):.4f}')

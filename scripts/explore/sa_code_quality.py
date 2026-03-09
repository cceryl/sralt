# This script visualizes the code quality learned by the Semantic Augmentation (SA) model
# 1. Visualize the learned user embeddings.
# 2. Visualize the distribution of codebook usage.
# 3. See the ratio of social connected users, who are mapped to the same code.

import os
import logging
import warnings

import torch
import torch.optim as optim
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torch_geometric.utils import to_undirected

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sralt.target.data import load_target_data, TargetData
from sralt.target.model import load_target_model, test_target_model, TargetModel
from sralt.shadow.data import load_shadow_data, ShadowData
from sralt.shadow.model import load_shadow_model, test_shadow_model, ShadowModel
from sralt.shadow.model.sa import SemanticAugmentation
from sralt.utils import setup_logging, get_device


def target_stage(
    seed: int, deterministic: bool, device: torch.device
) -> tuple[TargetData, TargetModel]:
    target_data = load_target_data(
        seed=seed,
        deterministic=deterministic,
        name='Ciao',
        directory='./data/ciao',
        config={
            'min_rating': 2,
            'min_social_connections': 1,
            'min_user_interactions': 5,
            'min_item_interactions': 1
        }
    ).to(device)

    target_model = load_target_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        name='DESIGN',
        model_config={
            'embed_dim': 64,
            'user_fusion_layers': [0, 1, 2],
            'item_fusion_layers': [0, 1, 2],
            'decay': 5e-3,
            'distill_weight': 1e-2
        },
        data=target_data,
        train_config={
            'train_batch_size': 32768,
            'val_batch_size': 4096,
            'val_num_negatives': 999,
            'num_epochs': 300,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': 1e-2},
            'metrics': [LinkPredRecall(k=10), LinkPredNDCG(k=10)],
            'metric_aggregator': 'lambda loss, metrics: sum(metrics)',
            'early_stopping_kwargs': {'patience': 10, 'greater_is_better': True}
        },
        writer=None
    )

    test_metrics = test_target_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        model=target_model,
        data=target_data,
        metrics=[LinkPredRecall(k=10), LinkPredNDCG(k=10)],
        test_config={
            'test_batch_size': 4096,
            'test_num_negatives': 999
        }
    )

    for name, value in zip(['Recall@10', 'NDCG@10'], test_metrics):
        logging.info(f'Target model test {name}: {value:.4f}')

    logging.info(f'Target model representation: {target_model}')
    return target_data, target_model


def shadow_stage(
    seed: int, deterministic: bool, device: torch.device,
    target_data: TargetData, target_model: TargetModel
) -> tuple[ShadowData, ShadowModel]:
    shadow_data = load_shadow_data(
        seed=seed,
        deterministic=deterministic,
        target_data=target_data,
        target_model=target_model,
        shadow_social_ratio=0.1,
        num_recommendations=30,
        inference_batch_size=1024
    ).to(device)

    shadow_model = load_shadow_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        name='SemanticAugmentation',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-3,
            'rvq_num_quantizers': 8,
            'rvq_codebook_size': 256,
            'commitment_weight': 0.0,
            'ablation_code': 'base',
            'ablation_feature': 'base'
        },
        data=shadow_data,
        train_config={
            'train_batch_size': 8192,
            'val_batch_size': 2048,
            'val_num_negatives': 999,
            'num_epochs': 100,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': 1e-2},
            'metrics': [LinkPredRecall(k=10), LinkPredNDCG(k=10)],
            'metric_aggregator': 'lambda loss, metrics: sum(metrics)',
            'early_stopping_kwargs': {'patience': 10, 'greater_is_better': True}
        },
        writer=None
    )

    test_metrics = test_shadow_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        model=shadow_model,
        data=shadow_data,
        metrics=[LinkPredRecall(k=10), LinkPredNDCG(k=10)],
        test_config={
            'test_batch_size': 2048,
            'test_num_negatives': 999,
            'stage': 'social'
        }
    )

    for name, value in zip(['Recall@10', 'NDCG@10'], test_metrics):
        logging.info(f'Shadow model test {name}: {value:.4f}')

    logging.info(f'Shadow model representation: {shadow_model}')
    return shadow_data, shadow_model


def vis_user_embeddings(shadow_data: ShadowData, shadow_model: ShadowModel, base_dir: str):
    assert isinstance(shadow_model, SemanticAugmentation)

    # Extract embeddings
    behavioral_edge_index = to_undirected(shadow_data.behavioral_edge_index)
    social_edge_index = shadow_data.social_edge_index

    embeds = {
        "Behavioral": shadow_model.get_embedding('behavioral', social_edge_index, behavioral_edge_index),
        "Social": shadow_model.get_embedding('social', social_edge_index, behavioral_edge_index),
        "Features": shadow_model.get_user_features(shadow_data)
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (title, data) in zip(axes, embeds.items()):
        # Project to 2D
        tsne = TSNE(n_components=2, random_state=42)
        low_dim = tsne.fit_transform(data.detach().cpu().numpy())

        ax.scatter(low_dim[:, 0], low_dim[:, 1], s=5, alpha=0.6)
        ax.set_title(f"2D Projection of {title}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'user_embeddings.png'))
    plt.close()


def vis_codebook_usage(shadow_model: ShadowModel, base_dir: str):
    assert isinstance(shadow_model, SemanticAugmentation)

    codes = shadow_model.user_codes.cpu().numpy()  # (num_users, num_quantizers)
    flat_codes = codes.flatten()

    # Calculate counts for each possible code index
    unique_codes, counts = np.unique(flat_codes, return_counts=True)
    usage_map = dict(zip(unique_codes, counts))

    all_indices = np.arange(shadow_model.rvq_num_quantizers * shadow_model.rvq_codebook_size)
    all_counts = np.array([usage_map.get(i, 0) for i in all_indices])

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Graph 1: Usage per Index
    ax1.bar(all_indices, all_counts, color='skyblue')
    ax1.set_xlabel("Code Index")
    ax1.set_ylabel("Usage Count")
    ax1.set_title("Codebook Usage Frequency")

    # Graph 2: Sorted Histogram (Long Tail)
    sorted_counts = np.sort(all_counts)[::-1]
    ax2.plot(sorted_counts, color='red')
    ax2.fill_between(range(len(sorted_counts)), sorted_counts, color='red', alpha=0.2)
    ax2.set_xlabel("Rank (High to Low)")
    ax2.set_ylabel("Usage Count")
    ax2.set_title("Usage Distribution (Sorted)")

    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'codebook_usage.png'))
    plt.close()


def compute_ratios(codes, edge_set, num_edges):
    num_users, num_quantizers = codes.shape
    hits = 0
    total_collisions = 0

    for q in range(num_quantizers):
        q_codes = codes[:, q]
        # Group users by code index
        for code_val in torch.unique(q_codes):
            users_with_code = torch.where(q_codes == code_val)[0].tolist()
            if len(users_with_code) < 2:
                continue

            # For every pair in this code bucket
            from itertools import combinations
            for u, v in combinations(users_with_code, 2):
                total_collisions += 1
                if (u, v) in edge_set or (v, u) in edge_set:
                    hits += 1

    hit_ratio = hits / num_edges if num_edges > 0 else 0
    miss_ratio = (total_collisions - hits) / num_edges if num_edges > 0 else 0
    return hit_ratio, miss_ratio


def vis_social_connected_user_ratio(target_data: TargetData, shadow_model: ShadowModel, base_dir: str):
    assert isinstance(shadow_model, SemanticAugmentation)

    # Create a fast lookup set for real social edges
    edges = target_data.social_edge_index.t().tolist()
    real_edge_set = set(tuple(e) for e in edges)
    num_real_edges = len(real_edge_set)

    # Actual Model Codes
    user_codes = shadow_model.user_codes
    h_ratio, m_ratio = compute_ratios(user_codes, real_edge_set, num_real_edges)

    # Random Baseline
    random_codes = torch.randint(
        0, shadow_model.rvq_num_quantizers * shadow_model.rvq_codebook_size, user_codes.shape
    )
    h_ratio_rnd, m_ratio_rnd = compute_ratios(random_codes, real_edge_set, num_real_edges)

    # Display Results
    logging.info(f"{'Metric':<20} | {'Model':<10} | {'Random':<10}")
    logging.info("-" * 45)
    logging.info(f"{'Hit Ratio':<20} | {h_ratio:<10.4f} | {h_ratio_rnd:<10.4f}")
    logging.info(f"{'Miss Ratio':<20} | {m_ratio:<10.4f} | {m_ratio_rnd:<10.4f}")

    # Save to file
    with open(os.path.join(base_dir, 'social_connected_user_ratio.txt'), 'w') as f:
        f.write(f"{'Metric':<20} | {'Model':<10} | {'Random':<10}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'Hit Ratio':<20} | {h_ratio:<10.4f} | {h_ratio_rnd:<10.4f}\n")
        f.write(f"{'Miss Ratio':<20} | {m_ratio:<10.4f} | {m_ratio_rnd:<10.4f}\n")


def main():
    root_path = f'runs/explore/sa/code_quality'
    if os.path.exists(root_path):
        import shutil
        shutil.rmtree(root_path)
    os.makedirs(root_path)
    setup_logging(log_file=os.path.join(root_path, 'search.log'), tz='Asia/Shanghai')

    seed = 42
    deterministic = True
    device = get_device(allow_share=True)

    target_data, target_model = target_stage(seed, deterministic, device)
    shadow_data, shadow_model = shadow_stage(seed, deterministic, device, target_data, target_model)

    del target_model
    cpu = torch.device('cpu')
    target_data = target_data.to(cpu)
    shadow_data = shadow_data.to(cpu)
    shadow_model = shadow_model.to(cpu)

    vis_user_embeddings(shadow_data, shadow_model, root_path)
    vis_codebook_usage(shadow_model, root_path)
    vis_social_connected_user_ratio(target_data, shadow_model, root_path)


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

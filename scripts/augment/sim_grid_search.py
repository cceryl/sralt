import os
import time
import logging
import warnings
import itertools

import torch
import torch.optim as optim
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG
from torchmetrics import AUROC, F1Score, Precision, Recall

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sralt.target import load_target_data, load_target_model, test_target_model, TargetData
from sralt.shadow import load_shadow_data, load_shadow_model, test_shadow_model, ShadowData, ShadowModel
from sralt.attack import load_attack_data, load_attack_model, test_attack_model
from sralt.utils import setup_logging, get_device
from sralt.utils import AdamWithGradientClipping, TPRatFPR


def prepare_data(
    seed: int, deterministic: bool, device: torch.device
) -> tuple[TargetData, ShadowData]:
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

    shadow_data = load_shadow_data(
        seed=seed,
        deterministic=deterministic,
        target_data=target_data,
        target_model=target_model,
        shadow_social_ratio=0.1,
        num_recommendations=30,
        inference_batch_size=1024
    ).to(device)

    return target_data, shadow_data


def shadow_stage_baseline(
    seed: int, deterministic: bool, device: torch.device, shadow_data: ShadowData
) -> tuple[ShadowModel, list[float]]:
    shadow_model = load_shadow_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        name='SMIA',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': 1e-2
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
            'test_num_negatives': 999
        }
    )

    for name, value in zip(['Recall@10', 'NDCG@10'], test_metrics):
        logging.info(f'Shadow model test {name}: {value:.4f}')

    logging.info(f'Shadow model representation: {shadow_model}')
    return shadow_model, test_metrics


def shadow_stage_sim_search(
    seed: int, deterministic: bool, device: torch.device, shadow_data: ShadowData, sim_config: dict
) -> tuple[ShadowModel, list[float]]:
    shadow_model = load_shadow_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        name='SimilarityAugmentation',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': sim_config['decay'],
            'top_k': sim_config['top_k'],
            'sim': sim_config['sim'],
            'undirected_augment': sim_config['undirected_augment'],
            'ablation_feature': sim_config['ablation_feature']
        },
        data=shadow_data,
        train_config={
            'train_batch_size': 8192,
            'val_batch_size': 2048,
            'val_num_negatives': 999,
            'num_epochs': 300,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': sim_config['lr']},
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
        test_config={'test_batch_size': 2048, 'test_num_negatives': 999, 'stage': 'social'}
    )
    return shadow_model, test_metrics


def attack_stage(
    base_seed: int, num_seeds: int, deterministic: bool,  device: torch.device,
    target_data: TargetData, shadow_data: ShadowData, shadow_model: ShadowModel
) -> dict[str, dict[str, tuple[float, float]]]:
    def attack_single_seed(seed: int):
        attack_data = load_attack_data(
            seed=seed,
            deterministic=deterministic,
            target_data=target_data,
            shadow_data=shadow_data,
            shadow_model=shadow_model,
            val_ratio=0.1
        ).to(device)

        attack_model = load_attack_model(
            seed=seed,
            deterministic=deterministic,
            device=device,
            name='SMIA',
            model_config={
                'dropout': 0.5,
                'leaky_relu_slope': 0.2,
                'pos_weight': 10.0
            },
            data=attack_data,
            tail_ratio=0.8,
            train_config={
                'train_batch_size': 512,
                'val_batch_size': 2048,
                'negative_ratio': 1.0,
                'alpha': 0.0,
                'num_epochs': 50,
                'optimizer_cls': AdamWithGradientClipping,
                'optimizer_kwargs': {'lr': 1e-3, 'weight_decay': 1e-4, 'grad_clip_norm': 1.0},
                'metrics': [
                    AUROC(task='binary'),
                    F1Score(task='binary'),
                    TPRatFPR(target_fpr=0.01),
                    Precision(task='binary'),
                    Recall(task='binary')
                ],
                'metric_aggregator': 'lambda loss, metrics: metrics[0] + metrics[1]',
                'early_stopping_kwargs': {'patience': 5, 'greater_is_better': True}
            },
            writer=None
        )

        test_metrics = test_attack_model(
            seed=seed,
            deterministic=deterministic,
            device=device,
            model=attack_model,
            data=attack_data,
            tail_ratio=0.8,
            metrics=[
                AUROC(task='binary'),
                F1Score(task='binary'),
                TPRatFPR(target_fpr=0.01),
                Precision(task='binary'),
                Recall(task='binary')
            ],
            test_config={
                'test_batch_size': 2048
            }
        )

        results_dict = {}
        for group, metrics in zip(['All', 'Head', 'Tail'], test_metrics):
            results_dict[group] = {
                name: value for name, value in
                zip(['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall'], metrics)
            }

        return results_dict

    # Aggregate results over multiple seeds as [mean, std]
    aggregated_results = {}
    all_seed_results = [attack_single_seed(base_seed + i) for i in range(num_seeds)]
    for group in ['All', 'Head', 'Tail']:
        aggregated_results[group] = {}
        for metric in ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']:
            values = [res[group][metric] for res in all_seed_results]
            mean_val = sum(values) / len(values)
            std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            aggregated_results[group][metric] = (mean_val, std_val)

    return aggregated_results


def logs_and_plots(
    root_path: str, sorted_results: list[dict],
    baseline_shadow_metrics: list[float],
    baseline_attack_metrics: dict[str, dict[str, tuple[float, float]]]
):
    def shadow_metrics_str(metrics: list[float]) -> str:
        base = baseline_shadow_metrics
        names = ['Recall@10', 'NDCG@10']
        diffs = [metrics[i] - base[i] for i in range(len(names))]
        return ', '.join(f"{name:}: {metrics[i]:.4f} (Δ {diffs[i]:+.4f})" for i, name in enumerate(names))

    def attack_metrics_str(metrics: dict[str, dict[str, tuple[float, float]]]) -> str:
        base = baseline_attack_metrics
        parts = []
        for group in ['All', 'Head', 'Tail']:
            names = ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']
            diffs_percentage = [
                (metrics[group][name][0] - base[group][name][0]) / base[group][name][0] * 100.0
                for name in names
            ]
            group_str = ', '.join(
                f"{name}: {metrics[group][name][0]:.4f} ± {metrics[group][name][1]:.4f} ({diffs_percentage[i]:+.2f}%)"
                for i, name in enumerate(names)
            )
            parts.append(f"{group:4} -> {group_str}")
        return '\n    '.join(parts)

    logging.info(f"Results saved to {root_path}")

    # log baseline metrics
    logging.info(f"\nBaseline Shadow Metrics: {shadow_metrics_str(baseline_shadow_metrics)}")
    logging.info(f"Baseline Attack Metrics:\n    {attack_metrics_str(baseline_attack_metrics)}")

    for rank, res in enumerate(sorted_results):
        config = res['config']
        shadow = res['shadow']
        attack = res['attack']

        logging.info(
            f"\n"
            f"Rank {rank+1}:\n"
            f"  Config: {config}\n"
            f"  Shadow Metrics: {shadow_metrics_str(shadow)}\n"
            f"  Attack Metrics:\n    {attack_metrics_str(attack)}"
        )

    # Save full results to csv
    summary_rows = []
    for rank, res in enumerate(sorted_results):
        config = res['config']
        shadow = res['shadow']
        attack = res['attack']

        row = {'Rank': rank + 1}
        row.update({f'Config_{k}': v for k, v in config.items()})
        row.update({f'Shadow_{name}': value for name, value in zip(
            ['Recall@10', 'NDCG@10'], shadow
        )})
        for group in ['All', 'Head', 'Tail']:
            for metric in ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']:
                base_mean = baseline_attack_metrics[group][metric][0]
                mean, std = attack[group][metric]
                row[f'Attack_{group}_{metric}_Mean'] = mean
                row[f'Attack_{group}_{metric}_Std'] = std
                row[f'Attack_{group}_{metric}_Diff'] = (mean - base_mean) / base_mean

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(root_path, 'search_summary.csv'), index=False)

    # Save best result to csv as comparison with baseline
    best_res = sorted_results[0]
    comparison_rows = []
    for group in ['All', 'Head', 'Tail']:
        group_name_in_row = group
        for metric in ['AUROC', 'F1', 'TPR@FPR=0.01', 'Precision', 'Recall']:
            base_mean, base_std = baseline_attack_metrics[group][metric]
            sa_mean, sa_std = best_res['attack'][group][metric]
            delta_percentage = (sa_mean - base_mean) / base_mean * 100.0

            comparison_rows.append({
                'Group': group_name_in_row,
                'Metric': metric,
                'Baseline_Mean': f"{base_mean:.4f}",
                'Baseline_Std': f"{base_std:.4f}",
                'Semantic_Augmentation_Mean': f"{sa_mean:.4f}",
                'Semantic_Augmentation_Std': f"{sa_std:.4f}",
                'Delta_Percentage': f"{delta_percentage:+.2f}%"
            })
            group_name_in_row = ''  # Clear group name for subsequent rows
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(root_path, 'best_comparison.csv'), index=False)

    # Plot AUROC, F1, TPR@FPR=0.01, only for the best configuration
    groups = ['All', 'Head', 'Tail']
    metrics_names = ['AUROC', 'F1', 'TPR@FPR=0.01']

    plot_labels = []
    raw_values = {metric: [] for metric in metrics_names}
    plot_errs = {metric: [] for metric in metrics_names}

    for g in groups:
        models_in_group = [
            (f"Base-{g}", baseline_attack_metrics[g]),
            (f"SA-{g}", best_res['attack'][g])
        ]
        for label, results in models_in_group:
            plot_labels.append(label)
            for m in metrics_names:
                mean, std = results[m]
                raw_values[m].append(mean)
                plot_errs[m].append(std)

    x = np.arange(len(plot_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(15, 8))

    rects_auc = ax.bar(x - width, raw_values['AUROC'], width, yerr=plot_errs['AUROC'],
                       label='AUC', capsize=3, color='#4E79A7', edgecolor='white', alpha=0.9)
    rects_f1 = ax.bar(x, raw_values['F1'], width, yerr=plot_errs['F1'],
                      label='F1', capsize=3, color='#F28E2B', edgecolor='white', alpha=0.9)
    rects_tpr = ax.bar(x + width, raw_values['TPR@FPR=0.01'], width, yerr=plot_errs['TPR@FPR=0.01'],
                       label='TPR@FPR=0.01', capsize=3, color='#E15759', edgecolor='white', alpha=0.9)

    def add_delta_labels(rects, metric_key):
        for i in range(1, len(rects), 2):
            base_val = raw_values[metric_key][i-1]
            sa_val = raw_values[metric_key][i]

            if base_val > 0:
                delta = (sa_val - base_val) / base_val * 100
                text = f'{delta:+.1f}%'
                color = 'green' if delta >= 0 else 'red'

                height = rects[i].get_height()
                ax.annotate(text,
                            xy=(rects[i].get_x() + rects[i].get_width() / 2, height),
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, color=color, fontweight='bold')

    add_delta_labels(rects_auc, 'AUROC')
    add_delta_labels(rects_f1, 'F1')
    add_delta_labels(rects_tpr, 'TPR@FPR=0.01')

    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Detailed Comparison with Delta (%)', fontsize=15, pad=30)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for i in range(2, len(plot_labels), 2):
        ax.axvline(x=i-0.5, color='black', linestyle='-', alpha=0.1)

    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, 'paired_comparison_with_delta.png'), dpi=300)
    plt.close()


def main():
    time_str = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    root_path = f'runs/augment/sim_grid_search/{time_str}'
    os.makedirs(root_path, exist_ok=True)
    setup_logging(log_file=os.path.join(root_path, 'search.log'), tz='Asia/Shanghai')

    seed = 42
    num_attacker_seeds = 10
    deterministic = True
    device = torch.device('cuda:1')
    target_data, shadow_data = prepare_data(seed, deterministic, device)

    # Baseline shadow model
    baseline_shadow, baseline_shadow_metrics = shadow_stage_baseline(seed, deterministic, device, shadow_data)
    baseline_attack_metrics = attack_stage(
        seed, num_attacker_seeds, deterministic, device, target_data, shadow_data, baseline_shadow
    )
    del baseline_shadow

    param_grid = {
        'decay': [1e-3],  # [1e-1, 1e-2, 1e-3, 1e-4],
        'top_k': [8],  # [2, 4, 8, 16],
        'sim': ['dot', 'cosine', 'random'],  # ['cosine', 'dot', 'random'],
        'undirected_augment': [True],  # [True, False],
        'ablation_feature': ['base'],  # ['base', 'without-social', 'without-behavioral'],
        'lr': [1e-2]
    }

    # cosine - top_k=4, undirected_augment=True, decay=1e-2
    #    All  -> AUROC: 0.9320 ± 0.0031 (+2.12%), F1: 0.8537 ± 0.0040 (+3.40%), TPR@FPR=0.01: 0.3960 ± 0.0124 (+0.64%), Precision: 0.8058 ± 0.0126 (-6.94%), Recall: 0.9081 ± 0.0157 (+15.07%)
    #    Head -> AUROC: 0.9116 ± 0.0034 (+1.80%), F1: 0.9279 ± 0.0030 (+2.67%), TPR@FPR=0.01: 0.2734 ± 0.0134 (+3.95%), Precision: 0.9039 ± 0.0055 (-1.02%), Recall: 0.9535 ± 0.0104 (+6.57%)
    #    Tail -> AUROC: 0.8919 ± 0.0048 (+5.82%), F1: 0.6848 ± 0.0078 (+14.65%), TPR@FPR=0.01: 0.2600 ± 0.0143 (+8.49%), Precision: 0.6042 ± 0.0183 (-14.27%), Recall: 0.7921 ± 0.0293 (+52.66%)
    # random - top_k=2, undirected_augment=True, decay=1e-3
    #    All  -> AUROC: 0.9290 ± 0.0013 (+1.79%), F1: 0.8455 ± 0.0038 (+2.41%), TPR@FPR=0.01: 0.3960 ± 0.0062 (+0.64%), Precision: 0.7814 ± 0.0142 (-9.76%), Recall: 0.9216 ± 0.0113 (+16.78%)
    #    Head -> AUROC: 0.9056 ± 0.0021 (+1.13%), F1: 0.9238 ± 0.0013 (+2.22%), TPR@FPR=0.01: 0.2683 ± 0.0108 (+2.03%), Precision: 0.8855 ± 0.0058 (-3.03%), Recall: 0.9657 ± 0.0063 (+7.94%)
    #    Tail -> AUROC: 0.8851 ± 0.0024 (+5.01%), F1: 0.6715 ± 0.0062 (+12.42%), TPR@FPR=0.01: 0.2670 ± 0.0078 (+11.41%), Precision: 0.5752 ± 0.0213 (-18.38%), Recall: 0.8085 ± 0.0245 (+55.83%)
    # random - top_k=4, undirected_augment=True, decay=1e-2
    #    All  -> AUROC: 0.9242 ± 0.0010 (+1.26%), F1: 0.8275 ± 0.0062 (+0.22%), TPR@FPR=0.01: 0.3774 ± 0.0062 (-4.10%), Precision: 0.7348 ± 0.0153 (-15.14%), Recall: 0.9474 ± 0.0095 (+20.05%)
    #    Head -> AUROC: 0.8976 ± 0.0018 (+0.24%), F1: 0.9204 ± 0.0014 (+1.83%), TPR@FPR=0.01: 0.2546 ± 0.0104 (-3.21%), Precision: 0.8703 ± 0.0062 (-4.69%), Recall: 0.9766 ± 0.0052 (+9.15%)
    #    Tail -> AUROC: 0.8819 ± 0.0019 (+4.63%), F1: 0.6418 ± 0.0101 (+7.46%), TPR@FPR=0.01: 0.2553 ± 0.0053 (+6.51%), Precision: 0.5083 ± 0.0195 (-27.87%), Recall: 0.8726 ± 0.0210 (+68.18%)
   
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    logging.info(f"Starting Hyperparameter Search. Total combinations: {len(combinations)}")

    results = []

    for i, config in enumerate(combinations):
        logging.info(f"\n{'='*20} [{i+1}/{len(combinations)}] Testing configuration {'='*20}")
        logging.info(f"Configuration: {config}")

        try:
            shadow_model, shadow_metrics = shadow_stage_sim_search(seed, deterministic, device, shadow_data, config)
            attack_metrics = attack_stage(
                seed, num_attacker_seeds, deterministic, device, target_data, shadow_data, shadow_model
            )
            del shadow_model

            results.append({
                'config': config,
                'shadow': shadow_metrics,
                'attack': attack_metrics
            })

            with open(os.path.join(root_path, 'intermediate_results.txt'), 'a') as f:
                f.write(f"Config {i+1}: {config}\n")
                f.write(f"  Shadow Metrics: {shadow_metrics}\n")
                f.write(f"  Attack Metrics: {attack_metrics}\n\n")

            with open(os.path.join(root_path, 'intermediate_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            

        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            logging.error(f"Run {i} failed with error: {e}")
            logging.error(traceback_str)

    # Summarize results
    logging.info("\n" + "#"*30 + " SEARCH SUMMARY " + "#"*30)

    def weighted_result(result: dict[str, dict[str, tuple[float, float]]]) -> float:
        all = result['All']
        head = result['Head']
        tail = result['Tail']

        all_val = all['AUROC'][0] + all['F1'][0] + all['TPR@FPR=0.01'][0] * 2.0
        head_val = head['AUROC'][0] + head['F1'][0] + head['TPR@FPR=0.01'][0] * 2.0
        tail_val = tail['AUROC'][0] + tail['F1'][0] + tail['TPR@FPR=0.01'][0] * 2.0

        return all_val + head_val * 0.5 + tail_val * 2.0

    sorted_results = sorted(results, key=lambda x: weighted_result(x['attack']), reverse=True)
    logs_and_plots(
        root_path,
        sorted_results,
        baseline_shadow_metrics,
        baseline_attack_metrics
    )

    logging.info(f"Grid search completed, results saved to {root_path}")


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

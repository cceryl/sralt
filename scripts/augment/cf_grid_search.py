import os
import time
import logging
import warnings
import itertools

import torch
import torch.optim as optim
import pickle
from torch_geometric.metrics import LinkPredRecall, LinkPredNDCG


from sralt.shadow import load_shadow_model, test_shadow_model, ShadowData, ShadowModel
from sralt.utils import setup_logging, get_device
from scripts.augment.base import shadow_stage_baseline, attack_stage, logs_and_plots, prepare_data


def shadow_stage_search(
    seed: int, deterministic: bool, device: torch.device, shadow_data: ShadowData, config: dict
) -> tuple[ShadowModel, list[float]]:
    shadow_model = load_shadow_model(
        seed=seed,
        deterministic=deterministic,
        device=device,
        name='CollaborativeFilter',
        model_config={
            'embed_dim': 64,
            'num_layers': 2,
            'decay': config['decay'],
            'ablation_feature': config['ablation_feature']
        },
        data=shadow_data,
        train_config={
            'train_batch_size': 8192,
            'val_batch_size': 2048,
            'val_num_negatives': 999,
            'num_epochs': 300,
            'optimizer_cls': optim.Adam,
            'optimizer_kwargs': {'lr': config['lr']},
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


def main():
    time_str = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    root_path = f'runs/augment/cf_grid_search/{time_str}'
    os.makedirs(root_path, exist_ok=True)
    setup_logging(log_file=os.path.join(root_path, 'search.log'), tz='Asia/Shanghai')

    seed = 42
    num_attacker_seeds = 10
    deterministic = True
    device = get_device(allow_share=True)

    target_data, shadow_data = prepare_data(seed, deterministic, device)

    # Baseline shadow model
    baseline_shadow, baseline_shadow_metrics = shadow_stage_baseline(seed, deterministic, device, shadow_data)
    baseline_attack_metrics = attack_stage(
        seed, num_attacker_seeds, deterministic, device, target_data, shadow_data, baseline_shadow
    )
    del baseline_shadow

    param_grid = {
        'decay': [1e-2, 1e-3, 1e-4],
        'ablation_feature': ['base'],
        'lr': [1e-2]
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    logging.info(f"Starting Hyperparameter Search. Total combinations: {len(combinations)}")

    results = []

    for i, config in enumerate(combinations):
        logging.info(f"\n{'='*20} [{i+1}/{len(combinations)}] Testing configuration {'='*20}")
        logging.info(f"Configuration: {config}")

        try:
            shadow_model, shadow_metrics = shadow_stage_search(seed, deterministic, device, shadow_data, config)
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

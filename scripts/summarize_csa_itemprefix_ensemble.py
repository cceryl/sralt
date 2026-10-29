#!/usr/bin/env python3
"""Read existing JSON only: paired metrics, complementarity, calibration and costs."""

import argparse
import json
from pathlib import Path

from summarize_tail_proxy import summarize as summarize_metrics, write_csv, GROUPS, METRICS


def summarize(run_dir, output_dir=None):
    output_dir = output_dir or run_dir
    success = summarize_metrics(run_dir, output_dir)
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    calibration, complementarity, costs, comparisons, oof_costs = [], [], [], [], []
    for case in manifest['cases']:
        path = run_dir / case['output'] / 'metrics.json'
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        identity = dict(target=case['target'], dataset=case['dataset'], seed=case['seed'])
        if (saved.get('experiment_revision') != manifest['identity']['experiment_revision'] or
                saved.get('source_fingerprint') != manifest['identity']['source_fingerprint']):
            raise ValueError(f'Cannot combine mixed ensemble implementations: {path}')
        for bucket, row in saved.get('public_complementarity', {}).items():
            complementarity.append(dict(**identity, degree_bucket=bucket,
                                         **{k: v for k, v in row.items() if not isinstance(v, dict)}))
        for fit in saved.get('oof', {}).get('auxiliary_fits', []):
            oof_costs.append(dict(**identity, **fit))
        for condition, result in saved['results'].items():
            costs.append(dict(**identity, condition=condition, parameters=result['trainable_parameters'],
                              elapsed_seconds=result['elapsed_seconds'], peak_memory_bytes=result['peak_memory_bytes'],
                              training=result.get('training')))
            for bucket, row in result.get('public_calibration', {}).items():
                calibration.append(dict(**identity, condition=condition, degree_bucket=bucket,
                                        samples=row['samples'], brier=row['brier'], ece=row['ece']))
            if condition in ('csa', 'item_prefix_joint'):
                continue
            references = [saved['results'].get(name) for name in ('csa', 'item_prefix_joint')]
            if not all(references):
                continue
            deltas = {}
            for reference, record in zip(('csa', 'item_prefix_joint'), references, strict=True):
                for group in GROUPS:
                    for metric in METRICS:
                        value, baseline = result['groups'][group][metric], record['groups'][group][metric]
                        deltas[(reference, group, metric)] = value - baseline if value is not None and baseline is not None else None
            valid = [value for value in deltas.values() if value is not None]
            if len(valid) != len(deltas):
                category = 'undefined_metric'
            elif all(value >= 0 for value in valid) and any(value > 0 for value in valid):
                category = 'Pareto_over_both'
            elif all(value < 0 for value in valid):
                category = 'below_both_everywhere'
            else:
                category = 'mixed_or_tradeoff'
            for (reference, group, metric), delta in deltas.items():
                comparisons.append(dict(**identity, condition=condition, reference=reference, group=group,
                                        metric=metric, delta=delta, category=category))
    write_csv(output_dir / 'public_complementarity.csv', complementarity,
              ['target', 'dataset', 'seed', 'degree_bucket', 'samples', 'positives', 'probability_pearson',
               'disagreement_rate', 'csa_only_correct', 'prefix_only_correct', 'both_correct', 'both_wrong'])
    write_csv(output_dir / 'public_calibration.csv', calibration,
              ['target', 'dataset', 'seed', 'condition', 'degree_bucket', 'samples', 'brier', 'ece'])
    write_csv(output_dir / 'training_costs.csv', costs,
              ['target', 'dataset', 'seed', 'condition', 'parameters', 'elapsed_seconds', 'peak_memory_bytes', 'training'])
    write_csv(output_dir / 'oof_costs.csv', oof_costs,
              ['target', 'dataset', 'seed', 'fold', 'expert', 'fit_pairs', 'held_pairs', 'negative_sampling',
               'load_or_fit_seconds', 'training'])
    write_csv(output_dir / 'expert_comparisons.csv', comparisons,
              ['target', 'dataset', 'seed', 'condition', 'reference', 'group', 'metric', 'delta', 'category'])
    with (output_dir / 'report.md').open('a') as handle:
        handle.write('\n公开V诊断见public_complementarity.csv/public_calibration.csv，逐概率bin保存在metrics.json。'
                     'expert_comparisons.csv逐seed同时比较两位原专家，Pareto判据覆盖五组AUC/验证All最优阈值F1/TPR@5%；混合结果不选组别赢家。'
                     'training_costs.csv记录新增参数量和拟合成本；oof_costs.csv将六次辅助拟合各计一次共享成本。\n')
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    return 0 if summarize(args.run_dir.resolve(), args.output_dir.resolve() if args.output_dir else None) else 1


if __name__ == '__main__':
    raise SystemExit(main())

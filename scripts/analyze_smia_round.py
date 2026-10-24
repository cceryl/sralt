#!/usr/bin/env python3
"""Aggregate the two round-1 workers; analyze validation results without loading models."""
import argparse
import json
from pathlib import Path

from tune_smia import reports, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output
    plan = json.loads((root / 'plan.json').read_text())
    records = []
    for dataset in ['ciao', 'flickr']:
        path = root / dataset / 'results.json'
        if path.exists():
            records.extend(json.loads(path.read_text()))
    reports(root, records, plan['total'])
    summary = json.loads((root / 'summary.json').read_text())
    comparisons = []
    lines = ['# SMIA round1：公开验证结果', '',
             f'完成 {len(records)}/{plan["total"]}；没有评估隐藏测试集。', '',
             '候选按两个种子的原始验证 AUROC+F1+TPR@1% 均值排序。', '']
    for dataset, target in [('ciao', 'design'), ('ciao', 'gdmsr'), ('flickr', 'design'), ('flickr', 'gdmsr')]:
        seeds = sorted(i['seed'] for i in plan['inputs'] if i['dataset'] == dataset and i['target'] == target)
        rows = [r for r in summary if r['dataset'] == dataset and r['target'] == target and r['n'] == len(seeds)]
        baseline = next((r for r in rows if r['label'] == 'baseline'), None)
        candidates = [r for r in rows if r['label'] == 'grid']
        lines.extend([f'## {dataset} / {target}', ''])
        if baseline is None or not candidates:
            lines.extend(['等待完整配对。', ''])
            continue
        best = candidates[0]
        best_auc_f1 = max(candidates, key=lambda r: r['auc_f1'])
        simultaneous = [r for r in candidates if r['AUROC'] > baseline['AUROC'] and r['F1'] > baseline['F1']]
        lines.extend(['| 配置 | AUC | F1 | TPR@1% | AUC+F1+TPR |',
                      '| --- | ---: | ---: | ---: | ---: |'])
        for row in [baseline] + candidates[:5]:
            label = f'{row["label"]} lr={row["lr"]} batch={row["batch"]}'
            lines.append(f'| {label} | {row["AUROC"]:.4f} | {row["F1"]:.4f} | {row["TPR01"]:.4f} | {row["selection_score"]:.4f} |')
        differences = []
        for seed in seeds:
            match = lambda setting: next(r for r in records if r['dataset'] == dataset and r['target'] == target
                                         and r['seed'] == seed and all(r['setting'][k] == setting[k] for k in r['setting']))
            b, c = match(baseline), match(best)
            delta = {m: c['validation']['all'][m] - b['validation']['all'][m] for m in ['AUROC', 'F1', 'TPR01']}
            differences.append({'seed': seed, 'delta': delta, 'baseline_best_epoch': b['best_epoch'],
                                'best_epoch': c['best_epoch'], 'stop_epoch': c['stop_epoch']})
        comparisons.append({'dataset': dataset, 'target': target, 'baseline': baseline, 'best_by_original_selection': best,
                            'best_auc_f1_among_restored_checkpoints': best_auc_f1,
                            'simultaneous_mean_improvements': simultaneous,
                            'paired_differences': differences,
                            'both_auc_f1_improve_on_mean': best['AUROC'] > baseline['AUROC'] and best['F1'] > baseline['F1']})
        lines.append('')
        for row in differences:
            d = row['delta']
            lines.append(f'- seed{row["seed"]}: ΔAUC={d["AUROC"]:+.4f}, ΔF1={d["F1"]:+.4f}, '
                         f'ΔTPR={d["TPR01"]:+.4f}; 最佳轮次 {row["baseline_best_epoch"]}→{row["best_epoch"]}，停止于{row["stop_epoch"]}。')
        lines.append(f'原选模checkpoint中AUC+F1最高的候选：lr={best_auc_f1["lr"]}, batch={best_auc_f1["batch"]}, '
                     f'AUC={best_auc_f1["AUROC"]:.4f}, F1={best_auc_f1["F1"]:.4f}。此项仅解释目标取舍，不重新选择epoch。')
        lines.append(f'均值上同时超过baseline AUC和F1的候选数：{len(simultaneous)}。')
        lines.append('')
    lines.extend(['完整结果仍属于调参验证结果，不能当作论文测试指标或独立泛化结论。', '',
                  '本轮完成后停止；任何下一轮或新增测试评估均等待用户确认。', ''])
    (root / 'analysis.md').write_text('\n'.join(lines))
    save_json(root / 'analysis.json', comparisons)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Summarize existing DESIGN IDF-probe artifacts without running experiments."""
import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('All', 'Head', 'Tail', 'TT', 'isolated_Tail')
METRICS = ('AUROC', 'F1', 'TPR@5%')


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads((args.run_dir / 'summary.json').read_text())
    cases, summary = payload['cases'], payload['summary']
    lookup = {(r['dataset'], r['method'], r['group']): r for r in summary}
    datasets = sorted({r['dataset'] for r in cases})
    paired = []
    for dataset in datasets:
        rows = [r for r in cases if r['dataset'] == dataset]
        for comparator in ('smia_mlp', 'raw_intersection', 'public_degree'):
            for group in GROUPS:
                for metric in METRICS:
                    differences = [r['groups']['idf_intersection'][group][metric] - r['groups'][comparator][group][metric]
                                   for r in rows if r['groups']['idf_intersection'][group][metric] is not None
                                   and r['groups'][comparator][group][metric] is not None]
                    paired.append(dict(dataset=dataset, contrast=f'idf_intersection - {comparator}', group=group,
                                       metric=metric, seeds=[r['seed'] for r in rows], values=differences,
                                       positive=sum(v > 0 for v in differences), negative=sum(v < 0 for v in differences),
                                       **describe(differences)))
    old_path = ROOT / 'docs/legacy/2026-10-18-analysis/evidence/cpu-probes.json'
    old = json.loads(old_path.read_text())
    history = []
    for dataset in datasets:
        rows = [r for r in cases if r['dataset'] == dataset]
        for method, old_method in [('idf_intersection', 'rec_idf_overlap'), ('raw_intersection', 'rec_overlap')]:
            for metric, old_metric in [('AUROC', 'auc')]:
                pairs = [(r, q) for r in rows for q in old if q['dataset'] == dataset and q['target'] == 'DESIGN'
                         and q['seed'] == r['seed']]
                current = [r['groups'][method]['All'][metric] for r, _ in pairs]
                previous = [q[old_method][old_metric] for _, q in pairs]
                history.append(dict(source='2026-10-18', path=str(old_path.relative_to(ROOT)), dataset=dataset,
                                    method=method, group='All', metric=metric, n=len(pairs),
                                    old_mean=statistics.mean(previous), current_mean=statistics.mean(current),
                                    difference=statistics.mean(a - b for a, b in zip(current, previous)),
                                    note='same seed labels and formula/RNG rule; old query/split identity not certified'))
    # Exact rounded aggregates transcribed from the archived 2026-10-22 review.
    # Original diagnostics.json was removed during reset; no invented F1/SD/seed values.
    old_tail_path = 'docs/archive/smia-tail-diagnosis/smia-tail-diagnosis-2026-10-22.md'
    old_tail = [('Ciao', 'Tail', 'AUROC', .8729), ('Ciao', 'Tail', 'TPR@1%', .2452),
                ('Ciao', 'TT', 'AUROC', .9223), ('Ciao', 'isolated_Tail', 'AUROC', .8929),
                ('Flickr', 'Tail', 'AUROC', .7904), ('Flickr', 'Tail', 'TPR@1%', .1534)]
    for dataset, group, metric, previous in old_tail:
        if dataset not in datasets or metric not in METRICS:
            continue
        current = lookup[(dataset, 'idf_intersection', group)][metric]
        history.append(dict(source='2026-10-22', path=old_tail_path, dataset=dataset,
                            method='idf_intersection', group=group, metric=metric, n=current['n'],
                            old_mean=previous, current_mean=current['mean'], difference=current['mean'] - previous,
                            note='historical rounded 5-seed aggregate; different target/MLP and independent pairs; descriptive only'))
    (args.run_dir / 'comparisons.json').write_text(json.dumps(dict(paired=paired, historical=history), indent=2) + '\n')
    write_csv(args.run_dir / 'paired_deltas.csv', paired)
    write_csv(args.run_dir / 'historical_comparison.csv', history)
    lines = ['# DESIGN：IDF交集统计评分', '',
             'Ciao/Flickr、seed42–46；IDF、原始交集、公开训练度数与现有SMIA→MLP在同一批诊断用户对上评价。', '',
             '分数为共同Top-30物品的sum log((N+1)/(c_i+1))。无Shadow/分类器训练；F1阈值以公开验证集All最大F1确定并冻结，五组共用；All、Tail、TT为重点。', '',
             '隐藏评价保留全部隐藏正边，独立采样等量真实非边；NumPy RNG=seed+1018。公开验证负例只排除全部公开边，独立RNG=seed+2018。', '',
             'Head/Tail与Isolated Tail按端点加权，TT每对计一次；孤立指公开训练图零度。完整图仅用于隐藏正边、真实非边及评价分组。', '',
             '以下为五seed均值±样本SD，分数以0–1表示。', '']
    for dataset in datasets:
        lines += [f'## {dataset}', '', '| 方法 | 分组 | AUC | F1 | TPR@5% |', '| --- | --- | ---: | ---: | ---: |']
        for method in ('idf_intersection', 'smia_mlp', 'raw_intersection', 'public_degree'):
            for group in GROUPS:
                row = lookup[(dataset, method, group)]
                def cell(metric):
                    value = row[metric]
                    return f'{value["mean"]:.4f}±{value["sample_sd"]:.4f}' if value['n'] > 1 else str(value['mean'])
                lines.append(f'| {method} | {group} | {cell("AUROC")} | {cell("F1")} | {cell("TPR@5%")} |')
        lines.append('')
    lines += ['## 与旧IDF结果比较', '',
              '10月18日逐seedAll诊断和10月22日长尾诊断分表记录；不同旧版本、独立负例、旧人口/ID映射未逐项认证，差值不能归因于单个改动。', '',
              '| 旧来源 | 数据 | 分组 | 指标 | 旧 | 当前 | 差值（百分点） |', '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for r in history:
        if r['method'] == 'idf_intersection':
            lines.append(f'| {r["source"]} | {r["dataset"]} | {r["group"]} | {r["metric"]} | '
                         f'{r["old_mean"]:.4f} | {r["current_mean"]:.4f} | {r["difference"]*100:+.2f} |')
    lines += ['', '旧IDF探针没有可用F1；当前F1统一使用验证All最优阈值，历史固定0.5阈值F1不可混用；TPR@5%不与旧TPR@1%作差。F1是新增结果，不填补旧值。这里的单分数无参数探针也不等于三项推荐统计训练分类器的Rec-IDF-only必要性对照。', '',
              '逐seed指标见metrics.csv；配对差值见paired_deltas.csv；历史差异见historical_comparison.csv。每case保留用户对、分数、阈值、公开验证F1、来源文件SHA256及依赖版本。', '',
              '现有MLP仅CPU推理，未重训。诊断负例与原生产报告不同，CPU/GPU度数并列用户的分组也可能不同；原正式结果不被覆盖。本次执行的是用户请求的统计实验，未运行软件测试。', '']
    (args.run_dir / 'report.md').write_text('\n'.join(lines))
    print(f'{len(cases)} cases; report: {args.run_dir / "report.md"}')


if __name__ == '__main__':
    main()

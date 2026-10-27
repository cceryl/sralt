#!/usr/bin/env python3
"""Summarize saved TailProxy cases and paired seed differences; no model imports."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


EXPERIMENT_REVISION = 'tail-proxy-e100-bce1-v4'
CONDITIONS = ('baseline', 'tail_proxy')

GROUPS = ('All', 'Head', 'Tail', 'TT', 'isolated_Tail')
METRICS = ('AUROC', 'F1Score', 'TPR@FPR=0.01')


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)


def average(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def read_record(run_dir, case, condition):
    directory = run_dir / case['output']
    combined = directory / 'metrics.json'
    if combined.exists():
        saved = json.loads(combined.read_text())
        if condition not in saved['results']:
            raise FileNotFoundError(f'Condition not completed: {directory}/{condition}')
        value = saved['results'][condition]
        record = dict(target=saved['target'], dataset=saved['dataset'], seed=saved['seed'], condition=condition,
                      groups=value['groups'], evaluation_id=saved['evaluation_id'])
    else:
        # Existing experiment evidence remains readable without requiring its
        # redundant score/config copies or verifying file-integrity hashes.
        old = json.loads((directory / condition / 'metrics.json').read_text())
        record = dict(target=old['target'], dataset=old['dataset'], seed=old['seed'], condition=old['condition'],
                      groups=old['groups'], evaluation_id=old['evaluation_sha256'])
    if (record['target'], record['dataset'], record['seed'], record['condition']) != (
            case['target'], case['dataset'], case['seed'], condition):
        raise ValueError('Recorded case identity differs from manifest')
    for group in GROUPS:
        for metric in METRICS:
            if not isinstance(record['groups'][group][metric], (int, float)) or not math.isfinite(record['groups'][group][metric]):
                raise ValueError(f'Undefined required metric: {group}/{metric}')
    return record


def summarize(run_dir, output_dir=None):
    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    expected = manifest.get('conditions', list(CONDITIONS))
    cases, values, records = [], [], {}
    for case in manifest['cases']:
        for condition in expected:
            identity = dict(target=case['target'], dataset=case['dataset'], seed=case['seed'], condition=condition)
            status = dict(**identity, status='missing', worker_status=case['status'])
            cases.append(status)
            try:
                record = read_record(run_dir, case, condition)
                status['status'] = 'completed'
                records[(case['target'], case['dataset'], case['seed'], condition)] = record
                for group in GROUPS:
                    for metric in METRICS:
                        values.append(dict(**identity, group=group, metric=metric, value=record['groups'][group][metric]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                status['status'] = 'missing' if isinstance(error, FileNotFoundError) else 'invalid'
                status['reason'] = str(error)
    buckets = {}
    for row in values:
        key = tuple(row[k] for k in ('target', 'dataset', 'condition', 'group', 'metric'))
        buckets.setdefault(key, []).append(row)
    summary = []
    # Include missing conditions/groups so partial runs cannot appear complete.
    for target in manifest['targets']:
        for dataset in manifest['datasets']:
            for condition in expected:
                for group in GROUPS:
                    for metric in METRICS:
                        rows = buckets.get((target, dataset, condition, group, metric), [])
                        seeds = [r['seed'] for r in rows]
                        summary.append(dict(target=target, dataset=dataset, condition=condition, group=group, metric=metric,
                                            seeds=seeds, missing_seeds=[s for s in manifest['seeds'] if s not in seeds],
                                            **average([r['value'] for r in rows])))
    contrasts = {(condition, 'baseline') for condition in expected if condition != 'baseline'}
    paired, pair_buckets = [], {}
    for case in manifest['cases']:
        for left, right in sorted(contrasts):
            a = records.get((case['target'], case['dataset'], case['seed'], left))
            b = records.get((case['target'], case['dataset'], case['seed'], right))
            status = 'completed' if a and b else 'incomplete'
            if a and b and a['evaluation_id'] != b['evaluation_id']:
                status = 'unpaired_evaluation'
            if a and b and any(a['groups'][g][k] != b['groups'][g][k]
                               for g in GROUPS for k in ('samples', 'positives', 'negatives')):
                status = 'unpaired_group_counts'
            for group in GROUPS:
                for metric in METRICS:
                    delta = a['groups'][group][metric] - b['groups'][group][metric] if status == 'completed' else None
                    row = dict(target=case['target'], dataset=case['dataset'], seed=case['seed'], left=left, right=right,
                               group=group, metric=metric, status=status, delta=delta)
                    paired.append(row)
                    key = tuple(row[k] for k in ('target', 'dataset', 'left', 'right', 'group', 'metric'))
                    pair_buckets.setdefault(key, []).append(row)
    pair_summary = []
    for key, rows in sorted(pair_buckets.items()):
        good = [r for r in rows if r['delta'] is not None]
        pair_summary.append(dict(zip(('target', 'dataset', 'left', 'right', 'group', 'metric'), key)) |
                            dict(seeds=[r['seed'] for r in good], missing_seeds=[r['seed'] for r in rows if r['delta'] is None],
                                 positive=sum(r['delta'] > 0 for r in good), negative=sum(r['delta'] < 0 for r in good),
                                 deltas=[r['delta'] for r in good],
                                 **average([r['delta'] for r in good])))
    complete = sum(c['status'] == 'completed' for c in cases)
    write_csv(output_dir / 'summary.csv', summary, ['target', 'dataset', 'condition', 'group', 'metric', 'n', 'mean', 'sample_sd', 'seeds', 'missing_seeds'])
    write_csv(output_dir / 'paired.csv', pair_summary, ['target', 'dataset', 'left', 'right', 'group', 'metric', 'n', 'mean', 'sample_sd', 'positive', 'negative', 'seeds', 'deltas', 'missing_seeds'])
    lookup = {(r['target'], r['dataset'], r['condition'], r['group'], r['metric']): r for r in summary}
    report = ['# TailProxy实验结果', '', f'{complete}/{len(cases)}条评价完成。AUC/F1为主，TPR@1%为次；所有表保留缺失seed。', '',
              'baseline采用原已接受规则及可复用权重；其余方法使用各自保存的配置，最终在CPU共用固定隐藏用户对及分组。', '',
              '差值为TailProxy−baseline；逐seed差值及汇总见paired.csv。未运行的计划不算完成。', '']
    for target in manifest['targets']:
        for dataset in manifest['datasets']:
            report += [f'## {target}/{dataset}', '', '| 条件 | 分组 | n | AUC | F1 | TPR@1% |', '| --- | --- | ---: | ---: | ---: | ---: |']
            for condition in expected:
                for group in GROUPS:
                    rows = [lookup[(target, dataset, condition, group, metric)] for metric in METRICS]
                    def cell(row):
                        return 'N/A' if not row['n'] else f'{row["mean"]:.4f}' + (f'±{row["sample_sd"]:.4f}' if row['sample_sd'] is not None else '')
                    report.append(f'| {condition} | {group} | {rows[0]["n"]} | ' + ' | '.join(map(cell, rows)) + ' |')
            report.append('')
    (output_dir / 'report.md').write_text('\n'.join(report) + '\n')
    print(f'{complete}/{len(cases)} evaluations complete; {output_dir / "report.md"}', flush=True)
    incomplete = [r for r in cases if r['status'] != 'completed']
    if incomplete:
        with (output_dir / 'report.md').open('a') as handle:
            handle.write('\n未完成条件：\n\n')
            for row in incomplete:
                handle.write(f'- {row["target"]}/{row["dataset"]}/seed-{row["seed"]}/{row["condition"]}: {row.get("reason", row["status"])}\n')
    return complete == len(cases) and all(r['status'] == 'completed' for r in paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, help='Optional directory for analysis of existing results.')
    args = parser.parse_args()
    return 0 if summarize(args.run_dir.resolve(), args.output_dir.resolve() if args.output_dir else None) else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize saved TailProxy cases and paired seed differences; no model imports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_restored_baseline import now, write_json

GROUPS = ('All', 'Head', 'Tail', 'TT', 'isolated_Tail')
METRICS = ('AUROC', 'F1Score', 'TPR@FPR=0.01')


def conditions(recipe):
    return ['baseline'] + [f'e{budget}-{v["name"]}' for budget in recipe['budgets']
                           for v in recipe['variants'] if budget in v['budgets']]


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)


def average(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def read_record(run_dir, case, condition):
    directory = run_dir / case['output'] / condition
    record = json.loads((directory / 'metrics.json').read_text())
    if (record['target'], record['dataset'], record['seed'], record['condition']) != (
            case['target'], case['dataset'], case['seed'], condition):
        raise ValueError('Recorded case identity differs from manifest')
    if not (directory / 'scores.pt').is_file() or not (directory / 'config.yaml').is_file():
        raise ValueError('Missing scores or resolved config')
    for path, expected in ((directory / 'scores.pt', record['scores_sha256']),
                           (directory / 'config.yaml', record['config_file_sha256']),
                           (directory.parent / 'evaluation_pairs.pt', record['evaluation_sha256'])):
        with path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                raise ValueError(f'Artifact SHA256 mismatch: {path}')
    for group in GROUPS:
        for metric in METRICS:
            if not isinstance(record['groups'][group][metric], (int, float)) or not math.isfinite(record['groups'][group][metric]):
                raise ValueError(f'Undefined required metric: {group}/{metric}')
    return record


def summarize(run_dir):
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    recipe = json.loads((run_dir / 'recipe.json').read_text())
    expected = conditions(recipe)
    cases, values, records, recommendation, training = [], [], {}, [], []
    seen_shadow = set()
    for case in manifest['cases']:
        for condition in expected:
            identity = dict(target=case['target'], dataset=case['dataset'], seed=case['seed'], condition=condition)
            status = dict(**identity, status='missing', worker_status=case['status'])
            cases.append(status)
            try:
                record = read_record(run_dir, case, condition)
                status['status'] = 'completed'
                records[(case['target'], case['dataset'], case['seed'], condition)] = record
                history = record.get('training') or {}
                training.append(dict(**identity, stage='attack', branch='', **{
                    k: history.get(k) for k in ('max_epochs', 'actual_epochs', 'best_epoch', 'best_metric', 'stop_reason', 'elapsed_seconds')
                }, metadata_status='recorded' if history else 'metadata_unavailable'))
                shadow = record.get('shadow') or {}
                shadow_key = (case['target'], case['dataset'], case['seed'], shadow.get('budget'))
                if shadow_key not in seen_shadow:
                    seen_shadow.add(shadow_key)
                    for branch, info in (shadow.get('training') or {}).items():
                        training.append(dict(target=case['target'], dataset=case['dataset'], seed=case['seed'],
                            condition=f'e{shadow["budget"]}', stage='shadow', branch=branch,
                            metadata_status='recorded', **{k: info.get(k) for k in (
                                'max_epochs', 'actual_epochs', 'best_epoch', 'best_metric', 'stop_reason', 'elapsed_seconds')}))
                for group in GROUPS:
                    for metric in METRICS:
                        values.append(dict(**identity, group=group, metric=metric, value=record['groups'][group][metric]))
                for stage in ('target', 'shadow'):
                    definitions = record['config'][stage]['test']['metrics']
                    for definition, value in zip(definitions, record['recommendation'][stage], strict=True):
                        metric = definition['_target_'].split('.')[-1].removeprefix('LinkPred') + f'@{definition["k"]}'
                        recommendation.append(dict(**identity, stage=stage, metric=metric, value=value))
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
    for budget in recipe['budgets']:
        native = f'e{budget}-native'
        if native in expected:
            contrasts.update((condition, native) for condition in expected
                             if condition.startswith(f'e{budget}-') and condition != native)
        if f'e{budget}-balanced_tail' in expected and f'e{budget}-balanced' in expected:
            contrasts.add((f'e{budget}-balanced_tail', f'e{budget}-balanced'))
    if 10 in recipe['budgets'] and 300 in recipe['budgets']:
        contrasts.update((f'e300-{v["name"]}', f'e10-{v["name"]}') for v in recipe['variants']
                         if f'e300-{v["name"]}' in expected and f'e10-{v["name"]}' in expected)
    paired, pair_buckets = [], {}
    for case in manifest['cases']:
        for left, right in sorted(contrasts):
            a = records.get((case['target'], case['dataset'], case['seed'], left))
            b = records.get((case['target'], case['dataset'], case['seed'], right))
            status = 'completed' if a and b else 'incomplete'
            if a and b and a['evaluation_sha256'] != b['evaluation_sha256']:
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
                                 **average([r['delta'] for r in good])))
    complete = sum(c['status'] == 'completed' for c in cases)
    write_json(run_dir / 'summary.json', dict(schema=1, updated_at=now(), completed=complete, expected=len(cases),
                                            cases=cases, metrics=summary, paired_metrics=pair_summary))
    write_csv(run_dir / 'metrics.csv', values, ['target', 'dataset', 'seed', 'condition', 'group', 'metric', 'value'])
    write_csv(run_dir / 'recommendation.csv', recommendation, ['target', 'dataset', 'seed', 'condition', 'stage', 'metric', 'value'])
    write_csv(run_dir / 'training.csv', training, ['target', 'dataset', 'seed', 'condition', 'stage', 'branch', 'metadata_status',
              'max_epochs', 'actual_epochs', 'best_epoch', 'best_metric', 'stop_reason', 'elapsed_seconds'])
    write_csv(run_dir / 'summary.csv', summary, ['target', 'dataset', 'condition', 'group', 'metric', 'n', 'mean', 'sample_sd', 'seeds', 'missing_seeds'])
    write_csv(run_dir / 'paired_cases.csv', paired, ['target', 'dataset', 'seed', 'left', 'right', 'group', 'metric', 'status', 'delta'])
    write_csv(run_dir / 'paired.csv', pair_summary, ['target', 'dataset', 'left', 'right', 'group', 'metric', 'n', 'mean', 'sample_sd', 'positive', 'negative', 'seeds', 'missing_seeds'])
    lookup = {(r['target'], r['dataset'], r['condition'], r['group'], r['metric']): r for r in summary}
    report = ['# TailProxy首阶段结果', '', f'{complete}/{len(cases)}条评价完成。AUC/F1为主，TPR@1%为次；所有表保留缺失seed。', '',
              'baseline采用原已接受规则及可复用权重；TailProxy各变体固定同一公开验证候选；全部方法在CPU共用固定隐藏用户对及分组。', '',
              '差值为left−right；训练预算比较与Attack规则比较分别在paired.csv标明。未运行的计划不算完成。', '']
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
    (run_dir / 'report.md').write_text('\n'.join(report) + '\n')
    print(f'{complete}/{len(cases)} evaluations complete; {run_dir / "report.md"}', flush=True)
    return complete == len(cases) and all(r['status'] == 'completed' for r in paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    return 0 if summarize(args.run_dir.resolve()) else 1


if __name__ == '__main__':
    raise SystemExit(main())

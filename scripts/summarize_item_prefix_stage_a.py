#!/usr/bin/env python3
"""Summarize completed Stage-A artifacts without loading or evaluating models."""

import argparse
import csv
import json
from pathlib import Path
import statistics

GROUPS = ('All', 'Head', 'Tail', 'TT', 'isolated_Tail')
METRICS = ('AUROC', 'F1', 'TPR@5%')


def stats(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values) if values else None, max=max(values) if values else None)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = []
    for case in manifest['sources']:
        path = directory / f"{case['target']}/{case['dataset']}/seed-{case['seed']}"
        if not (path / 'metrics.json').exists():
            continue
        row = json.loads((path / 'metrics.json').read_text())
        row['followup'] = json.loads((path / 'followup.json').read_text()) if (path / 'followup.json').exists() else None
        rows.append(row)
    paired, diagnostics, layers = [], [], []
    for row in rows:
        for view in ('train', 'full', 'full_fixed'):
            if view == 'full_fixed':
                if row['followup'] is None:
                    continue
                last = row['followup']['full_query_fixed_paths']
                item = row['views']['full']['item']
            else:
                last = row['views'][view]['levels'][-1]
                item = row['views'][view]['item']
            d = dict(target=row['target'], dataset=row['dataset'], seed=row['seed'], view=view,
                     queried_items=row['queried_items'], training_items=row['training_items'],
                     prefixes=last['prefixes'], singleton_ratio=last['unique_item_ratio'],
                     distinct_path_ratio=last['distinct_path_ratio'], collision_pairs=last['collision_pairs'],
                     frequency_equal_ratio=last['frequency']['equal_ratio'],
                     overlap_normalized_mae=last['equivalence']['overlap']['normalized_mae'],
                     overlap_max_error=last['equivalence']['overlap']['max_error'],
                     overlap_close_ratio=last['equivalence']['overlap']['close_ratio'],
                     overlap_pearson=last['equivalence']['overlap']['pearson'],
                     overlap_spearman=last['equivalence']['overlap']['spearman'],
                     jaccard_normalized_mae=last['equivalence']['jaccard']['normalized_mae'],
                     jaccard_max_error=last['equivalence']['jaccard']['max_error'],
                     user_prefix_edges=last['user_prefix_edges'],
                     full_path_change=row['view_shift']['per_prefix_path_change'][-1])
            if row['followup'] is not None and view in ('full', 'full_fixed'):
                detail = row['followup']['collision_analysis']['train' if view == 'full_fixed' else 'full']
                d['same_user_set_collision_pair_ratio'] = detail['same_user_set_pair_ratio']
                d['exactly_identical_embedding_pairs'] = detail['exactly_identical_embedding_pairs']
            diagnostics.append(d)
            for kind in ('overlap', 'jaccard'):
                for group in GROUPS:
                    reference, candidate = item[kind]['groups'][group], last['matching'][kind]['groups'][group]
                    entry = dict(target=row['target'], dataset=row['dataset'], seed=row['seed'], view=view, score=kind, group=group)
                    for metric in METRICS:
                        entry[f'item_{metric}'] = reference[metric]
                        entry[f'prefix_{metric}'] = candidate[metric]
                        entry[f'delta_{metric}'] = candidate[metric] - reference[metric]
                    paired.append(entry)
        for level in row['views']['full']['levels']:
            layers.append(dict(target=row['target'], dataset=row['dataset'], seed=row['seed'], level=level['level'],
                               singleton_ratio=level['unique_item_ratio'], collision_pairs=level['collision_pairs'],
                               prefix_count=level['prefixes'], user_prefix_edges=level['user_prefix_edges'],
                               overlap_pearson=level['equivalence']['overlap']['pearson'],
                               overlap_nmae=level['equivalence']['overlap']['normalized_mae'],
                               all_auc=level['matching']['overlap']['groups']['All']['AUROC'],
                               tt_auc=level['matching']['overlap']['groups']['TT']['AUROC']))
    def aggregate(records, keys):
        groups = {}
        for record in records:
            key = tuple(record[k] for k in keys)
            groups.setdefault(key, []).append(record)
        results = []
        for key, group in groups.items():
            fields = set.union(*(set(r) for r in group)) - set(keys) - {'seed'}
            results.append(dict(zip(keys, key)) | dict(seeds=[r['seed'] for r in group],
                values={field: stats([r.get(field) for r in group]) for field in sorted(fields)}))
        return results
    diag_summary = aggregate(diagnostics, ['target', 'dataset', 'view'])
    pair_summary = aggregate(paired, ['target', 'dataset', 'view', 'score', 'group'])
    layer_summary = aggregate(layers, ['target', 'dataset', 'level'])
    payload = dict(completed=len(rows), planned=len(manifest['sources']), followups=sum(r['followup'] is not None for r in rows),
                   diagnostics=diag_summary, paired=pair_summary, layers=layer_summary,
                   strict_full_cases=sum(r['views']['full']['assessment']['strict_equivalence'] for r in rows),
                   approximate_full_cases=sum(r['views']['full']['assessment']['approximate_representation'] for r in rows),
                   auc_close_full_cases=sum(r['views']['full']['assessment']['auc_difference_within_005'] for r in rows),
                   experiment_seconds=sum(r['elapsed_seconds'] for r in rows),
                   rvq_fit_seconds=sum(r['rvq']['seconds'] for r in rows))
    write_json(directory / 'analysis.json', payload)
    for filename, records in [('diagnostics.csv', diagnostics), ('paired.csv', paired), ('layers.csv', layers)]:
        if records:
            fields = list(dict.fromkeys(k for r in records for k in r))
            with (directory / filename).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
    report = ['# Stage A: item RVQ / prefix equivalence', '',
              f"Completed {len(rows)}/{len(manifest['sources'])}; fixed-path follow-ups {payload['followups']}/{len(manifest['sources'])}.", '',
              'Frozen e300-v2 behavior encoders; CPU residual k-means, 8×256, one initialization. No new target/Shadow/Attack training.',
              'Full-query results use re-encoded full-view embeddings. full_fixed uses training-view item paths with full-query sets/frequencies.',
              'Thresholds use public validation All. Hidden groups are descriptive only. All seeds retained.', '',
              f"Strict full-view equivalence: {payload['strict_full_cases']}/{len(rows)}; approximate representation: {payload['approximate_full_cases']}/{len(rows)}; all-group AUROC difference ≤.005: {payload['auc_close_full_cases']}/{len(rows)}.", '',
              '## Complete-prefix diagnostics (five-seed means)', '',
              '| Target | Data | View | Singleton ratio | Collision pairs | Frequency equal | Overlap NMAE | Jaccard NMAE | Path drift |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in diag_summary:
        if row['view'] == 'train':
            continue
        v = row['values']
        report.append(f"| {row['target']} | {row['dataset']} | {row['view']} | " + ' | '.join(
            f"{v[k]['mean']:.6g}" for k in ['singleton_ratio', 'collision_pairs', 'frequency_equal_ratio', 'overlap_normalized_mae',
                                           'jaccard_normalized_mae', 'full_path_change']) + ' |')
    report += ['', '## Paired relation results', '',
               'Each delta is prefix minus original item statistic, in absolute 0–1 units. Mean ± sample SD across seeds; no significance claim.', '',
               '| Target | Data | View | Score | Group | Item AUC | Prefix AUC | ΔAUC | ΔF1 | ΔTPR@5% |',
               '| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in pair_summary:
        if row['view'] == 'train':
            continue
        v = row['values']
        cells = []
        for metric in ['item_AUROC', 'prefix_AUROC', 'delta_AUROC', 'delta_F1', 'delta_TPR@5%']:
            value = v[metric]
            cells.append(f"{value['mean']:.6f} ± {value['sample_sd']:.6f}" if value['sample_sd'] is not None else f"{value['mean']:.6f} (n=1)")
        report.append(f"| {row['target']} | {row['dataset']} | {row['view']} | {row['score']} | {row['group']} | " + ' | '.join(cells) + ' |')
    report += ['', 'Original item statistics and all prefix levels/groups are in metrics.csv; per-seed paired differences in paired.csv.',
               'Collision causes, singleton/collided score decomposition and fixed-path results are in each followup.json.',
               'No software tests were run. These are authorized research fits and relation evaluations.', '']
    (directory / 'report.md').write_text('\n'.join(report))
    print(json.dumps({k: v for k, v in payload.items() if k not in ('diagnostics', 'paired', 'layers')}, indent=2))


if __name__ == '__main__':
    main()

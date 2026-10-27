#!/usr/bin/env python3
"""Measure restored DESIGN query caches and compare historical per-seed statistics.

Only existing ShadowData caches are read. No model loading, inference, training,
negative sampling or cache regeneration is performed.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

METRICS = (
    'query_unique', 'entropy', 'effective_items', 'top30_share', 'top100_share',
    'top300_share', 'top1_user_coverage', 'random_pair_mean_intersection',
    'unique_lists', 'uniform999_expected_ever_queried_negatives',
)
UNITS = {name: 'count' for name in METRICS}
UNITS.update({name: 'fraction' for name in METRICS if name.endswith('_share')})
UNITS.update(entropy='nats', top1_user_coverage='fraction',
             random_pair_mean_intersection='items per distinct user pair (not divided by 30)')
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
CACHE_PATH = re.compile(r'(?:Load cached shadow data|Generate shadow data)[^\n]*?"([^"\n]+\.pkl)"')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def stats(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}


class CPUUnpickler(pickle.Unpickler):
    """Remap tensor storage from repository-owned pickle caches to CPU."""

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda data: torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def measure(data, expected_k):
    users, items = int(data.num_users), int(data.num_items)
    if users < 2 or items <= expected_k - 1:
        raise ValueError('Need at least two users and a catalogue larger than the exclusion set.')
    edges = data.behavioral_edge_index.detach().cpu()
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError('Expected a 2 x E int64 behavioral edge index.')
    if edges.shape[1] != users * expected_k:
        raise ValueError(f'Expected {expected_k} recommendations per user; wrong total edge count.')
    if ((edges[0] < 0) | (edges[0] >= users) |
            (edges[1] < users) | (edges[1] >= users + items)).any():
        raise ValueError('Query user/item IDs are outside the declared global ranges.')
    keys = edges[0] * items + edges[1] - users
    keys = keys.sort().values
    if (keys[1:] == keys[:-1]).any():
        raise ValueError('Repeated item in a user query list.')
    if not torch.equal(torch.div(keys, items, rounding_mode='floor').reshape(users, expected_k),
                       torch.arange(users)[:, None].expand(users, expected_k)):
        raise ValueError('Not every user has exactly the configured number of recommendations.')
    lists = (keys % items).reshape(users, expected_k)
    counts = torch.bincount(lists.flatten(), minlength=items).double()
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-(probabilities * probabilities.log()).sum())
    ranked = counts.sort(descending=True).values
    covered = int((counts > 0).sum())
    # Same legacy candidate rule: exclude K-2 training items + one validation item.
    excluded = expected_k - 1
    return dict(users=users, items=items, query_k=expected_k, view='full-query',
                query_unique=covered, entropy=entropy, effective_items=math.exp(entropy),
                top30_share=float(ranked[:30].sum() / counts.sum()),
                top100_share=float(ranked[:100].sum() / counts.sum()),
                top300_share=float(ranked[:300].sum() / counts.sum()),
                top1_user_coverage=float(ranked[0] / users),
                random_pair_mean_intersection=float((counts * (counts - 1)).sum() / (users * (users - 1))),
                unique_lists=int(torch.unique(lists, dim=0).shape[0]),
                uniform999_expected_ever_queried_negatives=999 * (covered - excluded) / (items - excluded))


def candidate_from_run(config_path, project_root, datasets, seeds, revision, expected_k):
    config = OmegaConf.load(config_path)
    if config.target.model.name != 'DESIGN' or config.target.model.get('revision') != revision:
        return None
    dataset, seed = str(config.target.data.name), int(config.seed)
    if dataset not in datasets or seed not in seeds:
        return None
    run = config_path.parent.parent
    shadow_config = OmegaConf.to_container(config.shadow.data, resolve=True)
    if shadow_config['num_recommendations'] != expected_k:
        raise ValueError(f'{run}: unexpected query K={shadow_config["num_recommendations"]}')
    text = '\n'.join(ANSI.sub('', path.read_text()) for path in sorted(run.glob('*.log')))
    caches = set(CACHE_PATH.findall(text))
    if len(caches) != 1:
        raise ValueError(f'{run}: expected exactly one logged shadow-data cache, found {sorted(caches)}')
    path = Path(next(iter(caches)))
    path = (project_root / path).resolve() if not path.is_absolute() else path.resolve()
    representations = set(re.findall(r'Model representation: (DESIGN\([^\n]+\))', text))
    if len(representations) != 1:
        raise ValueError(f'{run}: expected exactly one logged DESIGN model representation')
    identity = dict(target=OmegaConf.to_container(config.target, resolve=True),
                    shadow_data=shadow_config, seed=seed, deterministic=bool(config.deterministic))
    return dict(case=f'{dataset}/DESIGN', dataset=dataset, seed=seed, run=str(run.resolve()),
                config_path=str(config_path.resolve()), config_sha256=digest(config_path),
                cache=str(path), target_model_repr=next(iter(representations)),
                target_revision=revision, config=identity)


def collect(args):
    candidates, issues = [], []
    configs = set()
    for root in args.run_roots:
        if not root.is_dir():
            raise ValueError(f'Run root does not exist: {root}')
        configs.update(root.rglob('.hydra/config.yaml'))
    for path in sorted(configs):
        try:
            row = candidate_from_run(path, args.project_root, args.datasets, args.seeds,
                                     args.target_revision, args.query_k)
            if row:
                candidates.append(row)
        except (KeyError, AttributeError, TypeError, ValueError, OSError, OmegaConfBaseException) as error:
            issues.append({'config_path': str(path), 'error': str(error)})
    rows, coverage, skipped_sources = [], [], []
    for dataset in args.datasets:
        for seed in args.seeds:
            case = f'{dataset}/DESIGN'
            matches = [row for row in candidates if row['case'] == case and row['seed'] == seed]
            status = dict(case=case, seed=seed, status='missing', candidates=len(matches))
            available = []
            unavailable = []
            for row in matches:
                if Path(row['cache']).is_file():
                    available.append(row)
                else:
                    unavailable.append(row)
                    skipped_sources.append(dict(case=case, seed=seed, run=row['run'],
                                                config_path=row['config_path'], cache=row['cache'],
                                                reason='Logged cache file is missing; source was skipped.'))
            status.update(available_candidates=len(available), skipped_candidates=len(unavailable),
                          unavailable_caches=sorted({row['cache'] for row in unavailable}))
            try:
                if not matches:
                    status['reason'] = 'No matching current DESIGN run with a usable config/log reference.'
                    continue
                if not available:
                    status['reason'] = ('All logged cache files are missing; restore a referenced cache '
                                        'or supply --run-roots with existing query caches. No regeneration was attempted.')
                    continue
                # Historical logs survive cache resets. Only extant sources can
                # supply data; missing files must not veto a valid current cache.
                matches = available
                if len({canonical(row['config']) for row in matches}) != 1:
                    raise ValueError('Conflicting target/query configurations; narrow --run-roots.')
                cache_paths = sorted({Path(row['cache']) for row in matches})
                hashes = {str(path): digest(path) for path in cache_paths}
                if len(set(hashes.values())) != 1:
                    raise ValueError('Distinct cache contents for the same case; narrow --run-roots.')
                with cache_paths[0].open('rb') as handle:
                    data = CPUUnpickler(handle).load()
                if type(data).__name__ != 'ShadowData' or data.name != dataset:
                    raise ValueError('Cache is not the expected dataset ShadowData.')
                if data.target_model_name != 'DESIGN' or data.num_recommendations != args.query_k:
                    raise ValueError('Cache target/query metadata does not match the run config.')
                if any(row['target_model_repr'] != data.target_model_repr for row in matches):
                    raise ValueError('Cache DESIGN fingerprint/representation differs from the run log.')
                reference = matches[0]
                for key in ('shadow_social_ratio', 'attack_val_ratio'):
                    if getattr(data, key) != reference['config']['shadow_data'][key]:
                        raise ValueError(f'Cache {key} differs from the run config.')
                result = measure(data, args.query_k)
                rows.append(dict(case=case, dataset=dataset, seed=seed, **result,
                                 target_revision=args.target_revision, sources=matches,
                                 cache_sha256=hashes, target_data_repr=data.target_data_repr,
                                 target_model_repr=data.target_model_repr))
                status['status'] = 'complete'
            except (OSError, ValueError, TypeError, AttributeError, RuntimeError,
                    pickle.UnpicklingError, EOFError) as error:
                status.update(status='error', reason=str(error))
            finally:
                coverage.append(status)
    return rows, coverage, issues, skipped_sources


def load_old(path):
    payload = json.loads(path.read_text())
    rows = payload['rows'] if isinstance(payload, dict) else payload
    indexed = {}
    for row in rows:
        key = (row['case'], int(row['seed']))
        if key in indexed:
            raise ValueError(f'Duplicate historical per-seed row: {key}')
        indexed[key] = row
    return indexed


def comparisons(rows, old):
    result = []
    for row in rows:
        previous = old.get((row['case'], row['seed']))
        for metric in METRICS:
            record = dict(case=row['case'], seed=row['seed'], metric=metric, unit=UNITS[metric],
                          current=row[metric], old=None, delta=None, status='missing_old')
            if previous is not None:
                record['old'] = previous.get(metric)
                # Historical tables use full Top-30. They contain no per-seed query tensors now.
                if previous.get('users') != row['users'] or previous.get('items') != row['items']:
                    record['status'] = 'incompatible_population'
                elif previous.get('query_k', 30) != row['query_k']:
                    record['status'] = 'incompatible_query_k'
                elif record['old'] is not None:
                    record.update(delta=record['current'] - record['old'], status='paired')
            result.append(record)
    return result


def summaries(rows, comparison):
    result = []
    for case in sorted({row['case'] for row in rows}):
        current = [row for row in rows if row['case'] == case]
        for metric in METRICS:
            paired = [row for row in comparison
                      if row['case'] == case and row['metric'] == metric and row['status'] == 'paired']
            result.append(dict(case=case, metric=metric, unit=UNITS[metric],
                               current_seeds=[row['seed'] for row in current],
                               current_all=stats([row[metric] for row in current]),
                               paired_seeds=[row['seed'] for row in paired],
                               paired_current=stats([row['current'] for row in paired]),
                               paired_old=stats([row['old'] for row in paired]),
                               paired_delta=stats([row['delta'] for row in paired])))
    return result


def write_csv(path, rows, columns):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT, help='Base for logged relative cache paths.')
    parser.add_argument('--run-roots', type=Path, nargs='+', default=[ROOT / 'runs/restored-baseline'])
    parser.add_argument('--datasets', nargs='+', choices=['Ciao', 'Flickr'], default=['Ciao', 'Flickr'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 47)))
    parser.add_argument('--target-revision', default='legacy-restored-v1')
    parser.add_argument('--query-k', type=int, default=30)
    parser.add_argument('--old', type=Path, default=ROOT / 'docs/archive/tail-proxy/evidence/mechanism-review-2026-10-23/concentration.json')
    parser.add_argument('--output', type=Path, required=True, help='A new output directory; existing directories are refused.')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.query_k < 3:
        parser.error('--threads must be positive; --query-k must be at least 3.')
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate datasets/seeds are not allowed.')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    torch.set_num_threads(args.threads)
    old = load_old(args.old)
    rows, coverage, issues, skipped_sources = collect(args)
    comparison = comparisons(rows, old)
    summary = summaries(rows, comparison)
    payload = dict(schema_version=2, created_at=datetime.now(timezone.utc).isoformat(),
                   scope='Existing full-query ShadowData tensors only; no model inference/training/evaluation.',
                   units=UNITS, old_source=str(args.old.resolve()), old_sha256=digest(args.old),
                   historical_query_k_assumption=30,
                   numerical_note='Current counts/entropy use float64; historical entropy-derived values used float32.',
                   request=dict(datasets=args.datasets, seeds=args.seeds, target_revision=args.target_revision,
                                query_k=args.query_k, run_roots=[str(p.resolve()) for p in args.run_roots]),
                   rows=rows, coverage=coverage, candidate_issues=issues, skipped_sources=skipped_sources,
                   summary=summary, comparison=comparison)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'statistics.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    write_csv(args.output / 'current.csv', rows, ['case', 'seed', 'users', 'items', 'query_k', *METRICS])
    write_csv(args.output / 'comparison.csv', comparison,
              ['case', 'seed', 'metric', 'unit', 'current', 'old', 'delta', 'status'])
    def number(value):
        return 'N/A' if value is None else f'{value:.6g}'
    lines = ['# DESIGN查询分布比较', '',
             f'仅统计已有全{args.query_k}查询缓存；缺失或冲突见statistics.json的coverage/candidate_issues。', '',
             f'跳过{len(skipped_sources)}条已失效的历史缓存引用，详情见skipped_sources；仅核对和使用现存来源。', '',
             '占比以0–1表示；平均交集为物品数，未除以30。新旧差值只在共同seed计算；N/A表示无可比旧值。', '',
             '| case | metric | 当前n | 当前均值 | 配对seed | 配对当前均值 | 配对旧均值 | 配对差值均值 |',
             '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for row in summary:
        lines.append(f'| {row["case"]} | {row["metric"]} | {row["current_all"]["n"]} | '
                     f'{number(row["current_all"]["mean"])} | {row["paired_seeds"]} | '
                     f'{number(row["paired_current"]["mean"])} | {number(row["paired_old"]["mean"])} | '
                     f'{number(row["paired_delta"]["mean"])} |')
    lines += ['', '逐seed值与差值见current.csv/comparison.csv；样本SD、配置、缓存SHA256和模型表示见statistics.json。', '',
              '覆盖情况：', '', *[f'- {row["case"]} seed{row["seed"]}: {row["status"]}'
                                   + (f' — {row["reason"]}' if 'reason' in row else '') for row in coverage]]
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    complete = sum(row['status'] == 'complete' for row in coverage)
    print(f'{complete}/{len(coverage)} cases measured; {len(skipped_sources)} stale sources skipped; '
          f'{len(issues)} candidate issues; output: {args.output}')
    for row in coverage:
        if row['status'] != 'complete':
            print(f'{row["case"]} seed{row["seed"]}: {row["status"]}: {row["reason"]}', file=sys.stderr)
    return 0 if complete == len(coverage) else 1


if __name__ == '__main__':
    raise SystemExit(main())

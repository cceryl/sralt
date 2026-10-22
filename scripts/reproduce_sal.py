#!/usr/bin/env python3
"""Run paired DESIGN/SAL experiments with fresh black-box SMIA attacks.

Use the aux-server Apptainer environment. The minimal phase runs Ciao/seed42,
scan uses Ciao/Flickr/seed42, and formal uses all four datasets and seeds42–46.
Model and metric caches make interrupted runs resumable without overwriting
earlier experiments. Selection remains the existing validation-only selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import itertools
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import comparison_report as report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['minimal', 'scan', 'formal'], default='minimal')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--weights', nargs='+', type=float)
    parser.add_argument('--output', type=Path, default=Path('runs/sal'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--refresh-metrics', action='store_true')
    return parser.parse_args(argv)


def cases(args):
    datasets = args.datasets or (['ciao'] if args.phase == 'minimal' else
                                 ['ciao', 'flickr'] if args.phase == 'scan' else report.DATASETS)
    seeds = args.seeds or ([42] if args.phase != 'formal' else report.SEEDS)
    weights = args.weights or ([.01, .02, .05, .1, .2, .3] if args.phase == 'scan' else [.01])
    for dataset, seed in itertools.product(dict.fromkeys(datasets), dict.fromkeys(seeds)):
        yield dataset, seed, None
        for weight in dict.fromkeys(weights):
            yield dataset, seed, weight


def paired_summary(rows):
    """Average per-seed relative losses over exact matched baseline/defense pairs."""
    baselines = {(row['dataset'], row['seed']): row for row in rows if row['defense'] == 'None'}
    buckets = {}
    for row in rows:
        if row['defense'] == 'None':
            continue
        baseline = baselines.get((row['dataset'], row['seed']))
        if baseline is None:
            continue
        for stage in ('target', 'attack'):
            for group, metrics in (row['results'][stage] or {}).items():
                for metric, value in metrics.items():
                    base = (baseline['results'][stage] or {}).get(group, {}).get(metric)
                    if base is None or value is None or base == 0:
                        continue
                    key = (row['dataset'], row['defense'], stage, group, metric)
                    buckets.setdefault(key, []).append(dict(seed=row['seed'], baseline=base,
                                                            defended=value, relative_loss=(base - value) / base))
    result = []
    for (dataset, defense, stage, group, metric), pairs in buckets.items():
        values = [pair['relative_loss'] for pair in pairs]
        result.append(dict(dataset=dataset, defense=defense, stage=stage, group=group, metric=metric,
                           pairs=pairs, n=len(values), relative_loss_mean=statistics.mean(values),
                           relative_loss_std=statistics.stdev(values) if len(values) > 1 else None))
    return result


def run_case(dataset, seed, weight, sources, refresh_metrics):
    import torch
    from hydra.utils import instantiate

    target = 'design' if weight is None else 'design_sal'
    config = report.compose_case(dataset, target, 'smia', 'mlp', seed, weight)
    defense = 'None' if weight is None else f'SAL λ={weight:g}'
    identities = {stage: report.cache_identity(config, stage, sources) for stage in ('target', 'shadow', 'attack')}
    records = {stage: json.loads(report.cache_path(identity).read_text())
               if report.cache_path(identity).is_file() and not refresh_metrics else None
               for stage, identity in identities.items()}
    status = {stage: 'cached' if record else 'missing' for stage, record in records.items()}
    started = time.perf_counter()
    device = instantiate(config.device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    if any(record is None for record in records.values()):
        report.evaluate_missing(config, identities, records, allow_training=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    row = dict(dataset=dataset, target='design', target_config=target, defense=defense,
               shadow='smia', attack='mlp', seed=seed, tail_ratio=float(config.attack.tail_ratio),
               status={stage: 'computed' if state == 'missing' else state for stage, state in status.items()},
               results={stage: record['metrics'] for stage, record in records.items()},
               evaluated_at={stage: record['created_at'] for stage, record in records.items()},
               elapsed_seconds=time.perf_counter() - started,
               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
               peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
               device=str(device), missing_reason=None)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return row


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    planned = list(cases(args))
    if args.dry_run:
        print(json.dumps([dict(dataset=d, seed=s, loss_weight=w) for d, s, w in planned], indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[
        logging.StreamHandler(), logging.FileHandler(args.output / f'{args.phase}.log')
    ])
    sources = report.source_digest()
    payload = dict(rows=[], stages=['target', 'shadow', 'attack'], generated_at='', sources=sources,
                   namespace=os.environ.get('SRALT_CACHE_VERSION', 'default'), allow_training=True,
                   attack_metrics=['AUROC', 'F1Score', 'TPR@FPR=1%'], phase=args.phase,
                   planned_cases=len(planned), completed_cases=0)
    for index, (dataset, seed, weight) in enumerate(planned, 1):
        logging.info('[%d/%d] %s seed=%d loss_weight=%s', index, len(planned), dataset, seed, weight)
        payload['rows'].append(run_case(dataset, seed, weight, sources, args.refresh_metrics))
        payload['completed_cases'] = index
        payload['generated_at'] = datetime.now().astimezone().isoformat()
        payload['paired_summary'] = paired_summary(payload['rows'])
        path = args.output / f'{args.phase}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
        temporary.replace(path)
        report.write_report(payload, args.output / f'{args.phase}.html')
    logging.info('Completed %d paired experiment cases: %s', len(planned), args.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Launch DESIGN/GDMSR baseline and optional SAL sweeps through SMIA/MLP.

Uses only the standard library; --summarize-only never imports model code.
Paths passed on the command line are relative to the repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize(run_dir):
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    buckets = {}
    cases = []
    records = {}
    target = manifest.get('target', 'design')
    for case in manifest['cases']:
        case_target = case.get('target', target)
        path = run_dir / case['output'] / 'metrics.json'
        status = case['status']
        if status == 'completed' and not path.is_file():
            status = 'missing_metrics'
        weight = case.get('sal_weight')
        condition = case.get('condition', 'baseline')
        case_summary = {key: case[key] for key in ('dataset', 'seed', 'output')} | {
            'target': case_target, 'status': status, 'condition': condition, 'sal_weight': weight,
            'sal_checkpoint': None,
        }
        cases.append(case_summary)
        if status != 'completed':
            continue
        record = json.loads(path.read_text())
        if (record['dataset'], record['seed']) != (case['dataset'], case['seed']):
            raise ValueError(f'Metric identity does not match manifest: {path}')
        if record['models']['target'].lower() != case.get('target_config', case_target):
            raise ValueError(f'Target model does not match manifest: {path}')
        if weight is not None and record.get('target_defense') != {'method': 'sal', 'loss_weight': weight}:
            raise ValueError(f'SAL weight does not match manifest: {path}')
        console = run_dir / case['output'] / 'console.log'
        if weight is not None and console.is_file():
            match = re.search(
                r'SAL checkpoint: best_epoch=(\d+), first_surrogate_epoch=(\d+), surrogate_fit_epochs=(\d+)',
                console.read_text()
            )
            if match:
                case_summary['sal_checkpoint'] = dict(zip(
                    ('best_epoch', 'first_surrogate_epoch', 'surrogate_fit_epochs'), map(int, match.groups())
                ))
        records[(case_target, case['dataset'], condition, case['seed'])] = record
        for stage, groups in record['metrics'].items():
            for group, metrics in groups.items():
                for metric, value in metrics.items():
                    key = (case_target, case['dataset'], condition, stage, group, metric)
                    bucket = buckets.setdefault(key, {'values': [], 'seeds': [], 'missing_seeds': []})
                    if value is None or not math.isfinite(value):
                        bucket['missing_seeds'].append(case['seed'])
                    else:
                        bucket['values'].append(value)
                        bucket['seeds'].append(case['seed'])
    rows = []
    for key, bucket in sorted(buckets.items()):
        values = bucket['values']
        rows.append(dict(zip(('target', 'dataset', 'condition', 'stage', 'group', 'metric'), key)) | {
            'sal_weight': next(case.get('sal_weight') for case in manifest['cases']
                               if case.get('target', target) == key[0] and case['dataset'] == key[1]
                               and case.get('condition', 'baseline') == key[2]),
            'mean': statistics.mean(values) if values else None,
            'sample_std': statistics.stdev(values) if len(values) > 1 else None,
            'n': len(values), 'seeds': bucket['seeds'], 'missing_seeds': bucket['missing_seeds'],
        })
    summary = {'schema': 2, 'target': target, 'updated_at': now(), 'cases': cases, 'metrics': rows}
    write_json(run_dir / 'summary.json', summary)
    fields = ['target', 'dataset', 'condition', 'sal_weight', 'stage', 'group', 'metric',
              'mean', 'sample_std', 'n', 'seeds', 'missing_seeds']
    with (run_dir / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row | {key: json.dumps(row[key]) for key in ('seeds', 'missing_seeds')})
    if manifest.get('sal_weights'):
        summarize_pairs(run_dir, manifest, records)
    completed = sum(case['status'] == 'completed' for case in cases)
    print(f'{completed}/{len(cases)} completed; summary: {run_dir / "summary.csv"}', flush=True)
    return completed == len(cases)


def summarize_pairs(run_dir, manifest, records):
    """Calculate decreases within each seed before averaging; never select lambda."""
    pairs, buckets = [], {}
    for case in manifest['cases']:
        weight = case.get('sal_weight')
        if weight is None:
            continue
        dataset, seed, condition = case['dataset'], case['seed'], case['condition']
        target = case.get('target', manifest.get('target', 'design'))
        baseline = records.get((target, dataset, 'baseline', seed))
        defended = records.get((target, dataset, condition, seed))
        pair = {'target': target, 'dataset': dataset, 'seed': seed, 'sal_weight': weight,
                'condition': condition, 'status': 'completed' if baseline and defended else 'incomplete',
                'metrics': []}
        pairs.append(pair)
        if pair['status'] != 'completed':
            continue
        for stage in ('target', 'attack'):
            for group, metrics in baseline['metrics'][stage].items():
                for metric, original in metrics.items():
                    value = defended['metrics'][stage][group][metric]
                    finite = original is not None and value is not None and math.isfinite(original) and math.isfinite(value)
                    decrease = original - value if finite else None
                    relative = 100 * decrease / original if finite and original != 0 else None
                    pair['metrics'].append({
                        'stage': stage, 'group': group, 'metric': metric,
                        'baseline': original, 'sal': value, 'decrease': decrease,
                        'relative_decrease_percent': relative,
                    })
                    key = (target, dataset, condition, weight, stage, group, metric)
                    bucket = buckets.setdefault(key, [])
                    bucket.append({'seed': seed, 'baseline': original, 'sal': value,
                                   'decrease': decrease, 'relative_decrease_percent': relative})
    rows = []
    for key, values in sorted(buckets.items()):
        row = dict(zip(('target', 'dataset', 'condition', 'sal_weight', 'stage', 'group', 'metric'), key))
        row['paired_seeds'] = [value['seed'] for value in values]
        row['n'] = len(values)
        for name in ('baseline', 'sal', 'decrease', 'relative_decrease_percent'):
            available = [value[name] for value in values if value[name] is not None and math.isfinite(value[name])]
            row[f'{name}_mean'] = statistics.mean(available) if available else None
            row[f'{name}_sample_std'] = statistics.stdev(available) if len(available) > 1 else None
            row[f'{name}_n'] = len(available)
        rows.append(row)
    write_json(run_dir / 'paired.json', {'schema': 1, 'target': manifest['target'], 'pairs': pairs, 'metrics': rows})
    fields = ['target', 'dataset', 'condition', 'sal_weight', 'stage', 'group', 'metric', 'n', 'paired_seeds']
    fields += [f'{name}_{stat}' for name in ('baseline', 'sal', 'decrease', 'relative_decrease_percent')
               for stat in ('mean', 'sample_std', 'n')]
    with (run_dir / 'paired.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row | {'paired_seeds': json.dumps(row['paired_seeds'])})


def snapshot(run_dir):
    destination = run_dir / 'source'
    for folder in ('sralt', 'scripts'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.suffix in {'.py', '.yaml', '.sh', '.json'}:
                target = destination / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for name in ('pyproject.toml', 'uv.lock'):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, destination / name)
    for name, command in (
        ('git-head.txt', ['git', 'rev-parse', 'HEAD']),
        ('git-status.txt', ['git', 'status', '--short']),
        ('git-diff.patch', ['git', 'diff', 'HEAD', '--', 'sralt', 'scripts', 'pyproject.toml', 'uv.lock']),
    ):
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        (destination / name).write_text(result.stdout)


def run_batch(args, run_dir):
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot(run_dir)
    env = os.environ.copy()
    env.pop('SRALT_CACHE_VERSION', None)
    env.pop('SRALT_CACHE_REFRESH', None)
    env['PYTHONUNBUFFERED'] = '1'
    manifest = {
        'schema': 2, 'created_at': now(), 'python': sys.executable,
        'target': args.targets[0] if len(args.targets) == 1 else 'mixed', 'targets': args.targets,
        'sal_weights': args.sal_weights,
        'cache_namespace': 'default', 'cache_refresh': False, 'jobs': args.jobs,
        'overrides': args.override, 'sal_overrides': args.sal_override,
        'sal_variants': args.variants, 'diagnostics': args.diagnostics,
        'status': 'running', 'cases': [],
    }
    for target in args.targets:
        conditions = [('baseline', target, None, [])] + [
            (f'{variant["name"]}-lambda-{weight}', f'{target}_sal', weight, variant['overrides'])
            for index, variant in enumerate(args.variants) for weight in args.sal_weights
            if weight != 0 or index == 0
        ]
        for dataset in args.datasets:
            for condition, target_config, weight, variant_overrides in conditions:
                for seed in args.seeds:
                    output = f'{dataset}/{condition}/seed-{seed}' if args.sal_weights else f'{dataset}/seed-{seed}'
                    if len(args.targets) > 1:
                        output = f'{target}/{output}'
                    command = [
                        sys.executable, str(ROOT / 'scripts/pipeline_low_memory.py'),
                        f'target/data={dataset}', f'target/model={target_config}', 'shadow/model=smia',
                        'attack/model=mlp', f'seed={seed}', *args.override,
                    ]
                    if weight is not None:
                        command += args.sal_override + variant_overrides
                        command.append(f'target.model.attribute.defense.loss_weight={weight}')
                        if args.diagnostics and weight != 0:
                            command += ['target.model.attribute.defense.diagnostics.enabled=true',
                                        f'target.model.attribute.defense.diagnostics.run_id=diag-{re.sub(r"[^0-9]", "", manifest["created_at"])}']
                    command += [f'hydra.run.dir={run_dir / output}', 'hydra.job.chdir=False']
                    manifest['cases'].append({
                        'target': target, 'target_config': target_config, 'sal_weight': weight,
                        'condition': condition, 'dataset': dataset, 'seed': seed, 'output': output,
                        'command': command, 'status': 'pending',
                    })
    manifest_path = run_dir / 'manifest.json'
    if args.plan_only:
        manifest['status'] = 'planned'
        write_json(manifest_path, manifest)
        print(f'Plan only: {len(manifest["cases"])} cases; {manifest_path}')
        return 0
    write_json(manifest_path, manifest)
    print(f'Run directory: {run_dir}', flush=True)
    active = []
    next_start = 0.0
    try:
        # Global task queue across all datasets, conditions, and seeds.
        pending = list(manifest['cases'])

        while pending or active:
            # Start new tasks up to the global concurrency limit.
            # Preserve the 10-second interval between task launches.
            while (
                pending
                and len(active) < args.jobs
                and time.monotonic() >= next_start
            ):
                case = pending.pop(0)
                output = run_dir / case['output']
                output.mkdir(parents=True)

                with (output / 'console.log').open('w') as log:
                    process = subprocess.Popen(
                        case['command'],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )

                active.append((process, case))
                next_start = time.monotonic() + 10.0

                case.update(
                    status='running',
                    pid=process.pid,
                    started_at=now(),
                )
                write_json(manifest_path, manifest)

                print(
                    f'Started {case["dataset"]}/{case["condition"]}/{case["seed"]}: '
                    f'PID {process.pid}',
                    flush=True,
                )

            # Check all active tasks for completion.
            for process, case in active[:]:
                returncode = process.poll()
                if returncode is None:
                    continue

                has_metrics = (
                    run_dir / case['output'] / 'metrics.json'
                ).is_file()

                case.update(
                    returncode=returncode,
                    finished_at=now(),
                    status=(
                        'completed'
                        if returncode == 0 and has_metrics
                        else 'failed'
                    ),
                )

                active.remove((process, case))
                write_json(manifest_path, manifest)

                print(
                    f'{case["dataset"]}/{case["condition"]}/{case["seed"]}: '
                    f'{case["status"]}',
                    flush=True,
                )

            if pending or active:
                time.sleep(0.5)

        # All tasks have finished.
        summarize(run_dir)

        if any(
            case['status'] != 'completed'
            for case in manifest['cases']
        ):
            manifest['status'] = 'failed'
            return 1

        manifest['status'] = 'completed'
        return 0
    finally:
        # Stop full child process groups, including a child currently training.
        for process, case in active:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 10
        for process, case in active:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            case.update(status='interrupted', returncode=process.returncode, finished_at=now())
        if manifest['status'] == 'running':
            manifest['status'] = 'interrupted'
        manifest['finished_at'] = now()
        write_json(manifest_path, manifest)
        summarize(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['design', 'gdmsr'], default='design',
                        help='target recommender; shadow=smia and attack=mlp')
    parser.add_argument('--targets', nargs='+', choices=['design', 'gdmsr'],
                        help='optional model list sharing the same global job queue; supersedes --target')
    parser.add_argument('--datasets', nargs='+', choices=['ciao', 'epinions', 'flickr', 'yelp'], default=['ciao'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--jobs', type=int, default=5, help='global maximum active pipelines across targets, datasets, SAL variants, lambdas and seeds')
    parser.add_argument('--sal-weights', nargs='+', type=float, default=[],
                        help='queue baseline and paired SAL lambdas globally; 0 adds a zero-weight control')
    parser.add_argument('--override', action='append', default=[], help='one Hydra key=value; repeat for more')
    parser.add_argument('--sal-override', action='append', default=[],
                        help='SAL-only Hydra defense key=value; repeatable, never applied to baseline')
    parser.add_argument('--sal-variants', type=Path,
                        help='JSON list of {name, overrides}; all variants share the global job queue')
    parser.add_argument('--diagnostics', action='store_true',
                        help='enable SAL diagnostics with a fresh per-batch target cache identity')
    parser.add_argument('--plan-only', action='store_true', help='write commands and source snapshot without launching')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        if args.run_dir is None:
            parser.error('--summarize-only requires --run-dir')
        return 0 if summarize((ROOT / args.run_dir).resolve()) else 1
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if any(not math.isfinite(weight) or weight < 0 for weight in args.sal_weights):
        parser.error('--sal-weights must be finite and nonnegative')
    # Keep the scheduled identity and model family consistent with every record.
    reserved = {'seed', 'target/data', 'target/model', 'shadow/model', 'attack/model', 'experiment', 'defaults'}
    for override in args.override:
        key, separator, _ = override.partition('=')
        if not separator or key.lstrip('+~') in reserved or key.lstrip('+~').startswith('hydra.'):
            parser.error(f'unsupported override: {override}; use dataset/seed options for scheduling')
        if args.sal_weights and '.defense' in key:
            parser.error('defense overrides cannot apply to the paired baseline; use --sal-weights for lambda')
    args.variants = (json.loads((ROOT / args.sal_variants).read_text()) if args.sal_variants
                     else [{'name': 'sal', 'overrides': []}])
    names = [variant['name'] for variant in args.variants]
    if not names or len(names) != len(set(names)) or any(not re.fullmatch(r'[a-zA-Z0-9_-]+', name) for name in names):
        parser.error('SAL variant names must be unique path-safe names')
    for override in args.sal_override + [value for variant in args.variants for value in variant['overrides']]:
        key, separator, value = override.partition('=')
        if (not separator or not key.startswith('target.model.attribute.defense.')
                or key.endswith(('.loss_weight', '._target_', '.diagnostics.run_id')) or key.endswith('._recursive_')):
            parser.error(f'unsupported SAL override: {override}')
    if (args.diagnostics or args.sal_override or args.sal_variants) and not args.sal_weights:
        parser.error('SAL options require --sal-weights')
    args.targets = list(dict.fromkeys(args.targets or [args.target]))
    args.datasets = list(dict.fromkeys(args.datasets))
    args.seeds = list(dict.fromkeys(args.seeds))
    args.sal_weights = list(dict.fromkeys(args.sal_weights))
    batch_name = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S_%f')
    if args.target != 'design':
        batch_name += f'_{args.target}'
    default_dir = Path('runs/restored-baseline') / batch_name
    run_dir = (ROOT / (args.run_dir or default_dir)).resolve()
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        return run_batch(args, run_dir)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    sys.exit(main())

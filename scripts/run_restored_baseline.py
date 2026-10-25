#!/usr/bin/env python3
"""Launch parallel DESIGN/SMIA/MLP seeds, or summarize existing run artifacts.

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
    for case in manifest['cases']:
        path = run_dir / case['output'] / 'metrics.json'
        status = case['status']
        if status == 'completed' and not path.is_file():
            status = 'missing_metrics'
        cases.append({key: case[key] for key in ('dataset', 'seed', 'output')} | {'status': status})
        if status != 'completed':
            continue
        record = json.loads(path.read_text())
        if (record['dataset'], record['seed']) != (case['dataset'], case['seed']):
            raise ValueError(f'Metric identity does not match manifest: {path}')
        for stage, groups in record['metrics'].items():
            for group, metrics in groups.items():
                for metric, value in metrics.items():
                    key = (case['dataset'], stage, group, metric)
                    bucket = buckets.setdefault(key, {'values': [], 'seeds': [], 'missing_seeds': []})
                    if value is None or not math.isfinite(value):
                        bucket['missing_seeds'].append(case['seed'])
                    else:
                        bucket['values'].append(value)
                        bucket['seeds'].append(case['seed'])
    rows = []
    for key, bucket in sorted(buckets.items()):
        values = bucket['values']
        rows.append(dict(zip(('dataset', 'stage', 'group', 'metric'), key)) | {
            'mean': statistics.mean(values) if values else None,
            'sample_std': statistics.stdev(values) if len(values) > 1 else None,
            'n': len(values), 'seeds': bucket['seeds'], 'missing_seeds': bucket['missing_seeds'],
        })
    summary = {'schema': 1, 'updated_at': now(), 'cases': cases, 'metrics': rows}
    write_json(run_dir / 'summary.json', summary)
    fields = ['dataset', 'stage', 'group', 'metric', 'mean', 'sample_std', 'n', 'seeds', 'missing_seeds']
    with (run_dir / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row | {key: json.dumps(row[key]) for key in ('seeds', 'missing_seeds')})
    completed = sum(case['status'] == 'completed' for case in cases)
    print(f'{completed}/{len(cases)} completed; summary: {run_dir / "summary.csv"}', flush=True)
    return completed == len(cases)


def snapshot(run_dir):
    destination = run_dir / 'source'
    for folder in ('sralt', 'scripts'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.suffix in {'.py', '.yaml', '.sh'}:
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
        'schema': 1, 'created_at': now(), 'python': sys.executable,
        'cache_namespace': 'default', 'cache_refresh': False, 'jobs': args.jobs,
        'overrides': args.override, 'status': 'running', 'cases': [],
    }
    for dataset in args.datasets:
        for seed in args.seeds:
            output = f'{dataset}/seed-{seed}'
            command = [
                sys.executable, str(ROOT / 'scripts/pipeline_low_memory.py'),
                f'target/data={dataset}', 'target/model=design', 'shadow/model=smia',
                'attack/model=mlp', f'seed={seed}', *args.override,
                f'hydra.run.dir={run_dir / output}', 'hydra.job.chdir=False',
            ]
            manifest['cases'].append({
                'dataset': dataset, 'seed': seed, 'output': output,
                'command': command, 'status': 'pending',
            })
    manifest_path = run_dir / 'manifest.json'
    write_json(manifest_path, manifest)
    print(f'Run directory: {run_dir}', flush=True)
    active = []
    next_start = 0.0
    try:
        for dataset in args.datasets:
            pending = [case for case in manifest['cases'] if case['dataset'] == dataset]
            while pending or active:
                while pending and len(active) < args.jobs and time.monotonic() >= next_start:
                    case = pending.pop(0)
                    output = run_dir / case['output']
                    output.mkdir(parents=True)
                    with (output / 'console.log').open('w') as log:
                        process = subprocess.Popen(
                            case['command'], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        )
                    active.append((process, case))
                    next_start = time.monotonic() + 10.0
                    case.update(status='running', pid=process.pid, started_at=now())
                    write_json(manifest_path, manifest)
                    print(f'Started {dataset}/{case["seed"]}: PID {process.pid}', flush=True)
                for process, case in active[:]:
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    has_metrics = (run_dir / case['output'] / 'metrics.json').is_file()
                    case.update(
                        returncode=returncode, finished_at=now(),
                        status='completed' if returncode == 0 and has_metrics else 'failed',
                    )
                    active.remove((process, case))
                    write_json(manifest_path, manifest)
                    print(f'{case["dataset"]}/{case["seed"]}: {case["status"]}', flush=True)
                if pending or active:
                    time.sleep(0.5)
            summarize(run_dir)
            if any(case['status'] != 'completed' for case in manifest['cases'] if case['dataset'] == dataset):
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
    parser.add_argument('--datasets', nargs='+', choices=['ciao', 'epinions', 'flickr', 'yelp'], default=['ciao'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--jobs', type=int, default=5, help='maximum parallel seeds within one dataset')
    parser.add_argument('--override', action='append', default=[], help='one Hydra key=value; repeat for more')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        if args.run_dir is None:
            parser.error('--summarize-only requires --run-dir')
        return 0 if summarize((ROOT / args.run_dir).resolve()) else 1
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    # Keep the scheduled identity and model family consistent with every record.
    reserved = {'seed', 'target/data', 'target/model', 'shadow/model', 'attack/model', 'experiment', 'defaults'}
    for override in args.override:
        key, separator, _ = override.partition('=')
        if not separator or key.lstrip('+~') in reserved or key.lstrip('+~').startswith('hydra.'):
            parser.error(f'unsupported override: {override}; use dataset/seed options for scheduling')
    args.datasets = list(dict.fromkeys(args.datasets))
    args.seeds = list(dict.fromkeys(args.seeds))
    default_dir = Path('runs/restored-baseline') / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S_%f')
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

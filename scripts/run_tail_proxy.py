#!/usr/bin/env python3
"""Schedule the fixed 100-epoch TailProxy comparison with one global job limit.

A worker handles one target/dataset/seed and reuses its upstream stages across
baseline and TailProxy. Planning and summarization use the standard library only.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from run_restored_baseline import ROOT, now, write_json
from summarize_tail_proxy import CONDITIONS, EXPERIMENT_REVISION, read_record, summarize


def run(args):
    run_dir = (ROOT / args.run_dir).resolve()
    identity = dict(targets=args.targets, datasets=args.datasets, seeds=args.seeds, experiment_revision=EXPERIMENT_REVISION,
                    cpu=args.cpu, threads=args.threads)
    if args.resume:
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        if manifest['identity'] != identity:
            raise ValueError('Resume options/version differ from the saved manifest')
        for case in manifest['cases']:
            valid = case['status'] == 'completed'
            if valid:
                try:
                    for condition in CONDITIONS:
                        read_record(run_dir, case, condition)
                except (OSError, ValueError, KeyError, TypeError):
                    valid = False
            if not valid:
                case['status'] = 'pending'
                case['command'][1] = str(ROOT / 'scripts/tail_proxy_case.py')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = dict(schema=3, created_at=now(), python=sys.executable, identity=identity,
                        targets=args.targets, datasets=args.datasets, seeds=args.seeds, jobs=args.jobs,
                        launch_interval_seconds=10, status='planned', conditions=list(CONDITIONS), cases=[])
        # Visit all four seed-42 combinations before later seeds.
        for seed in args.seeds:
            for target in args.targets:
                for dataset in args.datasets:
                    output = f'{target}/{dataset}/seed-{seed}'
                    command = [sys.executable, str(ROOT / 'scripts/tail_proxy_case.py'),
                               '--target', target, '--dataset', dataset, '--seed', str(seed),
                               '--output', str(run_dir / output),
                               '--threads', str(args.threads)]
                    if args.cpu:
                        command.append('--cpu')
                    manifest['cases'].append(dict(target=target, dataset=dataset, seed=seed, output=output,
                                                  command=command, status='pending'))
    manifest_path = run_dir / 'manifest.json'
    manifest['jobs'] = args.jobs
    write_json(manifest_path, manifest)
    count = len(manifest['cases']) * len(CONDITIONS)
    if args.plan_only:
        print(f'Plan: {len(manifest["cases"])} workers, {count} evaluations; {manifest_path}')
        return 0
    env = os.environ.copy()
    env.pop('SRALT_CACHE_VERSION', None)
    env.pop('SRALT_CACHE_REFRESH', None)
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    active, pending = [], [c for c in manifest['cases'] if c['status'] != 'completed']
    manifest.update(status='running', started_at=now())
    write_json(manifest_path, manifest)
    next_start = 0.
    summarized = False
    try:
        while pending or active:
            if pending and len(active) < args.jobs and time.monotonic() >= next_start:
                case = pending.pop(0)
                directory = run_dir / case['output']
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / 'console.log').open('a') as log:
                    process = subprocess.Popen(case['command'], cwd=ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                active.append((process, case))
                case.update(status='running', pid=process.pid, started_at=now())
                next_start = time.monotonic() + 10.
                write_json(manifest_path, manifest)
                print(f'Start {case["output"]}: pid={process.pid}', flush=True)
            for process, case in active[:]:
                code = process.poll()
                if code is None:
                    continue
                complete = False
                if code == 0:
                    try:
                        for condition in CONDITIONS:
                            read_record(run_dir, case, condition)
                        complete = True
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                case.update(status='completed' if code == 0 and complete else 'failed', returncode=code, finished_at=now())
                active.remove((process, case))
                write_json(manifest_path, manifest)
                print(f'{case["output"]}: {case["status"]}', flush=True)
            if pending or active:
                time.sleep(.5)
        success = summarize(run_dir)
        summarized = True
        manifest['status'] = 'completed' if success and all(c['status'] == 'completed' for c in manifest['cases']) else 'failed'
        return 0 if manifest['status'] == 'completed' else 1
    finally:
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
        if not summarized:
            summarize(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', choices=['design', 'gdmsr'], default=['design', 'gdmsr'])
    parser.add_argument('--datasets', nargs='+', choices=['ciao', 'flickr'], default=['ciao', 'flickr'])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(42, 47)))
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true', help='explicitly train/evaluate on CPU instead of selecting GPUs')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.summarize_only:
        if not args.run_dir:
            parser.error('--summarize-only requires --run-dir')
        return 0 if summarize((ROOT / args.run_dir).resolve()) else 1
    if args.jobs < 1 or args.threads < 1:
        parser.error('jobs and threads must be positive')
    if args.resume and not args.run_dir:
        parser.error('--resume requires --run-dir')
    for name in ('targets', 'datasets', 'seeds'):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    args.run_dir = args.run_dir or Path('runs/tail-proxy') / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('fixed100-%Y%m%d-%H%M%S')
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())

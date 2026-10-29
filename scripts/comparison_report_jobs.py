"""Subprocess scheduling for comparison_report; no model imports in the parent."""

import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OPTION_KEYS = ('datasets', 'targets', 'pairs', 'seeds', 'sal_weights', 'target_metrics',
               'shadow_metrics', 'allow_training', 'refresh_metrics', 'threads')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def row_key(row):
    return row['dataset'], row['target_config'], row['defense'], row['seed'], row['shadow'], row['attack']


def namespace():
    return {key: os.environ.get(key) for key in ('SRALT_CACHE_VERSION', 'SRALT_CACHE_REFRESH')}


def run_worker(args, collect, source_digest):
    request = json.loads(args.worker_request.read_text())
    if request['sources'] != source_digest() or request['environment'] != namespace():
        raise ValueError('Worker source/cache environment differs from the scheduler.')
    options = argparse.Namespace(**request['options'], dry_run=False)
    rows = []

    def record(row):
        rows.append(row)
        write_json(args.worker_output, dict(sources=request['sources'], rows=rows))

    # Different report batches may request the same upstream models. Serialize
    # those cases too, using a lock independent of the output directory/pairs.
    identity = [options.datasets, options.targets, options.seeds, options.sal_weights,
                request['environment']['SRALT_CACHE_VERSION']]
    lock_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    lock_dir = Path(tempfile.gettempdir()) / 'sralt-comparison-case-locks'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f'{lock_key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        payload = collect(options, sources=request['sources'], on_row=record)
    write_json(args.worker_output, payload)
    return 0


def merge(args, manifest, run_dir, collect, write_report):
    # Read metric caches to recover completed rows even after a worker failed
    # between writing its metric artifact and publishing its partial result.
    inspection = copy.copy(args)
    inspection.dry_run = True
    inspection.refresh_metrics = False
    payload = collect(inspection, sources=manifest['sources'], log_progress=False)
    replacements = {}
    for case in manifest['cases']:
        path = run_dir / case['output'] / 'metrics.json'
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved['sources'] != manifest['sources']:
                raise ValueError(f'Worker output has a different source version: {path}')
            replacements.update((row_key(row), row) for row in saved['rows'])
    payload['rows'] = [replacements.get(row_key(row), row) for row in payload['rows']]
    payload['generated_at'] = now()
    payload['jobs'] = args.jobs
    payload['worker_status'] = {case['output']: case['status'] for case in manifest['cases']}
    payload['run_dir'] = str(run_dir)
    write_report(payload, Path(manifest['report_output']))
    return payload


def print_progress(manifest):
    counts = {state: sum(c['status'] == state for c in manifest['cases'])
              for state in ('pending', 'running', 'completed', 'failed', 'interrupted')}
    logging.info('Workers: %s; run directory: %s', counts, manifest['run_dir'])


def execute(args, manifest, run_dir, collect, write_report):
    manifest_path = run_dir / 'manifest.json'
    pending = [c for c in manifest['cases'] if c['status'] != 'completed']
    active = []
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    manifest.update(status='running', jobs=args.jobs, started_at=now())
    write_json(manifest_path, manifest)
    next_start = next_progress = 0.
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        while pending or active:
            if pending and len(active) < args.jobs and time.monotonic() >= next_start:
                case = pending.pop(0)
                directory = run_dir / case['output']
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(ROOT / 'scripts/comparison_report.py'),
                           '--worker-request', str(directory / 'request.json'),
                           '--worker-output', str(directory / 'metrics.json')]
                with (directory / 'console.log').open('a') as handle:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                case.update(status='running', pid=process.pid, started_at=now(), command=command)
                active.append((process, case))
                next_start = time.monotonic() + args.launch_interval
                write_json(manifest_path, manifest)
                logging.info('Start %s: pid=%d', case['output'], process.pid)
            for process, case in active[:]:
                code = process.poll()
                if code is None:
                    continue
                case.update(status='completed' if code == 0 else 'failed', returncode=code, finished_at=now())
                active.remove((process, case))
                write_json(manifest_path, manifest)
                logging.info('%s: %s', case['output'], case['status'])
                merge(args, manifest, run_dir, collect, write_report)
            if time.monotonic() >= next_progress:
                print_progress(manifest)
                next_progress = time.monotonic() + args.progress_interval
            if pending or active:
                time.sleep(.5)
        manifest['status'] = 'completed' if all(c['status'] == 'completed' for c in manifest['cases']) else 'failed'
        return 0 if manifest['status'] == 'completed' else 1
    finally:
        for process, case in active:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 10.
        for process, case in active:
            try:
                process.wait(timeout=max(0., deadline - time.monotonic()))
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
        signal.signal(signal.SIGTERM, previous)
        merge(args, manifest, run_dir, collect, write_report)
        print_progress(manifest)


def run(args, collect, write_report, source_digest):
    run_dir = (args.run_dir or ROOT / 'runs/comparison-jobs' / datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')).resolve()
    if args.resume or args.summarize_only:
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        if manifest['sources'] != source_digest() or manifest['environment'] != namespace():
            raise ValueError('Source/cache environment changed; use a new comparison run directory.')
        for key, value in manifest['options'].items():
            setattr(args, key, value)
        args.output = Path(manifest['report_output'])
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        sources = source_digest()
        options = {key: getattr(args, key) for key in OPTION_KEYS}
        manifest = dict(schema=1, sources=sources, options=options, environment=namespace(),
                        run_dir=str(run_dir), report_output=str(args.output.resolve()),
                        created_at=now(), status='planned', cases=[])
        for seed in args.seeds:
            for target in args.targets:
                weights = (list(dict.fromkeys(args.sal_weights))
                           if target in ('gdmsr_sal', 'egcl_sal') and args.sal_weights else [None])
                for dataset in args.datasets:
                    for index, weight in enumerate(weights):
                        output = f'{target}/{dataset}/seed-{seed}' + (f'/weight-{index}' if weight is not None else '')
                        selected = dict(options, datasets=[dataset], targets=[target], seeds=[seed],
                                        sal_weights=[weight] if weight is not None else None)
                        write_json(run_dir / output / 'request.json',
                                   dict(sources=sources, environment=namespace(), options=selected))
                        manifest['cases'].append(dict(output=output, status='pending'))
        write_json(run_dir / 'manifest.json', manifest)
    # Avoid two schedulers updating the same run/report concurrently.
    with (run_dir / 'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.summarize_only:
            merge(args, manifest, run_dir, collect, write_report)
            return 0
        for case in manifest['cases']:
            if case['status'] != 'completed' or not (run_dir / case['output'] / 'metrics.json').is_file():
                case['status'] = 'pending'
        logging.info('Comparison: %d workers, jobs=%d, threads=%d; %s',
                     len(manifest['cases']), args.jobs, args.threads, run_dir)
        try:
            return execute(args, manifest, run_dir, collect, write_report)
        except KeyboardInterrupt:
            return 130

"""Read-only terminal progress for the shared multi-condition experiment runner."""

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def positive_interval(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError('progress interval must be a positive finite number')
    return seconds


def add_progress_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--progress-only', action='store_true', help='print saved progress without launching workers')
    group.add_argument('--watch', action='store_true', help='watch an existing run without launching workers')
    parser.add_argument('--progress-interval', type=positive_interval, default=30.,
                        help='seconds between progress updates (default: 30)')


def latest_activity(path):
    try:
        with path.open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 32768))
            tail = handle.read().decode('utf-8', errors='replace')
    except FileNotFoundError:
        return '等待日志'
    for line in reversed(tail.splitlines()):
        if ' INFO ' not in line:
            continue
        message = line.split(' INFO ', 1)[1]
        epoch = re.search(r'(.+?epoch[=:]\s*\d+)', message)
        if epoch:
            return epoch.group(1)
        if 'fused social:' in message:
            epoch = re.search(r"'epoch':\s*(\d+)", message)
            if epoch:
                return f'ItemPrefixCSA social epoch={epoch.group(1)}'
        # Also expose cache loading, evaluation and target preparation between epochs.
        return message[:140]
    return '等待阶段日志'


def print_progress(run_dir, manifest=None):
    run_dir = Path(run_dir)
    if manifest is None:
        manifest = json.loads((run_dir / 'manifest.json').read_text())
    conditions = manifest['conditions']
    revision = manifest['identity']['experiment_revision']
    states = Counter(case['status'] for case in manifest['cases'])
    done, finished, details = 0, 0, []
    for case in manifest['cases']:
        path = run_dir / case['output'] / 'metrics.json'
        results = {}
        note = ''
        try:
            saved = json.loads(path.read_text())
            if saved.get('experiment_revision') == revision:
                results = saved['results']
            else:
                note = '（结果版本与manifest不符）'
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            note = '（结果文件写入中，稍后刷新）'
        completed = sum(name in results for name in conditions)
        done += completed
        finished += completed == len(conditions)
        if case['status'] in ('running', 'failed', 'interrupted'):
            next_condition = next((name for name in conditions if name not in results), None)
            activity = latest_activity(path.parent / 'console.log')
            label = f'下一待完成={next_condition}' if next_condition else '条件已完成，等待进程收尾'
            details.append(f'  {case["output"]} [{case["status"]}] {completed}/{len(conditions)} '
                           f'{label}{note} | {activity}')
    total = len(manifest['cases']) * len(conditions)
    percent = 100 * done / total if total else 0.
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S')
    print(f'[{stamp}] Progress: 评价 {done}/{total} ({percent:.1f}%) | '
          f'条件齐全case {finished}/{len(manifest["cases"])} | '
          f'运行 {states["running"]} 排队 {states["pending"]} '
          f'失败 {states["failed"]} 中断 {states["interrupted"]} | '
          f'批次 {manifest["status"]}', flush=True)
    for detail in details:
        print(detail, flush=True)
    return manifest['status']


def handle_progress_mode(args, parser):
    if not (args.progress_only or args.watch):
        return None
    if not args.run_dir:
        parser.error('--progress-only/--watch requires --run-dir')
    if args.resume or args.plan_only or args.summarize_only:
        parser.error('--progress-only/--watch cannot be combined with resume, plan-only or summarize-only')
    run_dir = (ROOT / args.run_dir).resolve()
    try:
        while True:
            status = print_progress(run_dir)
            if not args.watch or status != 'running':
                return 0
            time.sleep(min(args.progress_interval, 60.))
            # Longer intervals still permit prompt Ctrl-C without a long blocking sleep.
            remaining = args.progress_interval - 60.
            while remaining > 0:
                time.sleep(min(remaining, 60.))
                remaining -= 60.
    except KeyboardInterrupt:
        return 130

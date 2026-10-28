#!/usr/bin/env python3
"""Schedule Stage-B main comparisons/ablations, resume or summarize saved runs.

Planning and summarization use only the standard library. Workers use existing
target/query/Shadow/Attack caches; no model weights are exported into runs.
"""

import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

from item_prefix_recipe import EXPERIMENT_REVISION, SUITES, contrasts
from run_tail_proxy import run
from run_restored_baseline import ROOT
from summarize_tail_proxy import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', choices=['design', 'gdmsr'], default=['design', 'gdmsr'])
    parser.add_argument('--datasets', nargs='+', choices=['ciao', 'flickr'], default=['ciao', 'flickr'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 47)))
    parser.add_argument('--suite', choices=list(SUITES), default='main')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
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
    for key in ('targets', 'datasets', 'seeds'):
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    args.run_dir = args.run_dir or Path('runs/hierarchical-discrete-proxies') / datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime('stage-b-%Y%m%d-%H%M%S')
    conditions = SUITES[args.suite]
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        return run(args, conditions=conditions, experiment_revision=EXPERIMENT_REVISION,
                   worker_name='item_prefix_case.py', worker_arguments=['--conditions', *conditions],
                   identity_extra=dict(suite=args.suite, conditions=list(conditions)),
                   metadata=dict(contrasts=contrasts(conditions), save_paired_cases=True,
                                 report_title='# Item Prefix 阶段B实验结果',
                                 method_description='若包含baseline，沿用已接受SMIA/MLP；TailProxy为固定100轮原Full。所有ItemPrefix消融共享缓存行为encoder/RVQ。最终CPU共用固定隐藏用户对及分组；公开验证All选模，TPR不参与选模。',
                                 contrast_description='差值方向见paired.csv的left−right；逐seed见paired_cases.csv。主比较与消融分别解释，不将阶段A结果计入本批完成数。'))
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())

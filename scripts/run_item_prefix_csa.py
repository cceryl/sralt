#!/usr/bin/env python3
"""Schedule ItemPrefixCSA comparisons and ablations; resume or summarize runs."""

import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

from item_prefix_csa_recipe import EXPERIMENT_REVISION, SUITES, contrasts
from run_tail_proxy import run
from run_restored_baseline import ROOT
from summarize_tail_proxy import summarize
from experiment_progress import add_progress_arguments, handle_progress_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', choices=['design', 'gdmsr'], default=['design', 'gdmsr'])
    parser.add_argument('--datasets', nargs='+', choices=['ciao', 'flickr'], default=['ciao', 'flickr'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 47)))
    parser.add_argument('--suite', choices=list(SUITES), default='core')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    add_progress_arguments(parser)
    args = parser.parse_args()
    progress_result = handle_progress_mode(args, parser)
    if progress_result is not None:
        return progress_result
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
    args.run_dir = args.run_dir or Path('runs/item-prefix-csa') / datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    conditions = SUITES[args.suite]
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        return run(args, conditions=conditions, experiment_revision=EXPERIMENT_REVISION,
            worker_name='item_prefix_csa_case.py', worker_arguments=['--conditions', *conditions],
            identity_extra=dict(suite=args.suite, conditions=list(conditions)),
            metadata=dict(contrasts=contrasts(conditions), save_paired_cases=True,
                report_title='# ItemPrefixCSA 实验结果',
                method_description='双RVQ行为联合训练后冻结；两图分别传播、融合后共享BPR。Full与图/matcher消融共享双RVQ缓存；评分消融使用独立行为缓存。CSA保留原300轮配置，SMIA/MLP与ItemPrefix沿用原配置，整模型参照不作单因素归因。',
                contrast_description='paired.csv/paired_cases.csv记录left−right；All/Head/Tail/TT/isolated_Tail使用同一评价用户对。公开验证选模，不用隐藏结果选择模型。'))
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())

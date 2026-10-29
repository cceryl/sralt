#!/usr/bin/env python3
"""Schedule ItemPrefixCSA comparisons and ablations; resume or summarize runs."""

import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

from item_prefix_csa_recipe import FOLLOWUP_REVISION, SUITES, contrasts, experiment_revision
from run_tail_proxy import run
from run_restored_baseline import ROOT
from summarize_tail_proxy import summarize
from experiment_progress import add_progress_arguments, handle_progress_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', choices=['design', 'gdmsr'], default=['design', 'gdmsr'])
    parser.add_argument('--datasets', nargs='+', choices=['ciao', 'flickr'], default=['ciao', 'flickr'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 47)))
    parser.add_argument('--suite', choices=list(SUITES), default='followup')
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
    revision = experiment_revision(conditions)
    method = ('均值融合Full；复查user图/matcher、增强边当前强度×50/20/10%、user边uniform/IDF。'
              '独立Social使用两套BPR/物品表/早停，Attack拼接五块特征。Behavior与双RVQ共享缓存。'
              if revision == FOLLOWUP_REVISION else
              '历史prefix10-v2配置：线性Full，两图共享BPR。CSA保留300轮及原MLP，整模型差异不作单因素归因。')
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        return run(args, conditions=conditions, experiment_revision=revision,
            worker_name='item_prefix_csa_case.py', worker_arguments=['--conditions', *conditions,
                '--expected-revision', revision],
            identity_extra=dict(suite=args.suite, conditions=list(conditions)),
            metadata=dict(contrasts=contrasts(conditions), save_paired_cases=True,
                report_title='# ItemPrefixCSA 实验结果',
                method_description=method,
                contrast_description='paired.csv/paired_cases.csv记录left−right；All/Head/Tail/TT/isolated_Tail使用同一评价用户对。公开验证选模，不用隐藏结果选择模型。'))
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())

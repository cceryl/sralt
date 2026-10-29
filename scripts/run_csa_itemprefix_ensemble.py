#!/usr/bin/env python3
"""Four-stage CSA/ItemPrefix experiment scheduler. Default: untrained S0 only."""

import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

from csa_itemprefix_ensemble_recipe import (EXPERIMENT_REVISION, SUITES, contrasts, source_fingerprint)
from run_tail_proxy import run
from run_restored_baseline import ROOT
from summarize_csa_itemprefix_ensemble import summarize
from experiment_progress import add_progress_arguments, handle_progress_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', choices=['design', 'gdmsr'], default=['design', 'gdmsr'])
    parser.add_argument('--datasets', nargs='+', choices=['ciao', 'flickr'], default=['ciao', 'flickr'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 47)))
    parser.add_argument('--suite', choices=list(SUITES), default='S0')
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
    args.run_dir = args.run_dir or Path('runs/csa-itemprefix-ensemble') / datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime(f'{args.suite}-%Y%m%d-%H%M%S')
    conditions, source = SUITES[args.suite], source_fingerprint()
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        return run(args, conditions=conditions, experiment_revision=EXPERIMENT_REVISION,
            worker_name='csa_itemprefix_ensemble_case.py', worker_arguments=['--conditions', *conditions,
                '--expected-revision', EXPERIMENT_REVISION, '--expected-source', source],
            identity_extra=dict(suite=args.suite, conditions=list(conditions), source_fingerprint=source),
            metadata=dict(contrasts=contrasts(conditions), save_paired_cases=True,
                report_title='# CSA + ItemPrefix 完整专家 Ensemble',
                method_description='S0使用原始概率固定融合；S1门控/校准用三折Attack级OOF（Shadow仍见完整T）。'
                    'H使用正式Attack固定隐藏坐标，U使用纯Shadow用户向量；H/U及degree-only在T拟合、V选模。'
                    'S1/H/U独立suite均附两原专家及原始均值；默认只运行S0，不自动触发后续阶段。',
                contrast_description='paired.csv/paired_cases.csv为left−right，统一用户对与五组评价，五seed报告样本SD。'
                    '公开V用于专家/副本和Fusion选模；隐藏结果只评价，不参与拟合或选权重。'), summarizer=summarize)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())

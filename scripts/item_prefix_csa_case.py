#!/usr/bin/env python3
"""One dual-RVQ case, sharing upstream caches and fixed evaluation pairs."""

import argparse
from pathlib import Path

from tail_proxy_case import configuration as baseline_configuration, run
from item_prefix_case import configuration as prefix_configuration
from item_prefix_csa_recipe import (CONDITION_OVERRIDES, FOLLOWUP_OVERRIDES, REFERENCES,
                                    SUITES, experiment_revision)
from hydra import compose, initialize_config_module


def configuration(args, condition):
    if condition == 'baseline':
        return baseline_configuration(args, condition)
    if condition in ('item_prefix', 'item_prefix_joint'):
        return prefix_configuration(args, 'rvq_end_to_end' if condition == 'item_prefix_joint' else condition)
    if condition == 'csa':
        selections = ['shadow/model=csa', 'attack/model=mlp']
    elif condition == 'independent_social':
        selections = ['shadow/model=item_prefix_csa_independent', 'attack/model=item_prefix_csa_independent']
    else:
        model = 'item_prefix_csa' if condition in FOLLOWUP_OVERRIDES else 'item_prefix_csa_legacy'
        selections = [f'shadow/model={model}', 'attack/model=item_prefix_csa', *CONDITION_OVERRIDES[condition]]
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        return compose(config_name='config', overrides=[f'target/data={args.dataset}',
            f'target/model={args.target}', f'seed={args.seed}', *selections])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['design', 'gdmsr'], required=True)
    parser.add_argument('--dataset', choices=['ciao', 'flickr'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--conditions', nargs='+', choices=[*REFERENCES, *CONDITION_OVERRIDES], default=list(SUITES['followup']))
    parser.add_argument('--expected-revision', help='scheduler-pinned experiment revision')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    revision = experiment_revision(args.conditions)
    if args.expected_revision is not None and args.expected_revision != revision:
        parser.error(f'Scheduler revision {args.expected_revision} differs from worker code {revision}; start a new batch.')
    run(args, conditions=args.conditions, experiment_revision=revision,
        configuration_fn=configuration, group_shadow_by_config=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""One dual-RVQ case, sharing upstream caches and fixed evaluation pairs."""

import argparse
from pathlib import Path

from tail_proxy_case import configuration as baseline_configuration, run
from item_prefix_case import configuration as prefix_configuration
from item_prefix_csa_recipe import EXPERIMENT_REVISION, OVERRIDES, REFERENCES, SUITES
from hydra import compose, initialize_config_module


def configuration(args, condition):
    if condition == 'baseline':
        return baseline_configuration(args, condition)
    if condition in ('item_prefix', 'item_prefix_joint'):
        return prefix_configuration(args, 'rvq_end_to_end' if condition == 'item_prefix_joint' else condition)
    if condition == 'csa':
        selections = ['shadow/model=csa', 'attack/model=mlp']
    else:
        selections = ['shadow/model=item_prefix_csa', 'attack/model=item_prefix_csa', *OVERRIDES[condition]]
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
    parser.add_argument('--conditions', nargs='+', choices=[*REFERENCES, *OVERRIDES], default=list(SUITES['main']))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    run(args, conditions=args.conditions, experiment_revision=EXPERIMENT_REVISION,
        configuration_fn=configuration, group_shadow_by_config=True)


if __name__ == '__main__':
    main()

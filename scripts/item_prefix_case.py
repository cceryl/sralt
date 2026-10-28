#!/usr/bin/env python3
"""One Stage-B case with shared upstream caches and fixed evaluation pairs."""

import argparse
from pathlib import Path

from item_prefix_recipe import EXPERIMENT_REVISION, MAIN, OVERRIDES
from tail_proxy_case import configuration as original_configuration, run
from hydra import compose, initialize_config_module


def configuration(args, condition):
    if condition in ('baseline', 'tail_proxy'):
        return original_configuration(args, condition)
    overrides = [f'target/data={args.dataset}', f'target/model={args.target}',
                 'shadow/model=item_prefix', 'attack/model=item_prefix', f'seed={args.seed}',
                 *OVERRIDES[condition]]
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        return compose(config_name='config', overrides=overrides)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['design', 'gdmsr'], required=True)
    parser.add_argument('--dataset', choices=['ciao', 'flickr'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--conditions', nargs='+', choices=['baseline', 'tail_proxy', *OVERRIDES], default=list(MAIN))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    run(args, conditions=args.conditions, experiment_revision=EXPERIMENT_REVISION,
        configuration_fn=configuration, group_shadow_by_config=True)


if __name__ == '__main__':
    main()

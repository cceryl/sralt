"""Adapters for the default S0 model and the explicit U+degree alternative.

Keep expert configurations native; persisted models still use existing loaders.
Saved experiment metrics can be reused without importing models or reading weights.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('score_mean', 'shadow_project_degree')


def expert_configs(config):
    from omegaconf import OmegaConf
    result = []
    for shadow, attack in (('csa', 'mlp'), ('item_prefix', 'item_prefix')):
        cfg = copy.deepcopy(config)
        cfg.shadow.model = OmegaConf.load(ROOT / f'sralt/configs/shadow/model/{shadow}.yaml')
        cfg.attack.model = OmegaConf.load(ROOT / f'sralt/configs/attack/model/{attack}.yaml')
        result.append(cfg)
    return result


def resolved_experiment_config(config):
    from omegaconf import OmegaConf
    from csa_itemprefix_ensemble_recipe import TRAINING
    return dict(experts=[{key: OmegaConf.to_container(c[key], resolve=True)
                           for key in ('target', 'shadow', 'attack', 'device')} for c in expert_configs(config)],
                deterministic=bool(config.deterministic), training=TRAINING)


def saved_result(config, condition, dataset, target):
    """Read only audited-schema artifacts with exact expert/protocol configs.

    The original source pin is checked against its own manifest, not rewritten
    to pretend a historical evaluation ran with the integration adapter.
    """
    import os
    from csa_itemprefix_ensemble_recipe import EXPERIMENT_REVISION
    if os.environ.get('SRALT_CACHE_VERSION', 'default') != 'default':
        return None
    expected = resolved_experiment_config(config)
    pattern = f'*/{target}/{dataset}/seed-{config.seed}/metrics.json'
    for path in sorted((ROOT / 'runs/csa-itemprefix-ensemble').glob(pattern)):
        manifest_path = path.parents[3] / 'manifest.json'
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        d = json.loads(path.read_text())
        r = d.get('results', {}).get(condition)
        if (r is None or 'f1_threshold' not in r.get('groups', {}).get('All', {})
                or 'TPR@FPR=0.05' not in r.get('groups', {}).get('All', {})
                or d.get('experiment_revision') != EXPERIMENT_REVISION
                or d.get('source_fingerprint') != manifest.get('identity', {}).get('source_fingerprint')
                or d.get('config') != expected or d.get('seed') != config.seed
                or (d.get('target'), d.get('dataset')) != (target, dataset)):
            continue
        data_dir = Path(str(config.target.data.directory))
        evaluated_at = datetime.fromisoformat(d['updated_at']).timestamp()
        if any(p.stat().st_mtime > evaluated_at for p in data_dir.iterdir() if p.is_file()):
            continue  # Do not import a snapshot predating a dataset-file change.
        fingerprints = [e['attack'] for e in d.get('experts', []) if 'attack' in e]
        if not fingerprints or r.get('frozen_fingerprints_before') != fingerprints or r.get('frozen_fingerprints_after') != fingerprints:
            continue
        if not all(d.get('acceptance', {}).get(k) for k in ('native_predict_parity', 'endpoint_symmetry', 'aligned_protocol')):
            continue
        return dict(document=d, result=r, path=str(path.relative_to(ROOT)))
    return None


def execute(config, output, *, condition=None, allow_training=True, threads=None, cache_loader=None):
    """Execute the existing case worker with native configs and guarded loaders."""
    import functools
    import torch
    from csa_itemprefix_ensemble_case import run
    from csa_itemprefix_ensemble_recipe import source_fingerprint
    from sralt.target import load_target_model
    from sralt.shadow import load_shadow_model
    from sralt.attack import load_attack_model
    from sralt.attack.ensemble.training import load_fusion_model
    if cache_loader is None:
        from comparison_report import load_report_model
    else:
        load_report_model = cache_loader

    condition = condition or str(config.attack.model.ensemble_condition)
    if condition not in CONDITIONS:
        raise ValueError(f'Unsupported deployed ensemble: {condition}')
    args = SimpleNamespace(dataset=str(config.target.data.name).lower(),
        target=str(config.target.model.name).lower(), seed=int(config.seed), output=Path(output),
        threads=threads or torch.get_num_threads(), cpu=False, conditions=[condition], source_id=source_fingerprint())
    configs = expert_configs(config)
    run(args, configuration_fn=lambda args, name: configs[0 if name == 'csa' else 1],
        target_loader=functools.partial(load_report_model, 'target', load_target_model, allow_training),
        expert_loaders=dict(
            shadow_loader=functools.partial(load_report_model, 'shadow', load_shadow_model, allow_training),
            attack_loader=functools.partial(load_report_model, 'attack', load_attack_model, allow_training)),
        fusion_loader=functools.partial(load_report_model, 'attack', load_fusion_model, allow_training,
            cache_directory='.cache/attack/ensemble/fusion',
            cache_ignore=['evidence', 'pool', 'validation', 'oof', 'device']))
    document = json.loads((Path(output) / 'metrics.json').read_text())
    return dict(document=document, result=document['results'][condition], path=str(Path(output) / 'metrics.json'))


def run_ensemble_pipeline(config):
    import logging
    from hydra.core.hydra_config import HydraConfig
    result = execute(config, HydraConfig.get().runtime.output_dir)
    for group, values in result['result']['groups'].items():
        logging.info('Ensemble %s: %s', group, values)

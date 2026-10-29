#!/usr/bin/env python3
"""Run one TailProxy case; retain one result JSON and use existing model caches."""
import argparse
import fcntl
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import tempfile

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sralt-matplotlib')

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sralt.target import load_target_data, load_target_model, test_target_model
from sralt.shadow import load_shadow_data, load_shadow_model, test_shadow_model
from sralt.attack import load_attack_data, load_attack_model
from sralt.attack.data import AttackData, EvalAttackDataset
from sralt.utils import head_tail_split, set_seed
from sralt.utils.model import tensor_fingerprint
from sralt.utils.attack_groups import grouped_attack_report, optimal_f1_threshold
from run_restored_baseline import now, write_json
from summarize_tail_proxy import CONDITIONS, EXPERIMENT_REVISION

GROUPS = ('All', 'Head', 'Tail', 'TT', 'isolated_Tail')
GROUP_FIELDS = ('samples', 'positives', 'negatives', 'f1_threshold', 'AUROC', 'F1Score', 'TPR@FPR=0.05')


def release(device):
    gc.collect()
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def configuration(args, condition):
    baseline = condition == 'baseline'
    overrides = [f'target/data={args.dataset}', f'target/model={args.target}',
                 f'shadow/model={"smia" if baseline else "tail_proxy"}',
                 f'attack/model={"mlp" if baseline else "tail_proxy"}', f'seed={args.seed}']
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        return compose(config_name='config', overrides=overrides)


def fixed_protocol(target, shadow, cfg):
    data = AttackData(target, shadow, None, cfg.attack.data.val_ratio)
    identity = {name: tensor_fingerprint(getattr(data, name)) for name in (
        'target_social_edge_index', 'public_social_edge_index', 'train_social_edge_index',
        'val_social_edge_index', 'test_social_edge_index')}
    result = {'identity': identity}
    for split in ('val', 'test'):
        set_seed(cfg.seed, deterministic=cfg.deterministic)
        dataset = EvalAttackDataset(data, split=split)
        result[split] = dict(edge_index=dataset.edge_index.cpu(), labels=dataset.labels.cpu())
    result['head'], result['tail'] = head_tail_split(data.num_users, data.target_social_edge_index,
                                                    float(cfg.attack.tail_ratio))
    evaluation_id = {split: tensor_fingerprint(result[split]['edge_index']) for split in ('val', 'test')}
    evaluation_id.update(head=tensor_fingerprint(result['head']), tail=tensor_fingerprint(result['tail']))
    return result, evaluation_id


@torch.no_grad()
def evaluate(model, data, cfg, protocol):
    model.cpu().eval()
    data.to(torch.device('cpu'))
    edges, labels = protocol['test']['edge_index'], protocol['test']['labels']
    batch_size = cfg.attack.test.evaluation.test_batch_size
    inputs = data if hasattr(model, 'endpoint_features') else data.user_features
    validation_edges = protocol['val']['edge_index']
    validation_probs = torch.cat([model.predict(validation_edges[:, start:start + batch_size], inputs).cpu()
                                  for start in range(0, validation_edges.size(1), batch_size)])
    threshold = optimal_f1_threshold(validation_probs, protocol['val']['labels'])
    probabilities = torch.cat([model.predict(edges[:, start:start + batch_size], inputs).cpu()
                               for start in range(0, labels.numel(), batch_size)])
    groups = grouped_attack_report(edges, probabilities, labels, data,
                                   protocol['head'], protocol['tail'], instantiate(cfg.attack.test.metrics),
                                   f1_threshold=threshold)
    return {group: {key: groups[group][key] for key in GROUP_FIELDS} for group in GROUPS}


def metrics_by_name(cfg, values):
    return {metric._target_.split('.')[-1].removeprefix('LinkPred') + f'@{metric.k}': value
            for metric, value in zip(cfg.metrics, values, strict=True)}


def short_training(info):
    return {name: value for name, value in info.items() if name != 'history'} if info else None


def run(args, *, conditions=CONDITIONS, experiment_revision=EXPERIMENT_REVISION,
        configuration_fn=configuration, group_shadow_by_config=False):
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    torch.set_num_threads(args.threads)
    cfg = configuration_fn(args, 'baseline')
    device = torch.device('cpu') if args.cpu else instantiate(cfg.device)
    common = dict(seed=args.seed, deterministic=cfg.deterministic)
    lock_dir = Path(tempfile.gettempdir()) / 'sralt-target-data-locks'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f'{args.dataset}-{args.seed}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = load_target_data(**common, config=cfg.target.data).to(device)
    model = load_target_model(**common, device=device, config=cfg.target.model, data=target, writer=None)
    target_scores = test_target_model(**common, device=device, model=model, data=target, config=cfg.target.test)
    shadow = load_shadow_data(**common, target_data=target, target_model=model, config=cfg.shadow.data)
    del model
    target.to(torch.device('cpu'))
    release(device)
    protocol, evaluation_id = fixed_protocol(target, shadow, cfg)
    path = args.output / 'metrics.json'
    shared_config = dict(target=OmegaConf.to_container(cfg.target, resolve=True),
                         shadow_data=OmegaConf.to_container(cfg.shadow.data, resolve=True),
                         shadow_test=OmegaConf.to_container(cfg.shadow.test, resolve=True),
                         attack_data=OmegaConf.to_container(cfg.attack.data, resolve=True),
                         attack_test=OmegaConf.to_container(cfg.attack.test, resolve=True),
                         tail_ratio=float(cfg.attack.tail_ratio), deterministic=bool(cfg.deterministic))
    if path.exists():
        result = json.loads(path.read_text())
        if (result['experiment_revision'] != experiment_revision
                or result['evaluation_id'] != evaluation_id or result['config'] != shared_config):
            raise ValueError('Resume input/configuration differs from the saved case.')
    else:
        result = dict(schema=3, experiment_revision=experiment_revision, target=args.target, dataset=args.dataset, seed=args.seed,
                      evaluation_id=evaluation_id, config=shared_config, device=str(device),
                      target_metrics=metrics_by_name(cfg.target.test, target_scores),
                      shadow={}, attack_configs={}, results={})
    for name in conditions:
        stage_cfg = configuration_fn(args, name)
        resolved = OmegaConf.to_container(stage_cfg.attack.model, resolve=True)
        if name in result['attack_configs'] and result['attack_configs'][name] != resolved:
            raise ValueError(f'Resume Attack settings changed: {name}')
        result['attack_configs'][name] = resolved
        shadow_config = OmegaConf.to_container(stage_cfg.shadow.model, resolve=True)
        if group_shadow_by_config:
            existing = result['results'].get(name)
            if existing and result['shadow'][existing['shadow_key']]['config'] != shadow_config:
                raise ValueError(f'Resume Shadow settings changed: {name}')
        if name in result['results']:
            continue
        logging.info('Starting condition %s [%d/%d]', name, conditions.index(name) + 1, len(conditions))
        budget = None if name == 'baseline' else int(stage_cfg.shadow.model.train.num_epochs)
        shadow.to(device)
        target.to(device)
        started = time.monotonic()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        shadow_model = load_shadow_model(**common, device=device, config=stage_cfg.shadow.model, data=shadow, writer=None)
        key = ('baseline' if budget is None else str(budget))
        if group_shadow_by_config:
            key = next((k for k, v in result['shadow'].items() if v['config'] == shadow_config), name)
            if hasattr(shadow_model, 'quantization_summary'):
                from sralt.utils import model_fingerprint
                shared = dict(behavior=model_fingerprint(shadow_model.behavioral_branch),
                              rvq=model_fingerprint(shadow_model.rvq))
                if hasattr(shadow_model, 'user_rvq'):
                    shared['user_rvq'] = model_fingerprint(shadow_model.user_rvq)
                shared_key = ('shared_item_encoder_end_to_end'
                              if getattr(shadow_model, 'rvq_training', 'offline') == 'end_to_end'
                              else 'shared_item_encoder')
                if hasattr(shadow_model, 'encoder_cache_group'):
                    shared_key = 'shared_encoder_' + shadow_model.encoder_cache_group
                if result.get(shared_key, shared) != shared:
                    raise ValueError('Item-prefix ablations did not reuse the same behavior encoder and RVQ')
                result[shared_key] = shared
        shadow_scores = (None if group_shadow_by_config and key in result['shadow'] else
                         test_shadow_model(**common, device=device, model=shadow_model, data=shadow, config=stage_cfg.shadow.test))
        data = load_attack_data(**common, target_data=target, shadow_data=shadow, shadow_model=shadow_model,
                                config=stage_cfg.attack.data)
        for attribute, value in protocol['identity'].items():
            if attribute.endswith('_edge_index') and tensor_fingerprint(getattr(data, attribute)) != value:
                raise ValueError(f'{attribute} changed between baseline and TailProxy')
        if shadow_scores is not None:
            result['shadow'][key] = dict(config=shadow_config,
                metrics=metrics_by_name(stage_cfg.shadow.test, shadow_scores),
                training={branch: short_training(info) for branch, info in getattr(shadow_model, 'branch_training', {}).items()},
                diagnostics=getattr(data, 'shadow_diagnostics', None), elapsed_seconds=time.monotonic() - started,
                peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
        del shadow_model
        target.to(torch.device('cpu'))
        shadow.to(torch.device('cpu'))
        data.to(torch.device('cpu'))
        release(device)
        if name != 'baseline':
            data.fixed_eval_pairs = {'val': protocol['val']}
        started = time.monotonic()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        data.to(device)
        model = load_attack_model(**common, device=device, config=stage_cfg.attack.model, data=data,
                                  tail_ratio=stage_cfg.attack.tail_ratio, writer=None)
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        groups = evaluate(model, data, stage_cfg, protocol)
        result['results'][name] = dict(groups=groups, shadow_budget=budget, shadow_key=key,
            training=getattr(model, 'training_summary', None), elapsed_seconds=time.monotonic() - started,
            peak_memory_bytes=peak)
        result['updated_at'] = now()
        write_json(path, result)
        logging.info('Completed %s: All AUC=%s F1=%s', name, groups['All']['AUROC'], groups['All']['F1Score'])
        del model
        del data
        release(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, choices=['design', 'gdmsr'])
    parser.add_argument('--dataset', required=True, choices=['ciao', 'flickr'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    run(args)


if __name__ == '__main__':
    main()

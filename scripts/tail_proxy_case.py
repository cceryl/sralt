#!/usr/bin/env python3
"""Execute one dataset/target/seed group of the TailProxy stage-one recipe."""
import argparse
import gc
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import os
from pathlib import Path
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sralt-matplotlib')

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from sralt.target import load_target_data, load_target_model, test_target_model
from sralt.shadow import load_shadow_data, load_shadow_model, test_shadow_model
from sralt.attack import load_attack_data, load_attack_model
from sralt.attack.data import AttackData, EvalAttackDataset
from sralt.utils import head_tail_split, set_seed
from sralt.utils.model import tensor_fingerprint
from sralt.utils.attack_groups import grouped_attack_report
from run_restored_baseline import now, write_json


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def release(device):
    gc.collect()
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def configuration(args, budget=None, variant=None):
    shadow, attack = ('smia', 'mlp') if budget is None else (f'tail_proxy_{budget}', variant['model'])
    overrides = [f'target/data={args.dataset}', f'target/model={args.target}',
                 f'shadow/model={shadow}', f'attack/model={attack}', f'seed={args.seed}']
    if variant:
        overrides += variant['overrides']
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        cfg = compose(config_name='config', overrides=overrides)
    if variant:
        cfg.attack.model.revision = f'tail-proxy-stage1-{variant["name"]}-v3'
    return cfg


def fixed_protocol(args, target, shadow, cfg):
    data = AttackData(target, shadow, None, cfg.attack.data.val_ratio)
    identity = {name: tensor_fingerprint(getattr(data, name)) for name in (
        'target_social_edge_index', 'public_social_edge_index', 'train_social_edge_index',
        'val_social_edge_index', 'test_social_edge_index')}
    identity.update(query=tensor_fingerprint(shadow.behavioral_edge_index), seed=args.seed,
                    target_revision=str(cfg.target.model.revision), tail_ratio=float(cfg.attack.tail_ratio))
    path = args.output / 'evaluation_pairs.pt'
    if path.exists():
        result = torch.load(path, map_location='cpu', weights_only=False)
        if result['identity'] != identity:
            raise ValueError('Resume evaluation pairs do not match the current query/social splits.')
    else:
        result = {'identity': identity}
        # Seed-reset matches the existing evaluator convention. These pairs are
        # generated once on CPU, then shared across every method in this group.
        for split in ('val', 'test'):
            set_seed(args.seed, deterministic=cfg.deterministic)
            dataset = EvalAttackDataset(data, split=split)
            result[split] = dict(edge_index=dataset.edge_index.cpu(), labels=dataset.labels.cpu())
        result['head'], result['tail'] = head_tail_split(data.num_users, data.target_social_edge_index,
                                                         float(cfg.attack.tail_ratio))
        temporary = path.with_suffix('.tmp')
        torch.save(result, temporary)
        temporary.replace(path)
    return result, sha(path)


@torch.no_grad()
def evaluate(model, data, cfg, protocol, directory):
    model.cpu().eval()
    data.to(torch.device('cpu'))
    edges, labels = protocol['test']['edge_index'], protocol['test']['labels']
    predictions = []
    batch_size = cfg.attack.test.evaluation.test_batch_size
    for start in range(0, labels.numel(), batch_size):
        inputs = data if hasattr(model, 'endpoint_features') else data.user_features
        predictions.append(model.predict(edges[:, start:start + batch_size], inputs).cpu())
    probabilities = torch.cat(predictions)
    groups = grouped_attack_report(edges, probabilities, labels, data,
                                   protocol['head'], protocol['tail'], instantiate(cfg.attack.test.metrics))
    torch.save(dict(edges=edges, labels=labels, probabilities=probabilities,
                    head=protocol['head'], tail=protocol['tail']), directory / 'scores.pt')
    return groups


def complete_existing(directory, cfg_hash, pair_hash):
    path = directory / 'metrics.json'
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record['config_sha256'] != cfg_hash or record['evaluation_sha256'] != pair_hash:
        raise ValueError(f'Resume record has a different identity: {directory}')
    if sha(directory / 'scores.pt') != record['scores_sha256']:
        raise ValueError(f'Resume scores were changed: {directory}')
    return True


def attack_run(args, cfg, data, protocol, pair_hash, device, name, recommendation, shadow_info):
    directory = args.output / name
    directory.mkdir(exist_ok=True)
    # Resolve only the research sections; writer/Hydra runtime interpolation is
    # irrelevant to identity and is supplied directly below.
    resolved = {key: OmegaConf.to_container(cfg[key], resolve=True)
                for key in ('target', 'shadow', 'attack')}
    resolved.update(seed=args.seed, deterministic=bool(cfg.deterministic))
    cfg_hash = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    if complete_existing(directory, cfg_hash, pair_hash):
        logging.info('Keep completed %s', name)
        return
    OmegaConf.save(OmegaConf.create(resolved), directory / 'config.yaml')
    # Existing accepted MLP weights retain their original validation/cache identity.
    # New TailProxy checkpoints include the fixed public-validation pair identity.
    if name != 'baseline':
        data.fixed_eval_pairs = {'val': protocol['val']}
    started = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with SummaryWriter(str(directory / 'tensorboard')) as writer:
        data.to(device)
        model = load_attack_model(seed=args.seed, deterministic=cfg.deterministic, device=device,
                                  config=cfg.attack.model, data=data, tail_ratio=cfg.attack.tail_ratio, writer=writer)
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    groups = evaluate(model, data, cfg, protocol, directory)
    torch.save(model.state_dict(), directory / 'attack_state.pt')
    record = dict(schema=1, target=args.target, dataset=args.dataset, seed=args.seed, condition=name,
                  config_sha256=cfg_hash, evaluation_sha256=pair_hash,
                  scores_sha256=sha(directory / 'scores.pt'), config=resolved,
                  config_file_sha256=sha(directory / 'config.yaml'),
                  model_repr=repr(model), groups=groups, recommendation=recommendation,
                  shadow=shadow_info, training=getattr(model, 'training_summary', None),
                  training_history=getattr(model, 'training_diagnostics', None),
                  normalization=getattr(model, 'normalization_diagnostics', None),
                  device=str(device), evaluation_device='cpu', peak_attack_memory_bytes=peak,
                  elapsed_seconds=time.monotonic() - started,
                  baseline_validation='cached model keeps original training validation candidates' if name == 'baseline'
                                      else 'shared fixed public-validation candidates', finished_at=now())
    write_json(directory / 'metrics.json', record)
    logging.info('Completed %s: All AUC=%s F1=%s', name, groups['All']['AUROC'], groups['All']['F1Score'])
    del model
    release(device)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / 'worker.log', mode='a')])
    torch.set_num_threads(args.threads)
    recipe = json.loads(args.recipe.read_text())
    versions = {}
    for package in ('torch', 'numpy', 'torch-geometric', 'vector-quantize-pytorch', 'hydra-core', 'torchmetrics'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    write_json(args.output / 'runtime.json', versions)
    cfg = configuration(args)
    device = torch.device('cpu') if args.cpu else instantiate(cfg.device)
    common = dict(seed=args.seed, deterministic=cfg.deterministic)
    lock_dir = args.recipe.parent / 'locks'
    lock_dir.mkdir(exist_ok=True)
    # DESIGN and GDMSR share TargetData for a dataset/seed. Serialize this
    # potentially missing cache even when those workers run concurrently.
    with (lock_dir / f'{args.dataset}-{args.seed}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = load_target_data(**common, config=cfg.target.data).to(device)
    with SummaryWriter(str(args.output / 'upstream-tensorboard')) as writer:
        model = load_target_model(**common, device=device, config=cfg.target.model, data=target, writer=writer)
        target_scores = test_target_model(**common, device=device, model=model, data=target, config=cfg.target.test)
        shadow = load_shadow_data(**common, target_data=target, target_model=model, config=cfg.shadow.data)
        write_json(args.output / 'upstream.json', dict(target_repr=repr(model), query_repr=repr(shadow),
                   target_metrics=target_scores, device=str(device), torch_version=torch.__version__))
        del model
        target.to(torch.device('cpu'))
        release(device)
        protocol, pair_hash = fixed_protocol(args, target, shadow, cfg)
        # Baseline and each Shadow budget are processed sequentially in this
        # worker, preventing concurrent writes to the same target/query/model key.
        for budget in [None, *recipe['budgets']]:
            variants = [None] if budget is None else [v for v in recipe['variants'] if budget in v['budgets']]
            if not variants:
                continue
            stage_cfg = cfg if budget is None else configuration(args, budget, variants[0])
            shadow.to(device)
            target.to(device)
            stage_started = time.monotonic()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            shadow_log = 'baseline-shadow-tensorboard' if budget is None else f'shadow-e{budget}-tensorboard'
            with SummaryWriter(str(args.output / shadow_log)) as shadow_writer:
                shadow_model = load_shadow_model(**common, device=device, config=stage_cfg.shadow.model,
                                                 data=shadow, writer=shadow_writer)
            shadow_scores = test_shadow_model(**common, device=device, model=shadow_model,
                                              data=shadow, config=stage_cfg.shadow.test)
            data = load_attack_data(**common, target_data=target, shadow_data=shadow, shadow_model=shadow_model,
                                    config=stage_cfg.attack.data)
            # Features may differ; all relation identities must match the fixed protocol.
            for attribute, value in protocol['identity'].items():
                if attribute.endswith('_edge_index') and tensor_fingerprint(getattr(data, attribute)) != value:
                    raise ValueError(f'{attribute} changed between baseline and TailProxy')
            shadow_info = dict(budget=budget, model_repr=repr(shadow_model),
                               training=getattr(shadow_model, 'branch_training', None),
                               diagnostics=getattr(data, 'shadow_diagnostics', None),
                               stage_elapsed_seconds=time.monotonic() - stage_started,
                               peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
            if budget is not None:
                torch.save(dict(state_dict=shadow_model.state_dict(), training=shadow_info['training'],
                                config=OmegaConf.to_container(stage_cfg.shadow.model, resolve=True)),
                           args.output / f'shadow-e{budget}.pt')
                write_json(args.output / f'shadow-e{budget}.json', shadow_info)
            del shadow_model
            target.to(torch.device('cpu'))
            shadow.to(torch.device('cpu'))
            data.to(torch.device('cpu'))
            release(device)
            for variant in variants:
                attack_cfg = cfg if budget is None else configuration(args, budget, variant)
                name = 'baseline' if budget is None else f'e{budget}-{variant["name"]}'
                attack_run(args, attack_cfg, data, protocol, pair_hash, device, name,
                           dict(target=target_scores, shadow=shadow_scores), shadow_info)
            del data
            release(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, choices=['design', 'gdmsr'])
    parser.add_argument('--dataset', required=True, choices=['ciao', 'flickr'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    run(args)


if __name__ == '__main__':
    main()

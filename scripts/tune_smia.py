#!/usr/bin/env python3
"""One finite, validation-only MLP sweep on frozen SMIA target/shadow caches.

Defaults: Ciao/Flickr x DESIGN/GDMSR x seeds42,43, 15 lr/batch settings
plus the original baseline (128 trials). Never evaluates hidden test labels,
never trains upstream models, and never starts a second round automatically.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import pickle
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=['ciao', 'flickr', 'epinions', 'yelp'], default=['ciao', 'flickr'])
    p.add_argument('--targets', nargs='+', choices=['design', 'gdmsr'], default=['design', 'gdmsr'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    p.add_argument('--lrs', nargs='+', type=float, default=[.001, .003, .01, .03, .05])
    p.add_argument('--batches', nargs='+', type=int, default=[256, 512, 1024])
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--selection', choices=['original', 'auc-f1'], default='original')
    p.add_argument('--output', type=Path, default=Path('runs/smia_tuning/round1'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--prepare-only', action='store_true', help='Resolve and freeze inputs, without attack training')
    return p.parse_args(argv)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def settings(args):
    yield {'label': 'baseline', 'lr': .05, 'batch': 512, 'epochs': 50, 'patience': 5}
    for lr, batch in itertools.product(args.lrs, args.batches):
        yield {'label': 'grid', 'lr': lr, 'batch': batch, 'epochs': args.epochs, 'patience': args.patience}


def cases(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    for dataset, target, seed in itertools.product(args.datasets, args.targets, args.seeds):
        with initialize_config_dir(config_dir=str(ROOT / 'sralt/configs'), version_base=None):
            cfg = compose(config_name='config', overrides=[f'target/data={dataset}', f'target/model={target}',
                                                          'shadow/model=smia', 'attack/model=mlp'])
        cfg.seed = seed
        identity = {'dataset': dataset, 'target': target, 'seed': seed, 'deterministic': cfg.deterministic,
                    'namespace': os.environ.get('SRALT_CACHE_VERSION', 'default'),
                    'target_config': OmegaConf.to_container(cfg.target, resolve=True),
                    'shadow_config': OmegaConf.to_container(cfg.shadow, resolve=True),
                    'attack_config': OmegaConf.to_container(cfg.attack, resolve=True)}
        yield cfg, identity, f'{dataset}-{target}-{seed}'


def freeze_input(cfg, device, folder):
    """Use exact pipeline dependencies; upstream model cache misses are fatal."""
    import torch
    from hydra.utils import instantiate
    from comparison_report import load_report_model
    from sralt.target import load_target_data, load_target_model
    from sralt.shadow import load_shadow_data, load_shadow_model
    from sralt.attack import load_attack_data

    instantiate(cfg.filter_warnings)
    common = dict(seed=cfg.seed, deterministic=cfg.deterministic)
    td = load_target_data(**common, config=cfg.target.data).to(device)
    target = load_report_model('target', load_target_model, False, **common, device=device,
                               config=cfg.target.model, data=td, writer=None)
    sd = load_shadow_data(**common, target_data=td, target_model=target, config=cfg.shadow.data).to(device)
    del target
    shadow = load_report_model('shadow', load_shadow_model, False, **common, device=device,
                               config=cfg.shadow.model, data=sd, writer=None)
    data = load_attack_data(**common, target_data=td, shadow_data=sd, shadow_model=shadow,
                            config=cfg.attack.data).to(torch.device('cpu'))
    del shadow, sd, td
    # Check the public protocol without consulting hidden test labels.
    encode = lambda edges: edges[0] * data.num_users + edges[1]
    train_ids, val_ids = encode(data.train_social_edge_index), encode(data.val_social_edge_index)
    assert not torch.isin(train_ids, val_ids).any(), 'Overlapping public train/validation relations'
    assert (data.train_social_edge_index[0] <= data.train_social_edge_index[1]).all()
    assert (data.val_social_edge_index[0] <= data.val_social_edge_index[1]).all()
    social = data.shadow_social_edge_index
    social_ids = encode(social[:, social[0] <= social[1]])
    assert torch.equal(social_ids.sort().values, train_ids.sort().values), 'Shadow must use attack training positives only'
    with (folder / 'input.pkl').open('wb') as handle:
        pickle.dump(data, handle)
    return data


def train_trial(data, cfg, setting, folder, device, selection):
    """Call the original MLP training loop and reconstruct its exact validation candidates."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.tensorboard import SummaryWriter
    from sralt.attack.model.base import create_attack_model
    from sralt.attack.data import EvalAttackDataset
    from sralt.utils import set_seed, DataLoader, EarlyStopping, head_tail_split, compute_attack_metrics

    config = OmegaConf.create(OmegaConf.to_container(cfg.attack.model, resolve=True))
    config.train.optimizer.lr = setting['lr']
    config.train.train_batch_size = setting['batch']
    config.train.num_epochs = setting['epochs']
    config.train.early_stopping.patience = setting['patience']
    if selection == 'auc-f1':
        config.train.metrics = config.train.metrics[:2]
    OmegaConf.save(config, folder / 'attack.yaml', resolve=True)
    common = dict(seed=cfg.seed, deterministic=cfg.deterministic)
    model = create_attack_model(**common, device=device, config=config.attribute,
                                user_feature_dim=data.user_features.size(1))
    # This repeats the seed at load_attack_model's train_model boundary. Dataset
    # construction before MLP validation consumes no random numbers.
    set_seed(**common)
    validation = EvalAttackDataset(data, split='val')
    from sralt.utils.model import tensor_fingerprint
    val_fingerprint = tensor_fingerprint(validation.edge_index)
    set_seed(**common)
    stopping = EarlyStopping(model, patience=setting['patience'], greater_is_better=True,
                             epsilon=config.train.early_stopping.epsilon)
    start = time.monotonic()
    writer = SummaryWriter(str(folder / 'tensorboard'))
    try:
        model._train_model(data=data, tail_ratio=cfg.attack.tail_ratio, writer=writer,
                           train_batch_size=setting['batch'], val_batch_size=config.train.val_batch_size,
                           negative_ratio=config.train.negative_ratio, num_epochs=setting['epochs'],
                           optimizer=instantiate(config.train.optimizer, params=model.parameters()),
                           metrics=instantiate(config.train.metrics), early_stopping=stopping)
    finally:
        writer.close()
    model.eval()
    metrics_config = cfg.attack.model.train.metrics
    head, tail = head_tail_split(data.num_users, data.target_social_edge_index, cfg.attack.tail_ratio)
    with torch.no_grad():
        values = compute_attack_metrics(
            device=device, predictor=lambda u, v: model.predict(torch.stack([u, v]), data.user_features),
            metrics=instantiate(metrics_config),
            eval_loader=DataLoader(validation, batch_size=config.train.val_batch_size, shuffle=False),
            head_indices=head, tail_indices=tail)
    names = ['AUROC', 'F1', 'TPR01']
    result = {'setting': setting, 'selection': selection, 'best_epoch': stopping.best_epoch,
              'stop_epoch': stopping.epoch, 'selection_score': stopping.best_metric,
              'elapsed_seconds': time.monotonic() - start, 'validation_pairs': val_fingerprint,
              'validation': {g: dict(zip(names, v, strict=True)) for g, v in zip(['all', 'head', 'tail'], values, strict=True)}}
    selected = result['validation']['all']
    expected_score = selected['AUROC'] + selected['F1'] + (selected['TPR01'] if selection == 'original' else 0)
    assert abs(expected_score - stopping.best_metric) < 1e-5, 'Restored checkpoint validation does not match selection'
    torch.save(model.cpu().state_dict(), folder / 'model.pt')
    del model
    return result


def reports(output, records, total):
    save_json(output / 'results.json', records)
    save_json(output / 'status.json', {'completed': len(records), 'total': total,
                                     'state': 'complete_waiting_for_user' if len(records) == total else 'running'})
    grouped = {}
    for row in records:
        key = row['dataset'], row['target'], json.dumps(row['setting'], sort_keys=True)
        grouped.setdefault(key, []).append(row)
    summary = []
    for (dataset, target, setting), rows in grouped.items():
        item = {'dataset': dataset, 'target': target, **json.loads(setting), 'n': len(rows),
                'seeds': ','.join(str(r['seed']) for r in rows)}
        for metric in ['AUROC', 'F1', 'TPR01']:
            vals = [r['validation']['all'][metric] for r in rows]
            item[metric] = statistics.mean(vals)
            item[metric + '_sd'] = statistics.stdev(vals) if len(vals) > 1 else None
        item['selection_score'] = statistics.mean(r['selection_score'] for r in rows)
        item['auc_f1'] = item['AUROC'] + item['F1']
        summary.append(item)
    summary.sort(key=lambda r: (r['dataset'], r['target'], -r['selection_score']))
    save_json(output / 'summary.json', summary)
    if summary:
        with (output / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)


def main(argv=None):
    args = arguments(argv)
    os.chdir(ROOT)
    output = args.output.resolve()
    inputs, grid = list(cases(args)), list(settings(args))
    total = len(inputs) * len(grid)
    plan = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'inputs': [identity for _, identity, _ in inputs], 'settings': grid, 'total': total,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((ROOT / 'sralt').rglob('*')) if p.suffix in {'.py', '.yaml'}},
            'test_evaluation': False, 'upstream_training': False}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    # Refuse accidental continuation with a different grid or data configuration.
    stable_plan = {k: v for k, v in plan.items() if k != 'args'}
    plan_path = output / 'plan.json'
    if plan_path.exists():
        prior = json.loads(plan_path.read_text())
        assert {k: v for k, v in prior.items() if k != 'args'} == stable_plan, 'Output belongs to a different sweep'
    else:
        save_json(plan_path, plan)
    import torch
    from hydra.utils import instantiate
    torch.set_num_threads(2)
    device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # Preflight all frozen inputs before running any candidate.
    for cfg, identity, name in inputs:
        folder = output / 'inputs' / name
        folder.mkdir(parents=True, exist_ok=True)
        instantiate(cfg.filter_warnings)
        if not (folder / 'input.pkl').exists():
            data = freeze_input(cfg, device, folder)
            save_json(folder / 'identity.json', {**identity, 'attack_data_repr': repr(data),
                       'train_positives': data.num_train_social_edges, 'validation_positives': data.num_val_social_edges})
            del data
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    if args.prepare_only:
        logging.info('Prepared %d frozen inputs. No training started.', len(inputs))
        return
    records = []
    for cfg, _, name in inputs:
        with (output / 'inputs' / name / 'input.pkl').open('rb') as handle:
            data = pickle.load(handle).to(device)
        for index, setting in enumerate(grid):
            folder = output / name / f'{index:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / 'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                handler = logging.FileHandler(folder / 'train.log')
                logging.getLogger().addHandler(handler)
                try:
                    logging.info('Trial %s/%02d: %s', name, index, setting)
                    result = train_trial(data, cfg, setting, folder, device, args.selection)
                    result.update(dataset=cfg.target.data.name.lower(), target=cfg.target.model.name.lower(), seed=cfg.seed)
                    save_json(result_path, result)
                finally:
                    logging.getLogger().removeHandler(handler)
                    handler.close()
            records.append(result)
            reports(output, records, total)
            logging.info('Progress %d/%d; val=%s', len(records), total, result['validation']['all'])
        del data
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    logging.info('Round complete: %d/%d. STOP. Await user confirmation before any further sweep.', len(records), total)


if __name__ == '__main__':
    main()

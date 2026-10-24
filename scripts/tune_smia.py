#!/usr/bin/env python3
"""Finite validation-only SMIA sweeps using the pipeline's data/model caches.

Default: frozen-shadow MLP sweep (128 trials). --sweep shadow scans SMIA
lr/decay with a fixed MLP by default. Targets must already be cached; shadow
training is allowed only in shadow mode. Never evaluates hidden test labels.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweep', choices=['attack', 'shadow'], default='attack')
    p.add_argument('--datasets', nargs='+', choices=['ciao', 'flickr', 'epinions', 'yelp'], default=['ciao', 'flickr'])
    p.add_argument('--targets', nargs='+', choices=['design', 'gdmsr'], default=['design', 'gdmsr'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    p.add_argument('--lrs', nargs='+', type=float, help='MLP rates; default attack grid or .05 in shadow mode')
    p.add_argument('--batches', nargs='+', type=int, help='MLP batches; default attack grid or 512 in shadow mode')
    p.add_argument('--norms', nargs='+', choices=['none', 'layer'], default=['none'])
    p.add_argument('--dropouts', nargs='+', type=float, default=[.5])
    p.add_argument('--baseline', action=argparse.BooleanOptionalAction, default=True,
                   help='Include original 50-epoch MLP control in attack mode')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--shadow-lrs', nargs='+', type=float, default=[.001, .003, .01])
    p.add_argument('--shadow-decays', nargs='+', type=float, default=[.001, .01])
    p.add_argument('--shadow-batches', nargs='+', type=int)
    p.add_argument('--shadow-embed-dims', nargs='+', type=int)
    p.add_argument('--shadow-layers', nargs='+', type=int)
    p.add_argument('--shadow-epochs', type=int)
    p.add_argument('--shadow-patience', type=int)
    p.add_argument('--selection', choices=['original', 'auc-f1'], default='original')
    p.add_argument('--output', type=Path, default=Path('runs/smia_tuning/round2'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--prepare-only', action='store_true', help='Prepare shared inputs/shadows without attack training')
    args = p.parse_args(argv)
    if args.lrs is None:
        args.lrs = [.05] if args.sweep == 'shadow' else [.001, .003, .01, .03, .05]
    if args.batches is None:
        args.batches = [512] if args.sweep == 'shadow' else [256, 512, 1024]
    for key in ('lrs', 'shadow_lrs'):
        if any(not 0 < value < float('inf') for value in getattr(args, key)):
            p.error(f'--{key.replace("_", "-")} must contain finite positive values')
    if any(not 0 <= value < float('inf') for value in args.shadow_decays):
        p.error('--shadow-decays must contain finite nonnegative values')
    for key in ('batches', 'shadow_batches', 'shadow_embed_dims', 'shadow_layers', 'epochs',
                'patience', 'shadow_epochs', 'shadow_patience'):
        values = getattr(args, key)
        if values is not None and any(value <= 0 for value in (values if isinstance(values, list) else [values])):
            p.error(f'--{key.replace("_", "-")} must be positive')
    if any(not 0 <= value < 1 for value in args.dropouts):
        p.error('--dropouts must be in [0, 1)')
    # Duplicate CLI values must not inflate seed counts or repeat cache work.
    for key, value in vars(args).items():
        if isinstance(value, list):
            setattr(args, key, list(dict.fromkeys(value)))
    return args


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def settings(args):
    if args.sweep == 'attack' and args.baseline:
        yield {'label': 'baseline', 'lr': .05, 'batch': 512, 'epochs': 50, 'patience': 5}
    for lr, batch, norm, dropout in itertools.product(args.lrs, args.batches, args.norms, args.dropouts):
        setting = {'label': 'grid', 'lr': lr, 'batch': batch, 'epochs': args.epochs, 'patience': args.patience}
        # Keep the original no-Norm configuration and cache key unchanged.
        if norm != 'none':
            setting['norm'] = norm
        if dropout != .5:
            setting['dropout'] = dropout
        yield setting


def shadow_settings(args):
    if args.sweep == 'attack':
        yield {}
        return
    fields = {'train.optimizer.lr': args.shadow_lrs, 'attribute.decay': args.shadow_decays,
              'train.train_batch_size': args.shadow_batches, 'attribute.embed_dim': args.shadow_embed_dims,
              'attribute.num_layers': args.shadow_layers}
    fields = {key: values for key, values in fields.items() if values is not None}
    for values in itertools.product(*fields.values()):
        setting = dict(zip(fields, values, strict=True))
        if args.shadow_epochs is not None:
            setting['train.num_epochs'] = args.shadow_epochs
        if args.shadow_patience is not None:
            setting['train.early_stopping.patience'] = args.shadow_patience
        yield setting


def cases(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    for dataset, target, seed in itertools.product(args.datasets, args.targets, args.seeds):
        with initialize_config_dir(config_dir=str(ROOT / 'sralt/configs'), version_base=None):
            base = compose(config_name='config', overrides=[f'target/data={dataset}', f'target/model={target}',
                                                           'shadow/model=smia', 'attack/model=mlp'])
        base.seed = seed
        for shadow_setting in shadow_settings(args):
            cfg = copy.deepcopy(base)
            for key, value in shadow_setting.items():
                OmegaConf.update(cfg.shadow.model, key, value)
            identity = {'dataset': dataset, 'target': target, 'seed': seed, 'deterministic': cfg.deterministic,
                        'namespace': os.environ.get('SRALT_CACHE_VERSION', 'default'),
                        'target_config': OmegaConf.to_container(cfg.target, resolve=True),
                        'shadow_config': OmegaConf.to_container(cfg.shadow, resolve=True),
                        'attack_config': OmegaConf.to_container(cfg.attack, resolve=True)}
            name = f'{dataset}-{target}-{seed}'
            if args.sweep == 'shadow':
                identity['shadow_setting'] = shadow_setting
                digest = hashlib.sha256(json.dumps(identity['shadow_config'], sort_keys=True).encode()).hexdigest()[:12]
                name += f'-shadow-{digest}'
            yield cfg, identity, name


def freeze_input(cfg, device, folder, allow_shadow_training=False):
    """Resolve shared pipeline caches; never write private data or weight copies."""
    import torch
    from hydra.utils import instantiate
    from torch.utils.tensorboard import SummaryWriter
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
    writer = SummaryWriter(str(folder / 'shadow-tensorboard')) if allow_shadow_training else None
    try:
        shadow = load_report_model('shadow', load_shadow_model, allow_shadow_training, **common, device=device,
                                   config=cfg.shadow.model, data=sd, writer=writer)
    finally:
        if writer is not None:
            writer.close()
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
    return data


def attack_config(cfg, setting, selection):
    from omegaconf import OmegaConf
    config = copy.deepcopy(cfg.attack.model)
    config.train.optimizer.lr = setting['lr']
    config.train.train_batch_size = setting['batch']
    config.train.num_epochs = setting['epochs']
    config.train.early_stopping.patience = setting['patience']
    if setting.get('norm', 'none') != 'none':
        OmegaConf.update(config.attribute, 'norm', setting['norm'], force_add=True)
    if 'dropout' in setting:
        config.attribute.dropout = setting['dropout']
    if selection == 'auc-f1':
        config.train.metrics = config.train.metrics[:2]
    return config


def train_trial(data, cfg, setting, folder, device, selection):
    """Use the shared attack loader, then score its restored checkpoint on validation."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.tensorboard import SummaryWriter
    from sralt.attack import load_attack_model
    from sralt.attack.data import EvalAttackDataset
    from sralt.utils import set_seed, DataLoader, head_tail_split, compute_attack_metrics, model_fingerprint
    from sralt.utils.model import tensor_fingerprint

    config = attack_config(cfg, setting, selection)
    OmegaConf.save(config, folder / 'attack.yaml', resolve=True)
    common = dict(seed=cfg.seed, deterministic=cfg.deterministic)
    start = time.monotonic()
    writer = SummaryWriter(str(folder / 'tensorboard'))
    try:
        model = load_attack_model(**common, device=device, config=config, data=data,
                                  tail_ratio=cfg.attack.tail_ratio, writer=writer)
    finally:
        writer.close()
    # Same seed and candidates as at the original train_model boundary.
    set_seed(**common)
    validation = EvalAttackDataset(data, split='val')
    val_fingerprint = tensor_fingerprint(validation.edge_index)
    model.eval()
    head, tail = head_tail_split(data.num_users, data.target_social_edge_index, cfg.attack.tail_ratio)
    with torch.no_grad():
        values = compute_attack_metrics(
            device=device, predictor=lambda u, v: model.predict(torch.stack([u, v]), data.user_features),
            metrics=instantiate(cfg.attack.model.train.metrics),
            eval_loader=DataLoader(validation, batch_size=config.train.val_batch_size, shuffle=False),
            head_indices=head, tail_indices=tail)
    names = ['AUROC', 'F1', 'TPR01']
    validation_metrics = {g: dict(zip(names, v, strict=True))
                          for g, v in zip(['all', 'head', 'tail'], values, strict=True)}
    selected = validation_metrics['all']
    score = selected['AUROC'] + selected['F1'] + (selected['TPR01'] if selection == 'original' else 0)
    training = getattr(model, 'training_summary', {})
    if training:
        assert abs(score - training['selection_score']) < 1e-5, 'Restored checkpoint validation does not match selection'
    result = {'setting': setting, 'selection': selection, 'best_epoch': training.get('best_epoch'),
              'stop_epoch': training.get('stop_epoch'), 'selection_score': score,
              'training_summary_available': bool(training),
              'elapsed_seconds': time.monotonic() - start, 'validation_pairs': val_fingerprint,
              'validation': validation_metrics, 'model_fingerprint': model_fingerprint(model),
              'model_cache_directory': f'.cache/attack/model/{config.name}'}
    del model
    return result


def reports(output, records, total):
    save_json(output / 'results.json', records)
    save_json(output / 'status.json', {'completed': len(records), 'total': total,
                                     'state': 'complete_waiting_for_user' if len(records) == total else 'running'})
    grouped = {}
    for row in records:
        key = (row['dataset'], row['target'], json.dumps(row['setting'], sort_keys=True),
               json.dumps(row.get('shadow_setting', {}), sort_keys=True), row.get('selection', 'original'))
        grouped.setdefault(key, []).append(row)
    summary = []
    for (dataset, target, setting, shadow, selection), rows in grouped.items():
        item = {'dataset': dataset, 'target': target, **json.loads(setting), 'n': len(rows),
                'seeds': ','.join(str(r['seed']) for r in rows), 'selection': selection,
                'shadow_setting': json.loads(shadow)}
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
        fields = list(dict.fromkeys(k for row in summary for k in row))
        with (output / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in summary:
                writer.writerow({**row, 'shadow_setting': json.dumps(row['shadow_setting'], sort_keys=True)})


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
            'test_evaluation': False, 'target_training': False,
            'shadow_training': args.sweep == 'shadow', 'selection': args.selection,
            'model_storage': 'pipeline_disk_cache'}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if os.environ.get('SRALT_CACHE_REFRESH') == '1':
        raise ValueError('Unset SRALT_CACHE_REFRESH for sweeps so shared model/data caches remain reusable')
    output.mkdir(parents=True, exist_ok=True)
    # Refuse accidental continuation with a different grid or data configuration.
    stable_plan = {k: v for k, v in plan.items() if k != 'args'}
    plan_path = output / 'plan.json'
    if plan_path.exists():
        prior = json.loads(plan_path.read_text())
        if {k: v for k, v in prior.items() if k != 'args'} != stable_plan:
            raise ValueError('Output belongs to a different sweep; use a new --output directory')
    else:
        save_json(plan_path, plan)
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(2)
    device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    records = []
    for cfg, identity, name in inputs:
        input_folder = output / 'inputs' / name
        input_folder.mkdir(parents=True, exist_ok=True)
        data = None
        pending = any(not (output / name / f'{i:02d}' / 'result.json').exists() for i in range(len(grid)))
        if args.prepare_only or pending:
            handler = logging.FileHandler(input_folder / 'prepare.log')
            logging.getLogger().addHandler(handler)
            try:
                data = freeze_input(cfg, device, input_folder, allow_shadow_training=args.sweep == 'shadow')
                save_json(input_folder / 'identity.json', {**identity, 'attack_data_repr': repr(data),
                          'train_positives': data.num_train_social_edges, 'validation_positives': data.num_val_social_edges})
                OmegaConf.save(cfg.shadow.model, input_folder / 'shadow.yaml', resolve=True)
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()
        if not args.prepare_only:
            if data is not None:
                data.to(device)
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
                        result.update(dataset=identity['dataset'], target=identity['target'], seed=cfg.seed)
                        if 'shadow_setting' in identity:
                            result['shadow_setting'] = identity['shadow_setting']
                        result['shadow_model_repr'] = data.shadow_model_repr
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
    if args.prepare_only:
        logging.info('Prepared %d shared inputs. No attack training started.', len(inputs))
    else:
        logging.info('Round complete: %d/%d. No further sweep started.', len(records), total)


if __name__ == '__main__':
    main()

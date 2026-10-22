#!/usr/bin/env python3
"""Train EGCL targets and compare hyperparameters without shadow/attack training.

Defaults: Flickr, seeds 42–46, learning rates .01/.003/.001, patience 50.
Run from any directory using the project's Python environment.
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
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=['flickr'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--grid', nargs='+', default=['train.optimizer.lr=0.01,0.003,0.001'],
                   help='Cartesian grid of model-relative keys; replaces the default learning-rate grid')
    p.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE',
                   help='Fixed model-relative overrides, e.g. train.early_stopping.patience=100')
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--output', type=Path, default=Path('runs/egcl_tuning'))
    p.add_argument('--cpu', action='store_true', help='Explicitly use CPU')
    p.add_argument('--dry-run', action='store_true', help='Print trial configurations without loading data/models')
    p.add_argument('--diagnostics', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--test-metrics', action='store_true', help='Also report test metrics; selection always uses validation')
    p.add_argument('--num-recommendations', type=int, default=30)
    p.add_argument('--query-batch-size', type=int, default=256)
    return p.parse_args(argv)


def make_cases(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    grids = [entry.split('=', 1) for entry in args.grid]
    for dataset in dict.fromkeys(args.datasets):
        with initialize_config_dir(config_dir=str(ROOT / 'sralt/configs'), version_base=None):
            base = compose(config_name='config', overrides=[f'target/data={dataset}', 'target/model=egcl', '~shadow', '~attack'])
        base.target.model.train.early_stopping.patience = args.patience
        for entry in args.set:
            key, value = entry.split('=', 1)
            OmegaConf.update(base.target.model, key, OmegaConf.from_dotlist([f'value={value}']).value)
        for values in itertools.product(*(value.split(',') for _, value in grids)):
            for seed in dict.fromkeys(args.seeds):
                cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
                cfg.seed = seed
                OmegaConf.set_struct(cfg.target.model, True)
                for (key, _), value in zip(grids, values):
                    OmegaConf.update(cfg.target.model, key, OmegaConf.from_dotlist([f'value={value}']).value)
                # Resolve only the target; the unused writer contains Hydra runtime interpolation.
                identity = {'dataset': dataset, 'seed': seed, 'deterministic': cfg.deterministic,
                            'target': OmegaConf.to_container(cfg.target, resolve=True),
                            'namespace': os.environ.get('SRALT_CACHE_VERSION', 'default')}
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
                yield cfg, identity, f'{dataset}-{seed}-{digest}'


def metric_names(metrics):
    return [f"{m._target_.split('.')[-1].removeprefix('LinkPred')}@{m.k}" for m in metrics]


def validate(model, data, config, seed, deterministic):
    """Recreate training's fixed validation candidates and evaluate the restored checkpoint."""
    from hydra.utils import instantiate
    from torch_geometric.utils import to_undirected
    from sralt.target.data import EvalTargetDataset
    from sralt.utils import DataLoader, compute_recommendation_metrics, set_seed

    set_seed(seed, deterministic=deterministic)
    dataset = EvalTargetDataset(data, split='val', num_negatives=config.val_num_negatives)
    loader = DataLoader(dataset, batch_size=config.val_batch_size, shuffle=False)
    model.eval()
    values = compute_recommendation_metrics(
        model.get_embedding(data.social_edge_index, to_undirected(data.behavioral_train_edge_index)),
        instantiate(config.metrics), loader,
    )
    return dict(zip(metric_names(config.metrics), values, strict=True))


def embedding_stats(embeds):
    import torch
    import torch.nn.functional as F

    x = embeds.float()
    unit = F.normalize(x, dim=1)
    n = len(x)
    energy = x.square().sum()
    return {
        'user_mean_cosine': float((unit.sum(0).square().sum() - unit.square().sum()) / (n * (n - 1))),
        'user_pc1_energy': float(torch.linalg.eigvalsh(x.T @ x)[-1] / energy),
    }


def diagnose(model, data, k, batch_size):
    import torch
    from torch_geometric.utils import to_undirected

    with torch.no_grad():
        model.eval()
        recommendations = model.recommend(data=data, num_recommendations=k, batch_size=batch_size)
        counts = torch.bincount(recommendations[1] - data.num_users, minlength=data.num_items)
        result = {
            'recommended_unique_items': int((counts > 0).sum()),
            'item_coverage': float((counts > 0).float().mean()),
            'most_frequent_k_share': float(counts.topk(k).values.sum() / counts.sum()),
        }
        embeds = model.get_embedding(data.social_edge_index, to_undirected(data.behavioral_edge_index))
        result.update(embedding_stats(embeds[:data.num_users]))
        return result


def write_reports(output, records):
    """Regenerate reports from this invocation's completed cases, never mix other sweeps."""
    def save_json(path, value):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    save_json(output / 'results.json', records)
    grouped = {}
    for record in records:
        identity = record['identity']
        key = (identity['dataset'], json.dumps(identity['target']['model'], sort_keys=True))
        grouped.setdefault(key, []).append(record)
    summary = []
    for (dataset, model_json), rows in grouped.items():
        values = [r['validation_score'] for r in rows]
        row = {'dataset': dataset, 'seeds': ','.join(str(r['identity']['seed']) for r in rows),
               'n': len(rows), 'validation_score_mean': statistics.mean(values),
               'validation_score_std': statistics.stdev(values) if len(values) > 1 else None,
               'model_config': json.loads(model_json)}
        for section in ('validation', 'test', 'diagnostics'):
            for metric in rows[0].get(section, {}):
                row[f'{section}/{metric}'] = statistics.mean(r[section][metric] for r in rows)
        summary.append(row)
    summary.sort(key=lambda r: (r['dataset'], -r['validation_score_mean']))
    save_json(output / 'summary.json', summary)
    if summary:
        fields = list(dict.fromkeys(k for row in summary for k in row))
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in summary:
                writer.writerow({**row, 'model_config': json.dumps(row['model_config'], sort_keys=True)})


def main(argv=None):
    args = parse_args(argv)
    # Match pipeline cache/data paths even when launched outside the project.
    args.output = args.output.resolve()
    os.chdir(ROOT)
    cases = list(make_cases(args))
    if args.dry_run:
        for _, identity, name in cases:
            print(name, json.dumps(identity, ensure_ascii=False))
        print(f'{len(cases)} target-only trials')
        return

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.tensorboard import SummaryWriter
    from sralt.target import load_target_data, load_target_model, test_target_model

    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    records = []
    for index, (cfg, identity, name) in enumerate(cases, 1):
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        handler = logging.FileHandler(folder / 'train.log')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logging.getLogger().addHandler(handler)
        writer = SummaryWriter(str(folder / 'tensorboard'))
        try:
            instantiate(cfg.filter_warnings)
            device = torch.device('cpu') if args.cpu else instantiate(cfg.device)
            logging.info('Trial %d/%d: %s', index, len(cases), name)
            OmegaConf.save(cfg.target, folder / 'target.yaml', resolve=True)
            common = dict(seed=cfg.seed, deterministic=cfg.deterministic)
            data = load_target_data(**common, config=cfg.target.data).to(device)
            model = load_target_model(**common, device=device, config=cfg.target.model, data=data, writer=writer)
            with torch.no_grad():
                validation = validate(model, data, cfg.target.model.train, **common)
                record = {'trial': name, 'identity': identity, 'validation': validation,
                          'validation_score': sum(validation.values())}
                if args.test_metrics:
                    values = test_target_model(**common, device=device, model=model, data=data, config=cfg.target.test)
                    record['test'] = dict(zip(metric_names(cfg.target.test.metrics), values, strict=True))
                if args.diagnostics:
                    record['diagnostics'] = diagnose(model, data, args.num_recommendations, args.query_batch_size)
                    record['diagnostic_k'] = args.num_recommendations
            records.append(record)
            write_reports(args.output, records)
            logging.info('Completed %s: %s', name, record)
            del model, data
            gc.collect()
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        finally:
            writer.close()
            logging.getLogger().removeHandler(handler)
            handler.close()
    logging.info('Finished. Compare validation scores in %s', args.output / 'summary.csv')


if __name__ == '__main__':
    main()

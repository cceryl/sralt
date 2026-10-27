#!/usr/bin/env python3
"""CPU-only statistical relation probes on existing restored DESIGN queries.

Evaluate IDF intersection, raw overlap and existing SMIA/MLP checkpoints on the
same independent diagnostic pairs. Never train/query target or shadow models.
"""
import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from design_query_distribution import ROOT, CPUUnpickler, digest
from sralt.utils import head_tail_split

GROUPS = ('All', 'Head', 'Tail', 'TT', 'isolated_Tail')
METHODS = ('idf_intersection', 'raw_intersection', 'public_degree', 'smia_mlp')
METRICS = ('AUROC', 'F1', 'TPR@1%')


def load(path):
    with path.open('rb') as handle:
        return CPUUnpickler(handle).load()


def tensor_hash(tensor):
    import hashlib
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(tensor.shape), str(tensor.dtype))).encode()
                          + tensor.numpy().tobytes()).hexdigest()


def logged_cache(run, stage, kind):
    text = '\n'.join(path.read_text() for path in sorted(run.glob('pipeline*.log')))
    matches = set(re.findall(rf'\.cache/{stage}/{kind}/[^"\s]+\.pkl', text))
    if len(matches) != 1:
        raise ValueError(f'{run}: expected one {stage}/{kind} reference, got {matches}')
    return ROOT / next(iter(matches))


def pair_ids(edges, n):
    edges = edges.detach().cpu().numpy()
    u, v = np.minimum(edges[0], edges[1]), np.maximum(edges[0], edges[1])
    return np.unique(u * n + v)


def sample_negatives(n, count, excluded, rng):
    """Legacy diagnostic RNG/rule: uniform unique unordered true nonedges."""
    excluded = set(map(int, excluded))
    found = set()
    while len(found) < count:
        a, b = rng.integers(0, n, size=(2, max(1024, 2 * (count - len(found)))))
        for u, v in zip(np.minimum(a, b), np.maximum(a, b)):
            key = int(u * n + v)
            if u != v and key not in excluded:
                found.add(key)
    ids = np.array(sorted(found), dtype=np.int64)
    rng.shuffle(ids)
    ids = ids[:count]
    return np.stack([ids // n, ids % n])


def pairdot(matrix, pairs, batch_size=8192):
    return np.concatenate([np.asarray(matrix[pairs[0, j:j + batch_size]].multiply(
        matrix[pairs[1, j:j + batch_size]]).sum(axis=1)).ravel()
        for j in range(0, pairs.shape[1], batch_size)])


def choose_threshold(scores, labels):
    order = np.argsort(-scores, kind='stable')
    scores, labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    tp = labels.cumsum()[ends]
    f1 = 2 * tp / (ends + 1 + labels.sum())
    index = ends[np.argmax(f1)]  # All score ties are included; prefer the higher threshold.
    threshold = ((scores[index] + scores[index + 1]) / 2 if index + 1 < len(scores)
                 else np.nextafter(scores[index], -np.inf))
    return float(threshold), float(f1.max())


def metrics(labels, scores, threshold, weight):
    selected = weight > 0
    y, s, w = labels[selected], scores[selected], weight[selected]
    positive, negative = int(w[y == 1].sum()), int(w[y == 0].sum())
    result = dict(samples=int(w.sum()), positives=positive, negatives=negative,
                  AUROC=None, F1=None, **{'TPR@1%': None})
    if not len(y):
        return result
    predicted = s > threshold
    tp = w[(y == 1) & predicted].sum()
    denominator = positive + w[predicted].sum()
    result['F1'] = float(2 * tp / denominator) if denominator else 0.
    if positive and negative:
        result['AUROC'] = float(roc_auc_score(y, s, sample_weight=w))
        fpr, tpr, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)
        fpr, indices = np.unique(fpr[::-1], return_index=True)
        tpr = tpr[::-1][indices]  # Last/highest TPR on each FPR plateau.
        result['TPR@1%'] = float(np.interp(.01, fpr, tpr))
    return result


def evaluate(row, args, output):
    seed, case = row['seed'], row['case']
    source = row['sources'][0]
    run = Path(source['run'])
    if digest(Path(source['config_path'])) != source['config_sha256']:
        raise ValueError(f'{case}/{seed}: source config changed after query audit')
    for path, expected in row['cache_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'{case}/{seed}: query cache changed after query audit')
    shadow = load(Path(source['cache']))
    data_path, model_path = logged_cache(run, 'attack', 'data'), logged_cache(run, 'attack', 'model')
    data, model = load(data_path).to(torch.device('cpu')), load(model_path).to(torch.device('cpu'))
    model.eval()
    if type(model).__name__ != 'MLP' or shadow.target_model_name != 'DESIGN':
        raise ValueError('Expected a restored DESIGN -> SMIA -> MLP source')
    if data.shadow_data_repr != repr(shadow) or data.num_users != row['users']:
        raise ValueError('Query/attack data identities differ')
    if data.target_data_repr != shadow.target_data_repr:
        raise ValueError('Target data identities differ')
    if not torch.equal(data.train_social_edge_index, shadow.attack_train_social_edge_index):
        raise ValueError('Attack/query training splits differ')
    if not torch.equal(data.val_social_edge_index, shadow.attack_val_social_edge_index):
        raise ValueError('Attack/query validation splits differ')
    config = source['config']
    # Full source configuration is preserved separately for attack revision and tail ratio.
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(source['config_path'])
    if (cfg.shadow.model.revision != 'legacy-best10-v2' or
            cfg.attack.model.revision != 'legacy-auc-f1-balanced-v3'):
        raise ValueError('Source is not the accepted restored SMIA/MLP revision')
    n, items = row['users'], row['items']
    edges = shadow.behavioral_edge_index.cpu().numpy()
    rec = sp.csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1] - n)), shape=(n, items))
    if np.any(rec.data != 1) or np.any(np.asarray(rec.sum(axis=1)).ravel() != 30):
        raise ValueError('Expected 30 distinct queried items for every user')
    counts = np.asarray(rec.sum(axis=0)).ravel()
    idf = np.log((n + 1) / (counts + 1))
    weighted = rec.multiply(np.sqrt(idf)).tocsr()
    degree = np.bincount(data.shadow_social_edge_index[0].numpy(), minlength=n)
    def probe_scores(pairs):
        return dict(idf_intersection=pairdot(weighted, pairs), raw_intersection=pairdot(rec, pairs),
                    public_degree=np.log1p(degree[pairs[0]]) + np.log1p(degree[pairs[1]]))
    # Validation accesses only public relations. Hidden graph is accessed below, after thresholds freeze.
    val_pos = data.val_social_edge_index.numpy()
    val_rng = np.random.default_rng(seed + 2018)
    val_neg = sample_negatives(n, val_pos.shape[1], pair_ids(data.public_social_edge_index, n), val_rng)
    val_pairs = np.concatenate([val_pos, val_neg], axis=1)
    val_labels = np.r_[np.ones(val_pos.shape[1]), np.zeros(val_neg.shape[1])]
    thresholds, validation = {}, {}
    for name, scores in probe_scores(val_pairs).items():
        thresholds[name], validation[name] = choose_threshold(scores, val_labels)
    thresholds['smia_mlp'] = .5
    rng = np.random.default_rng(seed + 1018)
    test_ids = pair_ids(data.test_social_edge_index, n)
    test_pos = np.stack([test_ids // n, test_ids % n])
    test_neg = sample_negatives(n, len(test_ids), pair_ids(data.target_social_edge_index, n), rng)
    pairs = np.concatenate([test_pos, test_neg], axis=1)
    labels = np.r_[np.ones(len(test_ids)), np.zeros(len(test_ids))].astype(np.int64)
    scores = probe_scores(pairs)
    with torch.no_grad():
        scores['smia_mlp'] = np.concatenate([
            model.predict(torch.from_numpy(pairs[:, start:start + args.batch_size]), data.user_features).numpy()
            for start in range(0, pairs.shape[1], args.batch_size)])
    head, tail = head_tail_split(n, data.target_social_edge_index, float(cfg.attack.tail_ratio))
    is_head, is_tail = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    is_head[head.numpy()], is_tail[tail.numpy()] = True, True
    head_count = is_head[pairs[0]].astype(int) + is_head[pairs[1]].astype(int)
    isolated = is_tail & (degree == 0)
    weights = dict(All=np.ones(pairs.shape[1], dtype=int), Head=head_count, Tail=2 - head_count,
                   TT=(head_count == 0).astype(int),
                   isolated_Tail=isolated[pairs[0]].astype(int) + isolated[pairs[1]].astype(int))
    report = {method: {group: metrics(labels, scores[method], thresholds[method], weights[group])
                        for group in GROUPS} for method in METHODS}
    provenance = dict(case=case, seed=seed, source=source, query_sha256=row['cache_sha256'],
                      attack_data=str(data_path), attack_data_sha256=digest(data_path),
                      attack_model=str(model_path), attack_model_sha256=digest(model_path),
                      attack_revision=str(cfg.attack.model.revision), shadow_revision=str(cfg.shadow.model.revision),
                      tail_ratio=float(cfg.attack.tail_ratio), threads=args.threads,
                      formula='sum(log((N+1)/(item_user_coverage+1))) over shared queried items',
                      thresholds=thresholds, validation_f1=validation,
                      validation_rng=seed + 2018, test_rng=seed + 1018,
                      validation_negative_rule='uniform unique unordered pairs excluding all public edges',
                      test_negative_rule='uniform unique unordered true nonedges; equal to all hidden positives',
                      score_diagnostics={method: dict(positive_mean=float(scores[method][labels == 1].mean()),
                                                      negative_mean=float(scores[method][labels == 0].mean()))
                                         for method in METHODS},
                      torch_version=torch.__version__, numpy_version=np.__version__)
    directory = output / row['dataset'].lower() / f'seed-{seed}'
    directory.mkdir(parents=True)
    np.savez_compressed(directory / 'pairs_scores.npz', validation_pairs=val_pairs,
                        validation_labels=val_labels, pairs=pairs, labels=labels, **scores)
    (directory / 'results.json').write_text(json.dumps(dict(provenance=provenance, groups=report), indent=2, allow_nan=False) + '\n')
    print(f'{case} seed{seed} ' + ' '.join(f'{name}:TailAUC={report[name]["Tail"]["AUROC"]:.4f}'
                                        for name in ('idf_intersection', 'smia_mlp')), flush=True)
    return dict(case=case, dataset=row['dataset'], seed=seed, provenance=provenance, groups=report)


def aggregate(rows):
    result = []
    for dataset in sorted({r['dataset'] for r in rows}):
        cases = [r for r in rows if r['dataset'] == dataset]
        for method in METHODS:
            for group in GROUPS:
                record = dict(dataset=dataset, method=method, group=group, seeds=[r['seed'] for r in cases])
                for metric in METRICS:
                    values = [r['groups'][method][group][metric] for r in cases
                              if r['groups'][method][group][metric] is not None]
                    record[metric] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                                          sample_sd=statistics.stdev(values) if len(values) > 1 else None)
                result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=ROOT / 'runs/tail-proxy/design-query-distribution-2026-10-27-v2/statistics.json')
    parser.add_argument('--datasets', nargs='+', default=['Ciao', 'Flickr'])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(42, 47)))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory must not already exist')
    if args.threads < 1 or args.batch_size < 1:
        parser.error('threads and batch-size must be positive')
    torch.set_num_threads(args.threads)
    payload = json.loads(args.queries.read_text())
    selected = [row for row in payload['rows'] if row['dataset'] in args.datasets and row['seed'] in args.seeds]
    if len(selected) != len(args.datasets) * len(args.seeds):
        parser.error('Query manifest does not cover exactly the requested dataset/seed cases')
    args.output.mkdir(parents=True)
    rows, errors = [], []
    (args.output / 'request.json').write_text(json.dumps(dict(queries=str(args.queries), query_manifest_sha256=digest(args.queries),
                                                           datasets=args.datasets, seeds=args.seeds), indent=2) + '\n')
    for row in selected:
        try:
            rows.append(evaluate(row, args, args.output))
        except Exception as error:
            errors.append(dict(case=row['case'], seed=row['seed'], error=repr(error)))
            (args.output / 'errors.json').write_text(json.dumps(errors, indent=2) + '\n')
            raise
        (args.output / 'summary.json').write_text(json.dumps(dict(cases=rows, summary=aggregate(rows)), indent=2, allow_nan=False) + '\n')
    records = [dict(dataset=r['dataset'], seed=r['seed'], method=method, group=group, **r['groups'][method][group])
               for r in rows for method in METHODS for group in GROUPS]
    with (args.output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'Completed {len(rows)} cases: {args.output}', flush=True)


if __name__ == '__main__':
    main()

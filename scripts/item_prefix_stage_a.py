#!/usr/bin/env python3
"""Fit item RVQ on existing frozen Shadow caches and evaluate prefix equivalence.

CPU research experiment; never trains/queries target, Shadow or Attack models.
Only new RVQ models are fitted, through the existing disk_cache mechanism.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sralt-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
import sklearn
import torch
from threadpoolctl import threadpool_limits

from design_query_distribution import CPUUnpickler
from design_idf_probe import choose_threshold, metrics
from sralt.attack.data import EvalAttackDataset
from sralt.shadow.model.item_rvq import load_item_rvq_model
from sralt.utils import head_tail_split, set_seed, model_fingerprint
from sralt.utils.model import tensor_fingerprint

REVISION = 'item-prefix-stage-a-v1'
GROUPS = ('All', 'Head', 'Tail', 'TT', 'isolated_Tail')
METRICS = ('AUROC', 'F1', 'TPR@1%')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_cpu(path):
    with Path(path).open('rb') as handle:
        return CPUUnpickler(handle).load()


def source_record(target, dataset, seed):
    directory = ROOT / f'runs/tail-proxy/stage1-{target}-2026-10-27/{target}/{dataset}/seed-{seed}'
    log = (directory / 'console.log').read_text()
    def paths(pattern):
        return list(dict.fromkeys(re.findall(pattern, log)))
    queries = paths(r'"(\.cache/shadow/data/[^"\s]+\.pkl)"')
    models = paths(r'"(\.cache/shadow/model/TailProxyShadow/[^"\s]+\.pkl)"')
    data = paths(r'"(\.cache/attack/data/[^"\s]+/TailProxyShadow/[^"\s]+\.pkl)"')
    if len(queries) != 1 or len(models) != 2 or len(data) != 2:
        raise ValueError(f'{directory}: expected one query and ordered e10/e300 cache references')
    if target == 'design':
        old = json.loads((directory / 'e300-native/metrics.json').read_text())
        config = old['config']['shadow']['model']
        target_config = old['config']['target']['model']
        evaluation_id = None
    else:
        old = json.loads((directory / 'metrics.json').read_text())
        config = old['shadow']['300']['config']
        target_config = old['config']['target']['model']
        evaluation_id = old['evaluation_id']
    if config['revision'] != 'tail-proxy-shadow-e300-v2' or config['train']['num_epochs'] != 300:
        raise ValueError('Unexpected frozen Shadow source revision/budget')
    for path in [queries[0], models[-1], data[-1]]:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    return dict(target=target, dataset=dataset, seed=seed, run=str(directory.relative_to(ROOT)),
                query_cache=queries[0], shadow_cache=models[-1], attack_data_cache=data[-1],
                shadow_config=config, target_revision=target_config['revision'],
                prior_evaluation_id=evaluation_id)


def incidence(edges, num_users, num_items):
    edge = edges.cpu().numpy()
    matrix = sp.csr_matrix((np.ones(edge.shape[1]), (edge[0], edge[1] - num_users)),
                           shape=(num_users, num_items))
    matrix.sum_duplicates()
    matrix.data.fill(1.)
    return matrix


def prefix_incidence(rec, mapping, num_prefixes):
    coo = rec.tocoo()
    matrix = sp.csr_matrix((np.ones(coo.nnz), (coo.row, mapping[coo.col])),
                           shape=(rec.shape[0], num_prefixes))
    matrix.sum_duplicates()
    matrix.data.fill(1.)
    return matrix


def score_pairs(matrix, pairs, batch_size=8192):
    counts = np.asarray(matrix.sum(0)).ravel()
    information = np.log((matrix.shape[0] + 1.) / (counts + 1.))
    weighted = matrix.multiply(information).tocsr()
    mass = np.asarray(weighted.sum(1)).ravel()
    count, overlap = [], []
    for start in range(0, pairs.shape[1], batch_size):
        left, right = pairs[:, start:start + batch_size]
        count.append(np.asarray(matrix[left].multiply(matrix[right]).sum(1)).ravel())
        overlap.append(np.asarray(weighted[left].multiply(matrix[right]).sum(1)).ravel())
    count, overlap = np.concatenate(count), np.concatenate(overlap)
    union = mass[pairs[0]] + mass[pairs[1]] - overlap
    return dict(count=count, overlap=overlap, jaccard=overlap / np.maximum(union, 1e-12))


def comparison(reference, candidate):
    delta = np.abs(reference - candidate)
    # Canonicalize floating summation noise before rank/ROC tie comparisons.
    x, y = np.round(reference, 10), np.round(candidate, 10)
    variable = np.ptp(x) > 0 and np.ptp(y) > 0
    return dict(mae=float(delta.mean()), max_error=float(delta.max()),
                normalized_mae=float(delta.mean() / max(np.abs(reference).mean(), 1e-12)),
                exact_ratio=float((reference == candidate).mean()),
                close_ratio=float(np.isclose(reference, candidate, rtol=1e-10, atol=1e-10).mean()),
                pearson=float(np.corrcoef(x, y)[0, 1]) if variable else None,
                spearman=float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]) if variable else None,
                reference_unique_scores=int(np.unique(x).size), candidate_unique_scores=int(np.unique(y).size))


def distribution(values):
    values = np.asarray(values)
    return dict(min=int(values.min()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), max=int(values.max()), mean=float(values.mean()))


def layer_diagnostics(codes, rec, mse):
    _, mapping, buckets = np.unique(codes, axis=0, return_inverse=True, return_counts=True)
    matrix = prefix_incidence(rec, mapping, len(buckets))
    freq = np.asarray(matrix.sum(0)).ravel()
    item_freq = np.asarray(rec.sum(0)).ravel()
    active = item_freq > 0
    # Quantized catalogue here is the union of full-query items, but occupancy
    # and collision statistics are restricted to the selected recommendation view.
    active_buckets = np.bincount(mapping[active], minlength=len(buckets))
    positive_buckets = active_buckets[active_buckets > 0]
    collision_pairs = int(np.sum(positive_buckets * (positive_buckets - 1) // 2))
    n = int(active.sum())
    delta = freq[mapping[active]] - item_freq[active]
    info = np.log((rec.shape[0] + 1.) / (freq[mapping[active]] + 1.))
    item_info = np.log((rec.shape[0] + 1.) / (item_freq[active] + 1.))
    probability = positive_buckets / n
    active_freq = freq[freq > 0]
    freq_probability = active_freq / active_freq.sum()
    diagnostic = dict(items=n, prefixes=len(positive_buckets),
                      unique_item_ratio=float((active_buckets[mapping[active]] == 1).mean()),
                      distinct_path_ratio=len(positive_buckets) / n,
                      collision_pairs=collision_pairs, collision_pair_ratio=collision_pairs / max(n * (n - 1) / 2, 1),
                      bucket_sizes=distribution(positive_buckets), user_frequency=distribution(active_freq),
                      user_prefix_edges=int(matrix.nnz), item_perplexity=float(np.exp(-np.sum(probability * np.log(probability)))),
                      user_activation_perplexity=float(np.exp(-np.sum(freq_probability * np.log(freq_probability)))),
                      max_item_fraction=float(probability.max()),
                      frequency=dict(equal_ratio=float((delta == 0).mean()), mae=float(np.abs(delta).mean()),
                                     max_error=float(np.abs(delta).max()), relative_mae=float((np.abs(delta) / item_freq[active]).mean()),
                                     information_mae=float(np.abs(info - item_info).mean()),
                                     exposure_weighted_information_mae=float(np.average(np.abs(info - item_info), weights=item_freq[active]))),
                      reconstruction_mse=float(mse[active].mean()))
    return matrix, diagnostic


def evaluation_protocol(data, seed):
    protocol, identity = {}, {}
    for split in ('val', 'test'):
        set_seed(seed, deterministic=False)
        sample = EvalAttackDataset(data, split=split)
        protocol[split] = dict(pairs=sample.edge_index.numpy(), labels=sample.labels.numpy())
        identity[split] = tensor_fingerprint(sample.edge_index)
    head, tail = head_tail_split(data.num_users, data.target_social_edge_index, .8)
    identity.update(head=tensor_fingerprint(head), tail=tensor_fingerprint(tail))
    pair = protocol['test']['pairs']
    head_mask = np.zeros(data.num_users, dtype=bool)
    head_mask[head.numpy()] = True
    head_count = head_mask[pair[0]].astype(int) + head_mask[pair[1]].astype(int)
    degree = np.bincount(data.shadow_social_edge_index[0].numpy(), minlength=data.num_users)
    isolated = (~head_mask) & (degree == 0)
    weights = dict(All=np.ones(pair.shape[1], dtype=int), Head=head_count, Tail=2 - head_count,
                   TT=(head_count == 0).astype(int),
                   isolated_Tail=isolated[pair[0]].astype(int) + isolated[pair[1]].astype(int))
    return protocol, weights, identity


def relation_report(scores, validation_scores, protocol, weights):
    score, validation = np.round(scores, 10), np.round(validation_scores, 10)
    threshold, val_f1 = choose_threshold(validation, protocol['val']['labels'])
    return dict(threshold=threshold, validation_f1=val_f1,
                groups={group: metrics(protocol['test']['labels'], score, threshold, weights[group]) for group in GROUPS})


def run_case(source, args):
    started = time.monotonic()
    shadow = load_cpu(ROOT / source['query_cache']).to(torch.device('cpu'))
    encoder = load_cpu(ROOT / source['shadow_cache']).cpu().eval()
    if (type(encoder).__name__ != 'TailProxyShadow' or encoder.branch_training['behavioral']['max_epochs'] != 300
            or encoder.num_users != shadow.num_users or encoder.num_items != shadow.num_items):
        raise ValueError('Frozen encoder and query source mismatch')
    source = dict(source, query_repr=repr(shadow), shadow_repr=repr(encoder),
                  behavior_fingerprint=model_fingerprint(encoder.behavioral_branch),
                  behavior_training={k: v for k, v in encoder.branch_training['behavioral'].items() if k != 'history'})
    full = incidence(shadow.behavioral_edge_index, shadow.num_users, shadow.num_items)
    train = incidence(shadow.behavioral_train_edge_index, shadow.num_users, shadow.num_items)
    ids = np.flatnonzero(np.asarray(full.sum(0)).ravel())
    full, train = full[:, ids].tocsr(), train[:, ids].tocsr()
    train_mask = np.asarray(train.sum(0)).ravel() > 0
    views = {}
    with torch.no_grad():
        idf = encoder.recommendation_idf(shadow.behavioral_edge_index)
        for name, edges in [('train', shadow.behavioral_train_edge_index), ('full', shadow.behavioral_edge_index)]:
            embeds = encoder.behavioral_branch(encoder.behavior_adj(edges, idf))[shadow.num_users:]
            views[name] = embeds[torch.from_numpy(ids)].clone()
    rvq = load_item_rvq_model(embeddings=views['train'][train_mask], item_ids=torch.from_numpy(ids[train_mask]),
                              source=dict(query=repr(shadow), behavior=source['behavior_fingerprint'], view='train'),
                              seed=source['seed'] + 17027, num_quantizers=args.levels,
                              codebook_size=args.capacity, max_iter=args.max_iter)
    code_paths, reconstruction = {}, {}
    for view in views:
        codes, error = rvq.encode(views[view])
        code_paths[view], reconstruction[view] = codes.numpy(), error.numpy()
    del encoder
    # Only after fitting is finished do we load the evaluator's relation labels.
    data = load_cpu(ROOT / source['attack_data_cache']).to(torch.device('cpu'))
    if (data.shadow_data_repr != repr(shadow) or data.shadow_model_repr != source['shadow_repr']
            or not torch.equal(data.train_social_edge_index, shadow.attack_train_social_edge_index)
            or not torch.equal(data.val_social_edge_index, shadow.attack_val_social_edge_index)):
        raise ValueError('Evaluation data differs from the frozen model/query source')
    protocol, weights, evaluation_id = evaluation_protocol(data, source['seed'])
    if source['prior_evaluation_id'] is not None and source['prior_evaluation_id'] != evaluation_id:
        raise ValueError('Reconstructed evaluation pairs differ from the prior stage1 CPU protocol')
    legacy_pairs = ROOT / source['run'] / 'evaluation_pairs.pt'
    if legacy_pairs.is_file():
        legacy = torch.load(legacy_pairs, map_location='cpu', weights_only=False)
        # Design's earlier runner also saved the same fixed CPU protocol.
        for split in ('val', 'test'):
            if not np.array_equal(legacy[split]['edge_index'].numpy(), protocol[split]['pairs']):
                raise ValueError('Reconstructed pairs differ from saved DESIGN protocol')
    all_pairs = np.concatenate([protocol['val']['pairs'], protocol['test']['pairs']], axis=1)
    nval = protocol['val']['pairs'].shape[1]
    result = dict(revision=REVISION, target=source['target'], dataset=source['dataset'], seed=source['seed'],
                  source=source, evaluation_id=evaluation_id, rvq=rvq.fit_summary,
                  quantizer_fingerprint=model_fingerprint(rvq), queried_items=len(ids),
                  training_items=int(train_mask.sum()), query_only_items=int((~train_mask).sum()), views={})
    for view, rec in [('train', train), ('full', full)]:
        logging.info('%s/%s/%s evaluating view=%s', source['target'], source['dataset'], source['seed'], view)
        item_scores = score_pairs(rec, all_pairs)
        view_result = dict(item={}, levels=[])
        for kind in ('count', 'overlap', 'jaccard'):
            view_result['item'][kind] = relation_report(item_scores[kind][nval:], item_scores[kind][:nval], protocol, weights)
        for level in range(args.levels):
            matrix, diagnostic = layer_diagnostics(code_paths[view][:, :level + 1], rec, reconstruction[view][:, level])
            prefix_scores = score_pairs(matrix, all_pairs)
            diagnostic.update(level=level + 1, equivalence={}, matching={})
            for kind in ('overlap', 'jaccard'):
                diagnostic['equivalence'][kind] = comparison(item_scores[kind][nval:], prefix_scores[kind][nval:])
                diagnostic['matching'][kind] = relation_report(prefix_scores[kind][nval:], prefix_scores[kind][:nval], protocol, weights)
            # False shared proxies and loss of exact-item matches are quantified
            # independently of labels; collisions need not always worsen AUC.
            diagnostic['false_overlap_pair_ratio'] = float(((item_scores['count'][nval:] == 0)
                                                            & (prefix_scores['count'][nval:] > 0)).mean())
            view_result['levels'].append(diagnostic)
        last = view_result['levels'][-1]
        strict = (last['collision_pairs'] == 0 and last['frequency']['equal_ratio'] == 1
                  and all(last['equivalence'][k]['close_ratio'] == 1 for k in ('overlap', 'jaccard')))
        approximate = (last['unique_item_ratio'] >= .99 and last['frequency']['equal_ratio'] >= .99
                       and all(last['equivalence'][k]['normalized_mae'] <= .01
                               and last['equivalence'][k]['pearson'] is not None and last['equivalence'][k]['pearson'] >= .99
                               and last['equivalence'][k]['spearman'] is not None and last['equivalence'][k]['spearman'] >= .99
                               for k in ('overlap', 'jaccard')))
        auc_close = all(abs(last['matching'][kind]['groups'][g]['AUROC'] - view_result['item'][kind]['groups'][g]['AUROC']) <= .005
                        for kind in ('overlap', 'jaccard') for g in GROUPS)
        view_result['assessment'] = dict(strict_equivalence=strict, approximate_representation=approximate,
                                         auc_difference_within_005=auc_close)
        result['views'][view] = view_result
    result['view_shift'] = dict(per_level_code_change=(code_paths['train'] != code_paths['full']).mean(0).tolist(),
                               per_prefix_path_change=[float(np.any(code_paths['train'][:, :l] != code_paths['full'][:, :l], axis=1).mean())
                                                       for l in range(1, args.levels + 1)],
                               embedding_relative_l2=float((views['full'] - views['train']).norm() / views['train'].norm()))
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def summarize(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    rows = [json.loads((output / f"{case['target']}/{case['dataset']}/seed-{case['seed']}/metrics.json").read_text())
            for case in manifest['sources'] if (output / f"{case['target']}/{case['dataset']}/seed-{case['seed']}/metrics.json").is_file()]
    records, summaries = [], []
    for row in rows:
        for view, values in row['views'].items():
            methods = {f'item_{k}': v for k, v in values['item'].items()}
            methods.update({f"prefix{layer['level']}_{k}": v for layer in values['levels'] for k, v in layer['matching'].items()})
            for method, value in methods.items():
                for group, metrics_row in value['groups'].items():
                    records.append(dict(target=row['target'], dataset=row['dataset'], seed=row['seed'],
                                        view=view, method=method, group=group, **metrics_row))
    grouped = {}
    for r in records:
        key = tuple(r[k] for k in ('target', 'dataset', 'view', 'method', 'group'))
        grouped.setdefault(key, []).append(r)
    for key, group in grouped.items():
        row = dict(zip(('target', 'dataset', 'view', 'method', 'group'), key))
        row.update(seeds=[v['seed'] for v in group], n=len(group))
        for metric in METRICS:
            values = [v[metric] for v in group if v[metric] is not None]
            row[metric] = dict(mean=statistics.mean(values) if values else None,
                               sample_sd=statistics.stdev(values) if len(values) > 1 else None)
        summaries.append(row)
    write_json(output / 'summary.json', dict(completed=len(rows), planned=len(manifest['sources']), groups=summaries,
                                             cases=[dict(target=r['target'], dataset=r['dataset'], seed=r['seed'],
                                                         full=r['views']['full']['assessment'], train=r['views']['train']['assessment'],
                                                         elapsed_seconds=r['elapsed_seconds']) for r in rows]))
    if records:
        with (output / 'metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f'Completed {len(rows)}/{len(manifest["sources"])} cases: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', choices=['design', 'gdmsr'], default=['design', 'gdmsr'])
    parser.add_argument('--datasets', nargs='+', choices=['ciao', 'flickr'], default=['ciao', 'flickr'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 47)))
    parser.add_argument('--levels', type=int, default=8)
    parser.add_argument('--capacity', type=int, default=256)
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.summarize_only:
        summarize(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=args.resume)
    identity = dict(revision=REVISION, targets=args.targets, datasets=args.datasets, seeds=args.seeds,
                    levels=args.levels, capacity=args.capacity, max_iter=args.max_iter, threads=args.threads,
                    device='cpu', torch=torch.__version__, sklearn=sklearn.__version__, numpy=np.__version__)
    if args.resume:
        manifest = json.loads((args.output / 'manifest.json').read_text())
        if manifest['identity'] != identity:
            raise ValueError('Resume settings differ from the preregistered manifest')
    else:
        sources = [source_record(t, d, seed) for seed in args.seeds for d in args.datasets for t in args.targets]
        manifest = dict(identity=identity, sources=sources, protocol=dict(
            source_budget=300, source_revision='tail-proxy-shadow-e300-v2', fit_view='training-covered item embeddings',
            fit='sequential residual k-means++; n_init=1; max_iter=50; tol=1e-4; unnormalized float64; seed+17027+level',
            frequency='distinct users; log((N+1)/(d+1)); independently within train/full views',
            scores='float64; round to 10 decimal places only for rank/ROC/threshold ties',
            selection='no relation labels; no codebook search; all 8 layers reported',
            strict='zero full-prefix collisions, exact frequency equality, all pair scores close at rtol=atol=1e-10',
            approximate='singleton/frequency equal ratios >=.99, normalized MAE<=.01, Pearson/Spearman>=.99 for both scores',
            descriptive_auc='absolute difference <=.005 in all five groups; reporting only, never model selection'))
        write_json(args.output / 'manifest.json', manifest)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(args.output / 'console.log'), logging.StreamHandler()], force=True)
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        for source in manifest['sources']:
            case_dir = args.output / f"{source['target']}/{source['dataset']}/seed-{source['seed']}"
            if (case_dir / 'metrics.json').is_file():
                continue
            logging.info('Starting %s/%s/%s', source['target'], source['dataset'], source['seed'])
            result = run_case(source, args)
            case_dir.mkdir(parents=True, exist_ok=True)
            write_json(case_dir / 'metrics.json', result)
            logging.info('Completed %s: %s', case_dir, result['views']['full']['assessment'])
            summarize(args.output)


if __name__ == '__main__':
    main()

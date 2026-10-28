#!/usr/bin/env python3
"""Analyze collision causes and fixed item paths using already fitted Stage-A RVQ.

No fitting or checkpoint writes. This supplements each completed case with a
full-query view using training-view item codes, for the Stage-B topology design.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import re

from item_prefix_stage_a import (
    ROOT, comparison, evaluation_protocol, incidence, layer_diagnostics, load_cpu,
    model_fingerprint, np, prefix_incidence, relation_report, score_pairs,
    threadpool_limits, torch, write_json,
)


def collision_details(codes, embeddings, rec, errors, pairs):
    _, mapping, sizes = np.unique(codes, axis=0, return_inverse=True, return_counts=True)
    collide = sizes[mapping] > 1
    csc = rec.tocsc()
    frequency = np.asarray(rec.sum(0)).ravel()
    same_coverage_pairs, one_user_pairs, identical_embedding_pairs = 0, 0, 0
    all_pairs = int(np.sum(sizes * (sizes - 1) // 2))
    buckets = {}
    for item in np.flatnonzero(collide):
        buckets.setdefault(int(mapping[item]), []).append(item)
    for members in buckets.values():
        owners = {}
        for item in members:
            key = tuple(csc.indices[csc.indptr[item]:csc.indptr[item + 1]])
            owners[key] = owners.get(key, 0) + 1
        for owners_key, count in owners.items():
            number = count * (count - 1) // 2
            same_coverage_pairs += number
            if len(owners_key) == 1:
                one_user_pairs += number
        _, counts = np.unique(embeddings[members], axis=0, return_counts=True)
        identical_embedding_pairs += int(np.sum(counts * (counts - 1) // 2))
    details = dict(collision_pairs=all_pairs, collided_items=int(collide.sum()),
                   same_user_set_pairs=same_coverage_pairs, single_user_same_set_pairs=one_user_pairs,
                   same_user_set_pair_ratio=same_coverage_pairs / all_pairs if all_pairs else None,
                   exactly_identical_embedding_pairs=identical_embedding_pairs,
                   collided_item_frequency_mean=float(frequency[collide].mean()) if collide.any() else None,
                   singleton_item_frequency_mean=float(frequency[~collide].mean()) if (~collide).any() else None,
                   collided_item_mse=float(errors[collide].mean()) if collide.any() else None,
                   singleton_item_mse=float(errors[~collide].mean()) if (~collide).any() else None,
                   parts={})
    for name, selected in [('singleton', ~collide), ('collided', collide)]:
        if not selected.any():
            details['parts'][name] = None
            continue
        subset = rec[:, selected].tocsr()
        _, subset_mapping = np.unique(mapping[selected], return_inverse=True)
        matrix = prefix_incidence(subset, subset_mapping, int(subset_mapping.max()) + 1)
        raw, proxy = score_pairs(subset, pairs), score_pairs(matrix, pairs)
        # Jaccard is restricted to this part, not an additive decomposition.
        details['parts'][name] = {key: comparison(raw[key], proxy[key]) for key in ('overlap', 'jaccard')}
        delta = proxy['overlap'] - raw['overlap']
        details['parts'][name]['pair_effect'] = dict(
            increased_ratio=float((delta > 1e-10).mean()), decreased_ratio=float((delta < -1e-10).mean()),
            false_overlap_ratio=float(((raw['count'] == 0) & (proxy['count'] > 0)).mean()))
    return details


def run_case(path, run_dir):
    primary = json.loads(path.read_text())
    source = primary['source']
    shadow = load_cpu(ROOT / source['query_cache']).to(torch.device('cpu'))
    encoder = load_cpu(ROOT / source['shadow_cache']).cpu().eval()
    text = (run_dir / 'console.log').read_text()
    marker = f"Starting {source['target']}/{source['dataset']}/{source['seed']}"
    section = text.split(marker)[-1].split(' INFO Starting ')[0]
    cache = re.findall(r'"(\.cache/shadow/model/ItemResidualQuantizer/[^"\s]+\.pkl)"', section)
    if len(cache) != 1:
        raise ValueError(f'Expected one fitted RVQ cache for {marker}')
    rvq = load_cpu(ROOT / cache[0]).eval()
    if model_fingerprint(rvq) != primary['quantizer_fingerprint']:
        raise ValueError('Follow-up codebook differs from completed Stage A')
    rec = incidence(shadow.behavioral_edge_index, shadow.num_users, shadow.num_items)
    ids = np.flatnonzero(np.asarray(rec.sum(0)).ravel())
    rec = rec[:, ids].tocsr()
    codes, embeddings, errors = {}, {}, {}
    with torch.no_grad():
        idf = encoder.recommendation_idf(shadow.behavioral_edge_index)
        for view, edge in [('train', shadow.behavioral_train_edge_index), ('full', shadow.behavioral_edge_index)]:
            embeds = encoder.behavioral_branch(encoder.behavior_adj(edge, idf))[shadow.num_users:][ids]
            code, error = rvq.encode(embeds)
            codes[view], embeddings[view], errors[view] = code.numpy(), embeds.numpy(), error.numpy()[:, -1]
    data = load_cpu(ROOT / source['attack_data_cache']).to(torch.device('cpu'))
    protocol, weights, identity = evaluation_protocol(data, source['seed'])
    if identity != primary['evaluation_id']:
        raise ValueError('Follow-up evaluation protocol changed')
    pairs = np.concatenate([protocol['val']['pairs'], protocol['test']['pairs']], axis=1)
    nval = protocol['val']['pairs'].shape[1]
    result = dict(target=primary['target'], dataset=primary['dataset'], seed=primary['seed'],
                  revision='item-prefix-fixed-path-followup-v1', quantizer_cache=cache[0],
                  evaluation_id=identity, decision='freeze train-view item codes; use full-query sets/frequency')
    matrix, diagnostic = layer_diagnostics(codes['train'], rec, errors['train'])
    raw, proxy = score_pairs(rec, pairs), score_pairs(matrix, pairs)
    diagnostic.update(equivalence={}, matching={})
    for key in ('overlap', 'jaccard'):
        diagnostic['equivalence'][key] = comparison(raw[key][nval:], proxy[key][nval:])
        diagnostic['matching'][key] = relation_report(proxy[key][nval:], proxy[key][:nval], protocol, weights)
    result['full_query_fixed_paths'] = diagnostic
    result['collision_analysis'] = {
        view: collision_details(codes[view], embeddings[view], rec, errors[view], protocol['test']['pairs'])
        for view in ('train', 'full')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(args.run_dir / 'followup.log'), logging.StreamHandler()])
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    write_json(args.run_dir / 'followup-plan.json', dict(
        purpose='Measure fixed item identities for Stage B and explain original full-prefix collisions',
        primary_manifest='manifest.json', fitting=False, source='same fitted RVQ and frozen encoder',
        decision_basis='observed cross-view path drift; no capacity/depth fitting or relation-label search',
        threads=args.threads))
    with threadpool_limits(limits=args.threads):
        for case in manifest['sources']:
            path = args.run_dir / f"{case['target']}/{case['dataset']}/seed-{case['seed']}/metrics.json"
            destination = path.with_name('followup.json')
            if not path.exists() or destination.exists():
                continue
            result = run_case(path, args.run_dir)
            write_json(destination, result)
            logging.info('Fixed-path follow-up completed: %s', destination.parent)


if __name__ == '__main__':
    main()

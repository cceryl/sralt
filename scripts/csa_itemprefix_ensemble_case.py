#!/usr/bin/env python3
"""One complete-expert ensemble case, public fitting and shared final evaluation."""

import argparse
import fcntl
import json
import logging
from pathlib import Path
import tempfile
import time

from tail_proxy_case import (torch, instantiate, OmegaConf, load_target_data, load_target_model,
                             load_shadow_data, fixed_protocol, write_json, now, GROUPS, GROUP_FIELDS)
from item_prefix_csa_case import configuration as expert_configuration
from csa_itemprefix_ensemble_recipe import (EXPERIMENT_REVISION, SUITES, OOF_CONDITIONS,
                                           TRAINING, source_fingerprint)
from sralt.attack.ensemble import ExpertBundle, EnsembleEvidence
from sralt.attack.ensemble.models import make_fusion
from sralt.attack.ensemble.evidence import requirements
from sralt.attack.ensemble.training import (load_fusion_pair_pool, training_identity, load_oof_attack_evidence,
                                           load_fusion_model, predict_fusion)
from sralt.attack.ensemble.diagnostics import complementarity, calibration_by_degree
from sralt.utils.attack_groups import grouped_attack_report, optimal_f1_threshold


PARITY_RTOL = 1e-5
PARITY_ATOL = 1e-7


def run(args, *, configuration_fn=expert_configuration, target_loader=load_target_model,
        expert_loaders=None, fusion_loader=load_fusion_model):
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    torch.set_num_threads(args.threads)
    configs = [configuration_fn(args, name) for name in ('csa', 'item_prefix_joint')]
    cfg = configs[0]
    device = torch.device('cpu') if args.cpu else instantiate(cfg.device)
    common = dict(seed=args.seed, deterministic=cfg.deterministic)
    lock_dir = Path(tempfile.gettempdir()) / 'sralt-target-data-locks'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f'{args.dataset}-{args.seed}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = load_target_data(**common, config=cfg.target.data).to(device)
    target_model = target_loader(**common, device=device, config=cfg.target.model, data=target, writer=None)
    shadow = load_shadow_data(**common, target_data=target, target_model=target_model, config=cfg.shadow.data)
    del target_model
    target.to(torch.device('cpu'))
    protocol, evaluation_id = fixed_protocol(target, shadow, cfg)
    # Hydra's writer runtime interpolation is unavailable in compose-only workers.
    config = dict(experts=[{key: OmegaConf.to_container(c[key], resolve=True)
                            for key in ('target', 'shadow', 'attack', 'device')} for c in configs],
                  deterministic=bool(cfg.deterministic), training=TRAINING)
    path = args.output / 'metrics.json'
    if path.exists():
        result = json.loads(path.read_text())
        if (result['experiment_revision'] != EXPERIMENT_REVISION or result['source_fingerprint'] != args.source_id
                or result['config'] != config or result['evaluation_id'] != evaluation_id
                or (result['target'], result['dataset'], result['seed']) != (args.target, args.dataset, args.seed)):
            raise ValueError('Saved ensemble case has different implementation/configuration/pairs.')
    else:
        result = dict(schema=3, experiment_revision=EXPERIMENT_REVISION, source_fingerprint=args.source_id,
                      target=args.target, dataset=args.dataset, seed=args.seed, evaluation_id=evaluation_id,
                      config=config, results={})
    if all(name in result['results'] for name in args.conditions):
        return
    need_attack = any(requirements(name)['scores'] or requirements(name)['hidden'] for name in args.conditions)
    bundle = ExpertBundle.load(configs=configs, target=target, shadow=shadow, protocol=protocol,
                               device=device, needs_attack=need_attack, **common, **(expert_loaders or {}))
    del target, shadow
    evidence = EnsembleEvidence(bundle)
    before = bundle.fingerprints()
    if result.get('experts', bundle.sources) != bundle.sources:
        raise ValueError('Formal expert cache fingerprints changed across resume.')
    result['experts'] = bundle.sources
    result['public_degree'] = dict(tau=evidence.tau, definition='unique undirected observed T, candidate edge excluded')
    validation = protocol['val']
    # S0 needs no training pool and never pays the six auxiliary Attack fits.
    trained = any(any(p.requires_grad for p in make_fusion(name, evidence.user_dims, evidence.hidden_dims,
                                                          evidence.tau).parameters()) for name in args.conditions)
    pool = (load_fusion_pair_pool(identity=dict(revision=EXPERIMENT_REVISION,
                source_fingerprint=args.source_id, public=bundle.sources[0]['public']),
                data=bundle.data[0], **common)
            if trained else {})
    identity = training_identity(bundle, pool, validation, TRAINING)
    identity['source_fingerprint'] = args.source_id
    oof = None
    if any(name in OOF_CONDITIONS and name not in result['results'] for name in args.conditions):
        oof = load_oof_attack_evidence(identity=identity, bundle=bundle, pool=pool, validation=validation,
                                       device=device, **common)
        result['oof'] = {k: v for k, v in oof.items() if k != 'logits'}
    if need_attack and 'public_complementarity' not in result:
        with torch.no_grad():
            batch_size = int(cfg.attack.test.evaluation.test_batch_size)
            logits = torch.cat([evidence.batch(batch.to(device), scores=True)['logits'].cpu()
                                for batch in validation['edge_index'].split(batch_size, dim=1)])
            degrees = evidence.degrees(validation['edge_index'].to(device)).cpu()
            result['public_complementarity'] = complementarity(logits.sigmoid(), validation['labels'], degrees)
            # Endpoint and native-interface parity checks on a public batch.
            edges = validation['edge_index'][:, :batch_size].to(device)
            direct = []
            for expert, data in zip(bundle.models, bundle.data, strict=True):
                inputs = data if hasattr(expert, 'endpoint_features') else data.user_features
                direct.append(expert.predict(edges, inputs))
            native = torch.stack(direct, -1).cpu()
            adapted = logits[:native.size(0)].sigmoid()
            # Native predict can apply sigmoid on GPU while stored evidence is
            # on CPU. Compare numerically instead of requiring identical bits.
            if not torch.allclose(native, adapted, rtol=PARITY_RTOL, atol=PARITY_ATOL):
                raise ValueError('Native expert predict differs from score adapter.')
            reverse = evidence.batch(edges.flip(0), scores=True)['logits'].sigmoid().cpu()
            if not torch.allclose(native, reverse, rtol=PARITY_RTOL, atol=PARITY_ATOL):
                raise ValueError('Expert pair orientation changed predictions.')
            result['acceptance'] = dict(native_predict_parity=True, endpoint_symmetry=True,
                aligned_protocol=True, oof_pair_disjointness_checked=oof is not None,
                parity_rtol=PARITY_RTOL, parity_atol=PARITY_ATOL,
                native_predict_max_abs_error=float((native - adapted).abs().max()),
                endpoint_symmetry_max_abs_error=float((native - reverse).abs().max()))
    for name in args.conditions:
        if name in result['results']:
            continue
        logging.info('Starting ensemble condition %s [%d/%d]', name, args.conditions.index(name) + 1, len(args.conditions))
        started = time.monotonic()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model = fusion_loader(identity=identity, condition=name, settings=TRAINING, evidence=evidence,
                                  pool=pool, validation=validation, oof=oof if name in OOF_CONDITIONS else None,
                                  device=device, **common)
        val_p = predict_fusion(model, evidence, validation['edge_index'], name)
        threshold = optimal_f1_threshold(val_p, validation['labels'])
        probability = predict_fusion(model, evidence, protocol['test']['edge_index'], name)
        groups = grouped_attack_report(protocol['test']['edge_index'], probability, protocol['test']['labels'],
                                       bundle.data[0], protocol['head'], protocol['tail'], instantiate(cfg.attack.test.metrics),
                                       f1_threshold=threshold)
        if name in ('csa', 'item_prefix_joint'):
            column = 0 if name == 'csa' else 1
            expected = torch.cat([evidence.batch(batch.to(device), scores=True)['logits'][:, column].sigmoid().cpu()
                                  for batch in validation['edge_index'].split(2048, dim=1)])
            if not torch.allclose(val_p, expected, rtol=PARITY_RTOL, atol=PARITY_ATOL):
                raise ValueError('Alpha endpoint does not reproduce the native expert.')
        degrees = evidence.degrees(validation['edge_index'].to(device)).cpu()
        after = bundle.fingerprints()
        if before != after or any(m.training for m in bundle.models) or any(
                p.requires_grad or p.grad is not None for m in bundle.models for p in m.parameters()):
            raise ValueError('Frozen formal expert state changed during fusion.')
        result['results'][name] = dict(groups={group: {key: groups[group][key] for key in GROUP_FIELDS}
                                              for group in GROUPS},
            training=getattr(model, 'training_summary', None),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            elapsed_seconds=time.monotonic() - started,
            peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            public_calibration=calibration_by_degree(val_p, validation['labels'], degrees),
            frozen_fingerprints_before=before, frozen_fingerprints_after=after)
        result['updated_at'] = now()
        write_json(path, result)
        del model
        logging.info('Completed ensemble condition %s: %s', name, result['results'][name]['groups']['All'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['design', 'gdmsr'], required=True)
    parser.add_argument('--dataset', choices=['ciao', 'flickr'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--conditions', nargs='+', choices=SUITES['all'], default=SUITES['S0'])
    parser.add_argument('--expected-revision')
    parser.add_argument('--expected-source')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    args.source_id = source_fingerprint()
    if args.expected_revision is not None and args.expected_revision != EXPERIMENT_REVISION:
        parser.error('Scheduler/worker experiment revision differs; start a new batch.')
    if args.expected_source is not None and args.expected_source != args.source_id:
        parser.error('Scheduler/worker source or expert config changed; start a new batch.')
    run(args)


if __name__ == '__main__':
    main()

"""Public-only pools, cached Attack OOF evidence and cached fusion fitting."""

import copy
import logging
import time

import torch
from hydra.utils import instantiate
from torch.nn import functional as F
from omegaconf import OmegaConf

from sralt.attack import load_attack_model
from sralt.attack.data import AttackData, BalancedAttackBatchSampler
from sralt.attack.data.dataset import sample_negative_pairs
from sralt.utils import disk_cache, set_seed, EarlyStopping
from sralt.utils.model import tensor_fingerprint
from .evidence import canonical_pairs, pair_ids, public_identity, requirements
from .models import make_fusion, PlattMean

REVISION = 'csa-itemprefix-ensemble-v2-resampled-T'


class CrossFitAttackData(AttackData):
    """Native payload and graph, partitioned supervision and fresh negatives."""

    def cache_key(self):
        return (REVISION, public_identity(self), self.attack_training_split,
                self.fixed_eval_pairs['val'], self.fold_identity)

    def __repr__(self):
        return f'CrossFitAttackData({self.fold_identity})'

    def sample_attack_training_negatives(self, num_samples):
        """Sample against T only, then apply this copy's OOF supervision split.

        V is deliberately not consulted: its pairs can be pseudo-negatives.
        Held-out OOF pairs cannot contribute gradients to their own copy.
        Refill after partitioning to retain native 1:1 balanced batches.
        """
        held = self.attack_training_split['held_edge_index'].to(self.device)
        held_ids = pair_ids(held, self.num_users)
        selected, selected_ids, count = [], held_ids, 0
        for _ in range(100):
            candidates = sample_negative_pairs(self.train_social_edge_index, self.num_users, num_samples)
            ids = pair_ids(candidates, self.num_users)
            keep = ~torch.isin(ids, selected_ids)
            candidates, ids = candidates[:, keep], ids[keep]
            take = min(num_samples - count, ids.numel())
            selected.append(candidates[:, :take])
            selected_ids = torch.cat([selected_ids, ids[:take]])
            count += take
            if count == num_samples:
                return torch.cat(selected, dim=1)
        raise ValueError('Cannot fill a balanced OOF training split from T-only pseudo-negatives.')


def fixed_public_pool(data, seed, deterministic):
    set_seed(seed, deterministic)
    n = data.num_users
    positives = canonical_pairs(data.train_social_edge_index, n)
    # This is the fixed OOF scoring pool, not an auxiliary epoch's negatives.
    # Match the native public pseudo-negative protocol: exclude T only, never V
    # or hidden positives. No validation candidate/label filters are applied.
    negatives = sample_negative_pairs(data.train_social_edge_index, n, positives.size(1))
    if negatives.size(1) != positives.size(1) or positives.size(1) < 3:
        raise ValueError('Need >=3 positive pairs and an equally large distinct negative pool.')
    edges = torch.cat([positives, negatives], 1).cpu()
    labels = torch.cat([torch.ones(positives.size(1)), torch.zeros(negatives.size(1))]).long()
    folds = torch.empty(labels.numel(), dtype=torch.long)
    # One canonical occurrence per pair; stratification gives equal class counts
    # in each fold. Auxiliary training negatives are resampled every epoch.
    for label in (0, 1):
        indices = torch.where(labels == label)[0]
        folds[indices[torch.randperm(indices.numel())]] = torch.arange(indices.numel()) % 3
    ids = pair_ids(edges, n)
    if ids.unique().numel() != ids.numel():
        raise ValueError('OOF scoring pool contains duplicate or opposite pairs.')
    return dict(edge_index=edges, labels=labels, folds=folds)


def pool_identity(pool):
    return {key: tensor_fingerprint(value) for key, value in pool.items()}


@disk_cache(cache_dir='.cache/attack/ensemble/pairs', ignore=['data'])
def load_fusion_pair_pool(*, identity, seed, deterministic, data):
    """Persist the public pair/fold pool without any hidden evaluation identity."""
    return fixed_public_pool(data, seed, deterministic)


def training_identity(bundle, pool, validation, settings):
    return dict(revision=REVISION, sources=bundle.sources,
                attack_configs=[OmegaConf.to_container(cfg.attack.model, resolve=True) for cfg in bundle.configs],
                pool=pool_identity(pool),
                validation={k: tensor_fingerprint(v) for k, v in validation.items()}, settings=settings)


@disk_cache(cache_dir='.cache/attack/ensemble/oof', ignore=['bundle', 'pool', 'validation', 'device'])
def load_oof_attack_evidence(*, identity, seed, deterministic, bundle, pool, validation, device):
    """Six independently cached native Attack fits; no Shadow retraining."""
    started = time.monotonic()
    scores = torch.empty(pool['labels'].numel(), 2)
    summaries = []
    for fold in range(3):
        keep, held = pool['folds'] != fold, pool['folds'] == fold
        held_ids = pair_ids(pool['edge_index'][:, held], bundle.data[0].num_users)
        keep_ids = pair_ids(pool['edge_index'][:, keep], bundle.data[0].num_users)
        if torch.isin(held_ids, keep_ids).any():
            raise ValueError('OOF pair leakage.')
        for column, (data, cfg) in enumerate(zip(bundle.data, bundle.configs, strict=True)):
            # Native graph/features remain untouched. Only supervised positives
            # and held-out OOF pair membership differ from the formal expert.
            view = object.__new__(CrossFitAttackData)
            view.__dict__ = copy.copy(data.__dict__)
            view.attack_training_split = dict(
                positive_edge_index=pool['edge_index'][:, keep & (pool['labels'] == 1)],
                held_edge_index=pool['edge_index'][:, held])
            view.fixed_eval_pairs = {'val': validation}
            view.fold_identity = dict(fold=fold, expert=column, pool=identity['pool'], revision=REVISION,
                                      source_fingerprint=identity['source_fingerprint'])
            fit_started = time.monotonic()
            model = load_attack_model(seed=seed, deterministic=deterministic, device=device,
                                      config=cfg.attack.model, data=view, tail_ratio=cfg.attack.tail_ratio, writer=None)
            model.eval().requires_grad_(False)
            batch_size = int(cfg.attack.model.train.val_batch_size)
            indices = torch.where(held)[0]
            with torch.no_grad():
                for batch in indices.split(batch_size):
                    edges = pool['edge_index'][:, batch].to(device)
                    inputs = view if hasattr(model, 'endpoint_features') else view.user_features
                    scores[batch, column] = model(edges, inputs).cpu()
            summaries.append(dict(fold=fold, expert=column, fit_pairs=int(keep.sum()),
                                  held_pairs=int(held.sum()), load_or_fit_seconds=time.monotonic() - fit_started,
                                  negative_sampling='resampled each epoch; graph exclusion=T only; OOF supervision partition',
                                  training=getattr(model, 'training_summary', None)))
            del model
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Nonfinite OOF scores.')
    return dict(logits=scores, identity=identity, auxiliary_fits=summaries,
                negative_sampling='resampled each epoch; T only; V candidates may overlap training pseudo-negatives',
                elapsed_seconds=time.monotonic() - started, supervision='3-fold Attack-only OOF; full-T frozen Shadow')


class PoolSamplerView:
    def __init__(self, labels):
        self.pos_edge_index = torch.empty(2, int((labels == 1).sum()))
        self.num_negatives = int((labels == 0).sum())


def balanced_indices(labels, batch_size):
    # BalancedAttackBatchSampler expects positives followed by negatives.
    order = torch.cat([torch.where(labels == 1)[0], torch.where(labels == 0)[0]])
    for batch in BalancedAttackBatchSampler(PoolSamplerView(labels), batch_size):
        yield order[batch]


def fusion_batch(evidence, edges, condition, oof_logits=None):
    flags = requirements(condition)
    if oof_logits is not None:
        flags['scores'] = False
    result = evidence.batch(edges, **flags)
    if oof_logits is not None:
        result['logits'] = oof_logits.to(edges.device).detach()
    return result


@torch.no_grad()
def predict_fusion(model, evidence, edges, condition, batch_size=2048):
    model.eval()
    device = evidence.degree.device
    return torch.cat([model(fusion_batch(evidence, batch.to(device), condition)).cpu()
                      for batch in edges.split(batch_size, dim=1)])


@disk_cache(cache_dir='.cache/attack/ensemble/fusion',
            ignore=['evidence', 'pool', 'validation', 'oof', 'device'],
            hook=lambda model, args, _: model.to(args['device']))
def load_fusion_model(*, identity, condition, seed, deterministic, settings, evidence, pool,
                      validation, oof, device):
    """Cache only the small fusion module; formal experts stay separate."""
    set_seed(seed, deterministic)
    model = make_fusion(condition, evidence.user_dims, evidence.hidden_dims, evidence.tau).to(device)
    started = time.monotonic()
    if not any(p.requires_grad for p in model.parameters()):
        model.training_summary = dict(supervision='none', actual_epochs=0)
        return model.cpu().eval()
    if condition in ('gate_degree', 'gate_user', 'platt_mean') and oof is None:
        raise ValueError('Trainable score fusion requires Attack-level OOF evidence.')
    labels = pool['labels']
    if isinstance(model, PlattMean):
        if oof is None:
            raise ValueError('Platt calibration requires OOF scores.')
        # Tiny two-expert regression uses CPU float64 for stable LBFGS fitting.
        model.cpu().double()
        logits, y = oof['logits'].double(), labels.double()
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=settings['platt_max_iter'],
                                      line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            calibrated_logits = F.softplus(model.scale) * logits + model.bias
            # Each expert is calibrated independently; optimize the summed BCE,
            # not the BCE of their mean, which could permit compensating errors.
            loss = F.binary_cross_entropy_with_logits(calibrated_logits, y[:, None].expand_as(logits))
            loss = loss + settings['platt_l2'] * (model.scale.square().sum() + model.bias.square().sum())
            loss.backward()
            return loss
        optimizer.step(closure)
        model.float()
        model.training_summary = dict(supervision='OOF Platt per expert', max_iter=settings['platt_max_iter'],
                                      elapsed_seconds=time.monotonic() - started,
                                      slopes=F.softplus(model.scale).detach().tolist(), bias=model.bias.detach().tolist())
        return model.cpu().eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    stop = EarlyStopping(model, patience=settings['patience'], greater_is_better=True)
    metrics = instantiate(evidence.bundle.configs[0].attack.model.train.metrics)
    history = []
    for epoch in range(1, settings['num_epochs'] + 1):
        model.train()
        evidence.bundle.train()  # Explicitly remains eval, including BatchNorm.
        loss_sum = 0.
        if oof is None:
            # Representation heads / degree-only follow the same T-only
            # per-epoch pseudo-negative convention. Gates use fixed OOF rows.
            native = evidence.bundle.data[0]
            positives = pool['edge_index'][:, labels == 1]
            negatives = sample_negative_pairs(native.train_social_edge_index, native.num_users,
                                               positives.size(1)).cpu()
            if negatives.size(1) != positives.size(1):
                raise ValueError('Cannot fill balanced Fusion pseudo-negatives.')
            epoch_edges = torch.cat([positives, negatives], dim=1)
        else:
            epoch_edges = pool['edge_index']
        for indices in balanced_indices(labels, settings['batch_size']):
            edges = epoch_edges[:, indices].to(device)
            batch = fusion_batch(evidence, edges, condition, oof['logits'][indices] if oof is not None else None)
            optimizer.zero_grad()
            probability = model(batch)
            loss = F.binary_cross_entropy(probability.clamp(1e-7, 1 - 1e-7), labels[indices].float().to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * indices.numel()
        probability = predict_fusion(model, evidence, validation['edge_index'], condition)
        values = []
        for metric in metrics:
            metric.cpu().reset()
            values.append(float(metric(probability, validation['labels'])))
        score = sum(values)
        history.append(dict(epoch=epoch, loss=loss_sum / labels.numel(), validation=values))
        logging.info('Fusion %s epoch=%d loss=%.6f val=%s', condition, epoch, history[-1]['loss'], values)
        if stop.step(score):
            break
    stop.load_best_state()
    model.training_summary = dict(actual_epochs=len(history), best_epoch=stop.best_epoch, best_metric=stop.best_metric,
        elapsed_seconds=time.monotonic() - started,
        negative_sampling='fixed OOF scoring rows' if oof is not None else 'resampled each epoch; T only',
        supervision='OOF scores' if oof is not None else 'formal fixed coordinates on public T', history=history)
    return model.cpu().eval()

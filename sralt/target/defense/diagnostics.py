"""Opt-in SAL observations; never feed observations back into model selection."""

from contextlib import contextmanager
import copy
import json
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def isolated(model):
    """Restore Python/NumPy/Torch RNG and every module's mode, including dropout."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, training in modes:
                module.training = training


def auc(logits, labels):
    """Exact rank AUROC with half credit for ties, calculated on CPU."""
    order = logits.argsort()
    scores, y = logits[order], labels[order]
    _, counts = scores.unique_consecutive(return_counts=True)
    ends = counts.cumsum(0)
    ranks = (ends - counts + 1 + ends).float() / 2
    ranks = ranks.repeat_interleave(counts)
    n = y.sum()
    return float((ranks[y.bool()].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


def describe(values):
    return dict(mean=float(values.mean()), variance=float(values.var()),
                quantiles=values.quantile(torch.tensor([0., .01, .1, .5, .9, .99, 1.])).tolist())


class SALDiagnostics:
    def __init__(self, model, sampler, config):
        self.model, self.config = model, config
        self.every = int(config.get('every', 1))
        self.gradient_batches = int(config.get('gradient_batches', 1))
        self.holdout = config.get('holdout', 'none')
        if self.every < 1 or self.gradient_batches < 0 or self.holdout not in {'none', 'surrogate', 'both'}:
            raise ValueError('Invalid SAL diagnostic interval, batch count or holdout scope')
        with isolated(model):
            seed = int(config.get('seed', 1729))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            count = min(int(config.get('pairs', 4096)), sampler.num_samples)
            if count < 2:
                raise ValueError('SAL diagnostics require at least two pairs per class')
            original = sampler.num_samples
            sampler.num_samples = count
            try:
                self.positive, self.negative = sampler.sample()
            finally:
                sampler.num_samples = original
        self.fit_sampler = sampler
        if self.holdout != 'none':
            self.fit_sampler = copy.copy(sampler)
            self.fit_sampler.exclude(self.positive, self.negative)
            if self.holdout == 'both':
                sampler.exclude(self.positive, self.negative)
        with isolated(model):
            original = self.fit_sampler.num_samples
            self.fit_sampler.num_samples = min(count, original)
            try:
                self.train_positive, self.train_negative = self.fit_sampler.sample()
            finally:
                self.fit_sampler.num_samples = original
        self.emit(kind='setup', pairs_per_class=count, holdout=self.holdout,
                  seed=seed, every=self.every, gradient_batches=self.gradient_batches,
                  graph_edges_removed=False, pair_feature_mode=model.defense.pair_feature_mode,
                  loss_kind=model.defense.loss_kind,
                  surrogate_parameters=sum(p.numel() for p in model.defense.surrogate.parameters()))

    @staticmethod
    def emit(**record):
        logging.info('SAL_DIAGNOSTIC %s', json.dumps(record, allow_nan=False))

    def active(self, epoch):
        return epoch == 1 or epoch % self.every == 0

    def _scores(self, defense, embeds, positive, negative):
        pairs = torch.cat((positive, negative), dim=1)
        logits = torch.cat([
            defense.surrogate(defense.pair_features(embeds, batch)).detach().cpu()
            for batch in pairs.split(int(self.config.get('batch_size', 1024)), dim=1)
        ])
        n = positive.size(1)
        labels = torch.cat((torch.ones(n), torch.zeros(n)))
        pos, neg = logits[:n].sigmoid(), logits[n:].sigmoid()
        return dict(pair_feature_mode=defense.pair_feature_mode, loss_kind=defense.loss_kind,
                    defense_objective=float(defense.loss_from_logits(logits[:n], logits[n:])), auc=auc(logits, labels), bce=float(F.binary_cross_entropy_with_logits(logits, labels)),
                    moment_distance=float(torch.linalg.vector_norm(torch.stack((pos.mean()-neg.mean(), pos.var()-neg.var())))),
                    positive_logits=describe(logits[:n]), negative_logits=describe(logits[n:]),
                    positive_probabilities=describe(pos), negative_probabilities=describe(neg),
                    saturated_fraction=float(((logits.sigmoid() < .01) | (logits.sigmoid() > .99)).float().mean()))

    def observe(self, epoch, phase, embedding, force=False):
        if not force and not self.active(epoch):
            return
        with isolated(self.model), torch.no_grad():
            self.model.eval()
            embeds = embedding()
            for mode in ('eval', 'train'):
                self.model.defense.surrogate.train(mode == 'train')
                for split, positive, negative in (
                    ('diagnostic', self.positive, self.negative),
                    ('fit_reference', self.train_positive, self.train_negative)
                ):
                    self.emit(kind='proxy', epoch=epoch, phase=phase, mode=mode, split=split,
                              holdout=self.holdout, **self._scores(self.model.defense, embeds, positive, negative))

    def gradients(self, epoch, batch, components, defense_loss, blocks):
        if not self.active(epoch) or batch >= self.gradient_batches:
            return
        losses = dict(components)
        losses['recommendation'] = sum(components.values())
        losses['weighted_defense'] = self.model.defense.loss_weight * defense_loss
        # One block at a time limits retained gradient storage. autograd.grad does
        # not populate .grad, and retain_graph preserves the real backward below.
        for block, parameters in blocks.items():
            parameters = list(parameters)
            reference = torch.autograd.grad(losses['weighted_defense'], parameters,
                                            retain_graph=True, allow_unused=True)
            for name, loss in losses.items():
                values = (reference if name == 'weighted_defense' else
                          torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))
                support = [(a, b) for a, b in zip(values, reference) if a is not None and b is not None]
                norm = sum(float(a.detach().square().sum()) for a, _ in support) ** .5
                sal_norm = sum(float(b.detach().square().sum()) for _, b in support) ** .5
                dot = sum(float((a.detach() * b.detach()).sum()) for a, b in support)
                self.emit(kind='gradient', epoch=epoch, batch=batch, block=block, component=name,
                          loss_kind=self.model.defense.loss_kind,
                          loss=float(losses[name].detach()), common_parameter_tensors=len(support), norm=norm, weighted_defense_norm=sal_norm,
                          ratio=sal_norm / norm if norm else None,
                          cosine=dot / (norm * sal_norm) if norm and sal_norm else None)
                del values, support
            del reference

    def refit(self, epoch, embedding):
        """Offline copy at the selected checkpoint, never modify the real proxy."""
        epochs = int(self.config.get('refit_epochs', 0))
        if epochs <= 0:
            return
        with isolated(self.model):
            self.model.eval()
            with torch.no_grad():
                embeds = embedding().detach()
            defense = copy.deepcopy(self.model.defense)
            # Reinitialize, so the probe does not inherit past diagnostic labels.
            for module in defense.surrogate.modules():
                if isinstance(module, torch.nn.Linear):
                    module.reset_parameters()
            defense.surrogate_epochs = epochs
            optimizer = defense.create_optimizer()
            self.emit(kind='refit', epoch=epoch, epochs=epochs, holdout=self.holdout,
                      online_bce=defense.fit_surrogate(embeds, self.fit_sampler, optimizer))
            defense.surrogate.eval()
            with torch.no_grad():
                self.emit(kind='proxy', epoch=epoch, phase='checkpoint_refit', mode='eval',
                          split='diagnostic', holdout=self.holdout,
                          **self._scores(defense, embeds, self.positive, self.negative))

"""Frozen-encoder residual vector quantization for the prefix feasibility study."""

import logging
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from sralt.utils import disk_cache, model_fingerprint


class ItemResidualQuantizer(nn.Module):
    """Greedy Euclidean residual codebooks, fitted without relation labels.

    Each stage fits k-means to the preceding stage's residuals. Codebooks are
    persistent buffers; encoding never updates them or depends on user identity.
    """

    def __init__(self, codebooks: torch.Tensor):
        super().__init__()
        self.register_buffer('codebooks', codebooks)

    @torch.no_grad()
    def encode(self, embeddings: torch.Tensor, batch_size: int = 2048):
        paths, errors = [], []
        for batch in embeddings.split(batch_size):
            residual = batch.to(self.codebooks)
            codes, residual_mse = [], []
            for centers in self.codebooks:
                distance = (residual.square().sum(-1, keepdim=True)
                            + centers.square().sum(-1)[None] - 2 * residual @ centers.T)
                indices = distance.argmin(-1)
                residual = residual - centers[indices]
                codes.append(indices)
                residual_mse.append(residual.square().mean(-1))
            paths.append(torch.stack(codes, dim=-1))
            errors.append(torch.stack(residual_mse, dim=-1))
        return torch.cat(paths), torch.cat(errors)

    def __repr__(self):
        return (f'{type(self).__name__}(shape={tuple(self.codebooks.shape)}, '
                f'fingerprint={model_fingerprint(self)})')

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        value = state_dict.get(prefix + 'codebooks')
        if value is not None:
            self.codebooks = self.codebooks.new_empty(value.shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@disk_cache(cache_dir='.cache/shadow/model/ItemResidualQuantizer')
def load_item_rvq_model(*, embeddings: torch.Tensor, item_ids: torch.Tensor,
                        source: dict, seed: int, num_quantizers: int = 8,
                        codebook_size: int = 256, max_iter: int = 50,
                        revision: str = 'item-rvq-lloyd-v1') -> ItemResidualQuantizer:
    """Fit only training-covered items; persist through the project model cache.

    Embeddings, item numbering, upstream identity and all fitting settings are
    part of the cache key. No hidden relation data is accepted by this function.
    """
    started = time.monotonic()
    residual = embeddings.detach().cpu().double().numpy().copy()
    capacity = min(codebook_size, len(residual))
    codebooks, history = [], []
    for level in range(num_quantizers):
        fit = KMeans(n_clusters=capacity, init='k-means++', n_init=1,
                     max_iter=max_iter, tol=1e-4, algorithm='lloyd',
                     random_state=seed + level).fit(residual)
        centers = fit.cluster_centers_
        residual = residual - centers[fit.labels_]
        codebooks.append(torch.from_numpy(centers.copy()))
        history.append(dict(level=level + 1, iterations=int(fit.n_iter_),
                            mse=float(np.square(residual).mean()),
                            active_codes=int(np.unique(fit.labels_).size)))
        logging.info('RVQ level=%d/%d items=%d codes=%d mse=%.9g',
                     level + 1, num_quantizers, len(residual), capacity, history[-1]['mse'])
    model = ItemResidualQuantizer(torch.stack(codebooks)).eval()
    model.fit_summary = dict(revision=revision, seed=seed, source=source,
                             items=len(item_ids), requested_capacity=codebook_size,
                             effective_capacity=capacity, history=history,
                             seconds=time.monotonic() - started)
    return model

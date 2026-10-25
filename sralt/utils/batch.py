"""Batch tensor indexing for recommendation training and evaluation."""

from collections.abc import Iterator
from typing import Protocol

import torch


class BatchTensorDataset(Protocol):
    def __len__(self) -> int: ...

    def get_epoch_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return aligned tensors, refreshing training samples once per epoch."""
        ...


class TensorBatchLoader:
    """Iterate tensors directly, without scalar dataset calls or collation.

    Each iteration requests fresh epoch tensors from the dataset. Training
    datasets resample then; evaluation datasets return their fixed candidates.
    Shuffling uses the tensors' device and every row is visited exactly once,
    including the final partial batch.
    """

    def __init__(self, dataset: BatchTensorDataset, batch_size: int, shuffle: bool = False):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive.')
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        num_samples = len(self.dataset)
        if num_samples == 0:
            return
        tensors = self.dataset.get_epoch_tensors()
        order = torch.randperm(num_samples, device=tensors[0].device) if self.shuffle else None
        for start in range(0, num_samples, self.batch_size):
            end = start + self.batch_size
            indices = order[start:end] if order is not None else slice(start, end)
            yield tuple(tensor[indices] for tensor in tensors)

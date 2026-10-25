"""Socially Adversarial Learning from SMIA, adapted to undirected user pairs."""

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch_geometric.utils import negative_sampling, to_undirected


class SocialRelationClassifier(nn.Module):
    """Surrogate relation classifier on concatenated propagated user embeddings.

    Input width is 2d. This is the author's ``Defender``, independent of the
    black-box SMIA classifier used to evaluate privacy after target training.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """Return relation logits of shape [num_pairs]."""
        return self.net(pair_features).squeeze(-1)


class SocialPairSampler:
    """Sample balanced target-side labels using the complete training social graph.

    All pairs are unique, have u < v, and exclude self-loops. Unlike the public
    attack sampler, defense negatives exclude *all* true social relationships.
    Sparse negative sampling avoids allocating a quadratic user-pair mask.
    """

    def __init__(self, social_edge_index: torch.Tensor, num_users: int, sample_ratio: float):
        self.social_edge_index = to_undirected(social_edge_index, num_nodes=num_users)
        self.positive_pairs = self.social_edge_index[:, self.social_edge_index[0] < self.social_edge_index[1]]
        self.num_users = num_users
        self.num_samples = round(self.positive_pairs.size(1) * sample_ratio)
        available_negatives = num_users * (num_users - 1) // 2 - self.positive_pairs.size(1)
        if not 2 <= self.num_samples <= self.positive_pairs.size(1):
            raise ValueError('SAL needs at least two sampled positive pairs, with sample_ratio <= 1.')
        if self.num_samples > available_negatives:
            raise ValueError('SAL needs as many distinct social non-edges as sampled positive pairs.')

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return independently shuffled positive and negative [2, num_pairs] tensors."""
        device = self.positive_pairs.device
        indices = torch.randperm(self.positive_pairs.size(1), device=device)[:self.num_samples]
        positives = self.positive_pairs[:, indices]
        negatives = negative_sampling(
            self.social_edge_index, num_nodes=self.num_users,
            num_neg_samples=2 * self.num_samples, method='sparse', force_undirected=True
        )
        negatives = negatives[:, negatives[0] < negatives[1]]
        if negatives.size(1) != self.num_samples:
            raise RuntimeError('SAL negative sampling did not produce the requested balanced pair count.')
        order = torch.randperm(self.num_samples, device=device)
        return positives, negatives[:, order]

    def batches(self, num_steps: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Partition a fresh sample; merge singleton tails for sample variance.

        Callers cycle these batches if recommendation training needs more steps.
        A short social sample never truncates the recommendation epoch.
        """
        positives, negatives = self.sample()
        batch_size = max(2, self.num_samples // num_steps)
        sizes = [batch_size] * (self.num_samples // batch_size)
        remainder = self.num_samples % batch_size
        if remainder == 1:
            sizes[-1] += 1
        elif remainder:
            sizes.append(remainder)
        return iter(zip(positives.split(sizes, dim=1), negatives.split(sizes, dim=1)))


class SocialAdversarialLearning(nn.Module):
    """Own the surrogate classifier and SAL training settings as target state.

    ``surrogate_warmup_epochs`` delays only surrogate fitting; the defense loss
    is active from the first recommendation epoch, matching the released code.
    """

    def __init__(
        self, embed_dim: int, loss_weight: float, sample_ratio: float,
        surrogate_warmup_epochs: int, surrogate_epochs: int,
        surrogate_batch_size: int, surrogate_grad_clip: float,
        optimizer: DictConfig
    ):
        super().__init__()
        self.surrogate = SocialRelationClassifier(embed_dim)
        self.surrogate.requires_grad_(False)
        self.loss_weight = loss_weight
        self.sample_ratio = sample_ratio
        self.surrogate_warmup_epochs = surrogate_warmup_epochs
        self.surrogate_epochs = surrogate_epochs
        self.surrogate_batch_size = surrogate_batch_size
        self.surrogate_grad_clip = surrogate_grad_clip
        self.optimizer_config = optimizer

    def create_optimizer(self) -> Optimizer:
        """Create the independent surrogate optimizer once per target training run."""
        return instantiate(self.optimizer_config, params=self.surrogate.parameters())

    @staticmethod
    def pair_features(user_embeds: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """Concatenate endpoints of canonical user pairs into [num_pairs, 2d]."""
        return torch.cat([user_embeds[pairs[0]], user_embeds[pairs[1]]], dim=-1)

    def fit_surrogate(
        self, user_embeds: torch.Tensor, sampler: SocialPairSampler, optimizer: Optimizer
    ) -> float:
        """Fit BCE on a detached embedding snapshot and a fresh balanced sample."""
        positives, negatives = sampler.sample()
        pairs = torch.cat([positives, negatives], dim=1)
        # Construct features per batch instead of materializing O(|E| * d) storage.
        user_embeds = user_embeds.detach()
        labels = torch.cat([
            user_embeds.new_ones(positives.size(1)), user_embeds.new_zeros(negatives.size(1))
        ])
        self.surrogate.train()
        self.surrogate.requires_grad_(True)
        total_loss, total_samples = 0.0, 0
        for _ in range(self.surrogate_epochs):
            order = torch.randperm(pairs.size(1), device=pairs.device)
            for indices in order.split(self.surrogate_batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = self.surrogate(self.pair_features(user_embeds, pairs[:, indices]))
                loss = F.binary_cross_entropy_with_logits(logits, labels[indices])
                loss.backward()
                nn.utils.clip_grad_norm_(self.surrogate.parameters(), self.surrogate_grad_clip)
                optimizer.step()
                total_loss += loss.item() * indices.numel()
                total_samples += indices.numel()
        optimizer.zero_grad(set_to_none=True)
        self.surrogate.requires_grad_(False)
        return total_loss / total_samples

    def distribution_loss(
        self, user_embeds: torch.Tensor, positive_pairs: torch.Tensor, negative_pairs: torch.Tensor
    ) -> torch.Tensor:
        """Match means and sample variances of the surrogate probabilities.

        vector_norm gives the author's exact distance, with a finite zero
        subgradient when both moments match. Dropout remains active in training.
        Frozen surrogate weights still propagate gradients to target embeddings.
        """
        positive_probabilities = self.surrogate(self.pair_features(user_embeds, positive_pairs)).sigmoid()
        negative_probabilities = self.surrogate(self.pair_features(user_embeds, negative_pairs)).sigmoid()
        differences = torch.stack([
            positive_probabilities.mean() - negative_probabilities.mean(),
            positive_probabilities.var() - negative_probabilities.var()
        ])
        return torch.linalg.vector_norm(differences)

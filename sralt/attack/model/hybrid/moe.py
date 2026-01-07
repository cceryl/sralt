import typing

import torch
import torch.nn as nn


class Expert(nn.Module):
    """
    Expert network for MoE architecture.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        """
        :param input_dim: Input feature dimension.
        :param hidden_dim: Hidden layer dimension.
        :param dropout: Dropout rate.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input features of shape (batch_size, input_dim).
        :return: Logits of shape (batch_size, 1).
        """
        return self.net(x)


class GatingNetwork(nn.Module):
    """
    Gating network to dynamically combine expert outputs based on user characteristics and sample density.
    """

    def __init__(self, input_dim: int, num_experts: int, noise_multiplier: float = 0.0):
        """
        :param input_dim: Dimension of individual user features.
        :param num_experts: Number of experts in MoE.
        """
        super().__init__()

        self.main_gate = nn.Linear(input_dim, num_experts)
        self.density_bias = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, num_experts)
        )
        self.noise_multiplier = noise_multiplier

    def forward(self, x: torch.Tensor, sample_density: torch.Tensor) -> torch.Tensor:
        """
        :param x: Combined user features of shape (batch_size, input_dim).
        :param sample_density: Sample density of shape (batch_size, 2) indicating the probability of sampling each user.
        :return: Gating weights of shape (batch_size, num_experts).
        """
        symmetric_sample_density = torch.stack([
            sample_density[:, 0] + sample_density[:, 1],
            torch.abs(sample_density[:, 0] - sample_density[:, 1])
        ], dim=1)

        logits = self.main_gate(x) + self.density_bias(symmetric_sample_density)

        if self.noise_multiplier != 0.0 and self.training:
            noise = torch.randn_like(logits) * self.noise_multiplier
            logits = logits + noise

        gate_weights = torch.softmax(logits, dim=1)
        return gate_weights


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module to combine multiple experts.
    """

    def __init__(self, num_experts: int, input_dim: int, hidden_dim: int, dropout: float, noise_multiplier: float = 0.0):
        """
        :param num_experts: Number of experts.
        :param input_dim: Input feature dimension for each expert.
        :param hidden_dim: Hidden layer dimension for each expert.
        :param dropout: Dropout rate for each expert.
        :param noise_multiplier: Noise multiplier for gating network.
        """
        super().__init__()

        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, dropout) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts, noise_multiplier)

    def forward(self, x: torch.Tensor, sample_density: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input features of shape (batch_size, input_dim).
        :param sample_density: Sample density of shape (batch_size, 2) indicating the probability of sampling each user.
        :return: Combined output of shape (batch_size, 1).
        """
        expert_outputs = torch.stack(
            [typing.cast(torch.Tensor, expert(x)).squeeze(-1) for expert in self.experts],
            dim=1
        )
        gate_weights = self.gating_network(x, sample_density)
        return torch.sum(expert_outputs * gate_weights, dim=1, keepdim=True)

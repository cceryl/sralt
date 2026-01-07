import torch
import torch.nn as nn
from abc import abstractmethod


class BaseFusion(nn.Module):
    """
    Base class for feature fusion modules.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        :param feat1: User1 features of shape (batch_size, feature_dim).
        :param feat2: User2 features of shape (batch_size, feature_dim).
        :return: Fused features of shape (batch_size, fused_feature_dim).
        """
        pass

    @abstractmethod
    def fused_feature_dim(self, feature_dim: int) -> int:
        """
        :param feature_dim: Dimension of individual user features.
        :return: Dimension of fused features.
        """
        pass


fusions: dict[str, type[BaseFusion]] = {}


def register_fusion(name: str):
    def decorator(cls: type[BaseFusion]):
        fusions[name] = cls
        return cls
    return decorator


def get_fusion(name: str) -> type[BaseFusion]:
    if name not in fusions:
        raise ValueError(f'Fusion "{name}" not found.')
    return fusions[name]


@register_fusion('sum')
class SumFusion(BaseFusion):
    """
    Sum-based feature fusion for pairwise features.
    """

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        return feat1 + feat2

    def fused_feature_dim(self, feature_dim: int) -> int:
        return feature_dim


@register_fusion('diff')
class DifferenceFusion(BaseFusion):
    """
    Absolute difference-based feature fusion for pairwise features.
    """

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        return torch.abs(feat1 - feat2)

    def fused_feature_dim(self, feature_dim: int) -> int:
        return feature_dim


@register_fusion('product')
class ProductFusion(BaseFusion):
    """
    Element-wise product-based feature fusion for pairwise features.
    """

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        return feat1 * feat2

    def fused_feature_dim(self, feature_dim: int) -> int:
        return feature_dim


@register_fusion('bilinear')
class BilinearFusion(BaseFusion):
    """
    Bilinear kernel-based feature fusion for pairwise features.
    """

    def __init__(self, feature_dim: int, **kwargs):
        super().__init__()
        self.bilinear = nn.Bilinear(feature_dim, feature_dim, feature_dim, bias=True)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        return self.bilinear(feat1, feat2)

    def fused_feature_dim(self, feature_dim: int) -> int:
        return feature_dim


@register_fusion('low-rank')
class LowRankFusion(BaseFusion):
    """
    Low-rank factorization-based feature fusion for pairwise features.
    """

    def __init__(self, feature_dim: int, rank: int, **kwargs):
        """
        :param feature_dim: Dimension of individual user features.
        :param rank: Rank for low-rank factorization.
        """
        super().__init__()

        self.rank = rank
        self.U = nn.Parameter(torch.randn(feature_dim, rank) / (rank ** 0.5))
        self.V = nn.Parameter(torch.randn(feature_dim, rank) / (rank ** 0.5))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        feat1_proj = feat1 @ self.U
        feat2_proj = feat2 @ self.V
        interaction = (feat1_proj * feat2_proj).sum(dim=1, keepdim=True)
        low_rank_feat = interaction * (feat1 + feat2) / 2 + self.bias

        return low_rank_feat

    def fused_feature_dim(self, feature_dim: int) -> int:
        return feature_dim


@register_fusion('attention')
class AttentionFusion(BaseFusion):
    """
    Attention-based encoder for feature fusion.
    """

    def __init__(self, feature_dim: int, num_heads: int, **kwargs):
        """
        :param feature_dim: Dimension of individual user features.
        :param num_heads: Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim

        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        stacked_feat = torch.stack([feat1, feat2], dim=1)

        attn_output, _ = self.attention(stacked_feat, stacked_feat, stacked_feat)
        attn_output = self.proj(attn_output)

        attn_feat1, attn_feat2 = attn_output[:, 0], attn_output[:, 1]
        return torch.cat([attn_feat1 + attn_feat2, torch.abs(attn_feat1 - attn_feat2)], dim=1)

    def fused_feature_dim(self, feature_dim: int) -> int:
        return 2 * feature_dim

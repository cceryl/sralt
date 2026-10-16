import torch
from torch_geometric.utils import coalesce

from .smia import SMIA
from sralt.shadow.data import ShadowData


class MINER(SMIA):
    """Shadow adapter for the social MINER baseline released with SMIA.

    Training and recommendation evaluation follow SRALT's SMIA model. Attack
    features use the learned social embedding table, as in the authors'
    ``mia_attack.py``, rather than propagated or fused embeddings.
    Reference: https://github.com/XuHao-bit/SMIA/tree/a7e79411244a574ba0b539510f105094a8956682
    """

    @torch.no_grad()
    def get_user_features(self, data: ShadowData) -> torch.Tensor:
        """Pack each user's embedding and their K recommended item embeddings.

        The result has shape ``[num_users, (K + 1) * embed_dim]``. Item vectors
        follow increasing global item ID, since SRALT exposes recommendation
        sets without ranks. All K queried items are available to the attacker.
        """
        self.eval()
        edges = coalesce(data.behavioral_edge_index, num_nodes=self.num_nodes)
        counts = torch.bincount(edges[0], minlength=self.num_users)
        if not torch.all(counts == data.num_recommendations):
            raise ValueError('MINER requires exactly K distinct recommended items per user.')

        recommended_items = edges[1].reshape(self.num_users, data.num_recommendations)
        embeddings = self.social_branch.embedding.weight
        item_features = embeddings[recommended_items].reshape(self.num_users, -1)
        return torch.cat([embeddings[:self.num_users], item_features], dim=1)

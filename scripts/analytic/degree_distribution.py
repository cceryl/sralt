import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import degree

from sralt.target import load_target_data
from sralt.utils import get_device


def main():
    path = './runs/analytic'
    os.makedirs(path, exist_ok=True)

    seed = 42
    deterministic = True
    device = get_device(allow_share=True)

    target_data = load_target_data(
        seed=seed,
        deterministic=deterministic,
        name='Ciao',
        directory='./data/ciao',
        config={
            'min_rating': 2,
            'min_social_connections': 1,
            'min_user_interactions': 5,
            'min_item_interactions': 1
        }
    ).to(device)

    social = target_data.social_edge_index
    deg = degree(social[0], num_nodes=target_data.num_users)
    deg = deg.cpu().numpy()

    # plot
    hist, bins = np.histogram(deg, bins=100)

    x = (bins[:-1] + bins[1:]) / 2
    y = hist

    x = x[y > 0]
    y = y[y > 0]

    plt.figure()
    plt.loglog(x, y, 'o')
    plt.xlabel("degree k")
    plt.ylabel("count")
    plt.title("Degree distribution (log-log)")
    plt.savefig(os.path.join(path, 'degree_distribution.png'))

    # triangles
    def count_triangles(edge_index: torch.Tensor, num_nodes: int):
        adj: list[set[int]] = [set() for _ in range(num_nodes)]
        u: int
        v: int
        for u, v in edge_index.t().tolist():
            if u != v:
                adj[u].add(v)
                adj[v].add(u)

        triangles = 0
        for u, v in edge_index.t().tolist():
            if u < v:
                common = adj[u].intersection(adj[v])
                triangles += len(common)

        return triangles // 3

    print(f'Social nodes: {target_data.num_users}, total_edges: {social.size(1)}')
    print(f'Triangles in social: {count_triangles(social, target_data.num_users)}')


if __name__ == '__main__':
    main()

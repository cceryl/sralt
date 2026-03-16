import os
import torch
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
    user_degree = degree(social[0], num_nodes=target_data.num_users)
    directed_social = social[:, social[0] < social[1]]
    print(f'User degree: min={user_degree.min().item()}, max={user_degree.max().item()}, mean={user_degree.mean().item():.4f}')
    print(f'User edges in directed social graph: {directed_social.size(1)}')

    u, v = directed_social
    linear_scores = user_degree[u] * user_degree[v] / 2.0 / directed_social.size(1)
    exp_scores = 1 - torch.exp(-linear_scores)
    log_exp_scores = torch.log(exp_scores)

    print(f'Linear scores: min={linear_scores.min():.4f}, max={linear_scores.max():.4f}, mean={linear_scores.mean():.4f}')
    print(f'Exponential scores: min={exp_scores.min():.4f}, max={exp_scores.max():.4f}, mean={exp_scores.mean():.4f}')
    print(f'LogExp scores: min={log_exp_scores.min():.4f}, max={log_exp_scores.max():.4f}, mean={log_exp_scores.mean():.4f}')

    # sort and plot
    save_path = os.path.join(path, 'edge_score_distribution.png')
    sorted_linear_scores, _ = torch.sort(linear_scores)
    sorted_exp_scores, _ = torch.sort(exp_scores)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sorted_linear_scores.cpu().numpy(), label='Linear Scores')
    plt.title('Sorted Linear Scores')
    plt.xlabel('Edge Index')
    plt.ylabel('Score')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(sorted_exp_scores.cpu().numpy(), label='Exponential Scores', color='orange')
    plt.title('Sorted Exponential Scores')
    plt.xlabel('Edge Index')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Saved score distribution plot to {save_path}')



if __name__ == '__main__':
    main()

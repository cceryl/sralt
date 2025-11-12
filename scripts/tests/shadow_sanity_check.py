# This file conducts sanity tests for the shadow data generation process, the dataset class, and a shadow model.

import torch

from sralt.utils import set_seed, get_device, setup_logging
from sralt.target.data import load_dataset
from sralt.target.model import create_target_model
from sralt.shadow.data import ShadowData, TrainShadowDataset, EvalShadowDataset
from sralt.shadow.model import create_shadow_model

model_path = './runs/design_ciao_test/model.pth'

set_seed(42)
setup_logging(tz='Asia/Shanghai')
device = get_device()


data = load_dataset(
    name='Ciao',
    directory='./data/ciao',
    config={
        'min_rating': 2,
        'min_social_connections': 1,
        'min_user_interactions': 5,
        'min_item_interactions': 0
    }
).to(device)

model = create_target_model(
    name='DESIGN',
    num_users=data.num_users,
    num_items=data.num_items,
    config={
        'embed_dim': 64,
        'user_fusion_layers': [0, 1, 2],
        'item_fusion_layers': [0, 1, 2],
        'decay': 5e-3,
        'distill_weight': 0.1
    }
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))


shadow_data = ShadowData(
    target_data=data,
    target_model=model,
    shadow_social_ratio=0.1,
    num_recommendations=30,
).to(device)

print('Shadow data representation:')
print(shadow_data)

# Verify the shapes
assert shadow_data.num_users == data.num_users
assert shadow_data.num_items == data.num_items
assert shadow_data.num_social_edges == int(data.num_social_edges * 0.1)
assert shadow_data.behavioral_train_edge_index.size(1) == 28 * data.num_users
assert shadow_data.behavioral_val_edge_index.size(1) == data.num_users
assert shadow_data.behavioral_test_edge_index.size(1) == data.num_users
print('Shape checks passed.')

# Check that the social edges are a subset of the target data
social_edges_set = set(
    (u, v) for u, v in data.social_edge_index.t().tolist()
)
for u, v in shadow_data.social_edge_index.t().tolist():
    assert (u, v) in social_edges_set or (v, u) in social_edges_set
print('Social edges subset check passed.')

# Check that the behavioral edges do not intersect with the target data behavioral edges
behavioral_edges_set = set(
    (u, v) for u, v in data.behavioral_edge_index.t().tolist()
)
for u, v in shadow_data.behavioral_edge_index.t().tolist():
    assert (u, v) not in behavioral_edges_set
print('Behavioral edges disjoint check passed.')


train_dataset = TrainShadowDataset(shadow_data)
val_dataset = EvalShadowDataset(shadow_data, split='val', num_negative_samples=999)
test_dataset = EvalShadowDataset(shadow_data, split='test', num_negative_samples=999)

print('Datasets representations:')
print(train_dataset)
print(val_dataset)
print(test_dataset)

shadow_model = create_shadow_model(
    name='SMIA',
    num_users=shadow_data.num_users,
    num_items=shadow_data.num_items,
    config={
        'embed_dim': 64,
        'num_layers': 2,
        'decay': 5e-3
    }
).to(device)

print('Shadow model representation:')
print(shadow_model)

# Forward pass test
train_scores = shadow_model(
    social_edge_index=shadow_data.social_edge_index,
    behavioral_edge_index=shadow_data.behavioral_train_edge_index,
    target_edge_index=shadow_data.behavioral_train_edge_index
)
assert train_scores.size(0) == 2
assert train_scores.size(1) == shadow_data.behavioral_train_edge_index.size(1)
print('Forward pass on training data check passed.')

# Get embedding test
embeddings = shadow_model.get_embedding(
    social_edge_index=shadow_data.social_edge_index,
    behavioral_edge_index=shadow_data.behavioral_train_edge_index
)
assert embeddings.size(0) == shadow_data.num_users + shadow_data.num_items
assert embeddings.size(1) == 2 * 64
print('Get embedding check passed.')

# Get attack features test
attack_features = shadow_model.get_attack_features(
    social_edge_index=shadow_data.social_edge_index,
    behavioral_edge_index=shadow_data.behavioral_train_edge_index
)
assert attack_features.size(0) == shadow_data.num_users
assert attack_features.size(1) == 4 * 64
print('Get attack features check passed.')

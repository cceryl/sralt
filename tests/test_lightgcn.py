import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from torch_geometric.nn import LightGCN as PyGLightGCN
from torch_geometric.utils import to_undirected

from sralt.target.data import TargetData
from sralt.target.model import create_target_model, load_target_model, test_target_model
from sralt.target.model.lightgcn import LightGCN
from sralt.utils import model_fingerprint, set_seed


def make_data(device: torch.device = torch.device('cpu')) -> TargetData:
    data = TargetData(name='synthetic', directory='unused')
    data.num_users, data.num_items = 3, 8
    data.social_edge_index = torch.tensor([[0, 1], [1, 0]])
    data.behavioral_train_edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [3, 4, 4, 5, 5, 6]])
    data.behavioral_val_edge_index = torch.tensor([[0, 1, 2], [5, 6, 7]])
    data.behavioral_test_edge_index = torch.tensor([[0, 1, 2], [6, 7, 8]])
    return data.to(device)


class LightGCNTests(unittest.TestCase):
    def setUp(self):
        set_seed(42, deterministic=True)
        with initialize_config_module(config_module='sralt.configs', version_base=None):
            self.config = compose(config_name='config', overrides=[
                'target/data=ciao', 'target/model=lightgcn', 'shadow/model=csa', 'attack/model=mlp',
                'target.model.attribute.embed_dim=4', 'target.model.train.num_epochs=2',
                'target.model.train.train_batch_size=4', 'target.model.train.val_batch_size=2',
                'target.model.train.val_num_negatives=3', 'target.model.train.metrics.0.k=2',
                'target.model.train.metrics.1.k=2', 'target.test.evaluation.test_batch_size=2',
                'target.test.evaluation.test_num_negatives=3', 'target.test.metrics.0.k=2',
                'target.test.metrics.1.k=2'
            ])

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def test_propagation_matches_dense_reference_and_backward(self):
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        for device in devices:
            for deterministic in [False, True]:
                with self.subTest(device=device, deterministic=deterministic):
                    set_seed(42, deterministic=deterministic)
                    data = make_data(device)
                    graph = to_undirected(data.behavioral_train_edge_index)
                    model = LightGCN(3, 8, 4, 3, 1e-4).to(device)
                    torch.manual_seed(42)
                    pyg_model = PyGLightGCN(11, 4, 3).to(device)
                    torch.testing.assert_close(model.gcn.embedding.weight, pyg_model.embedding.weight)
                    dense = torch.zeros(11, 11, device=device)
                    dense[graph[1], graph[0]] = 1
                    inv_degree = dense.sum(dim=1).clamp(min=1).rsqrt()
                    dense = inv_degree[:, None] * dense * inv_degree[None, :]
                    initial = model.gcn.embedding.weight.detach().clone().requires_grad_()
                    layers = [initial]
                    for _ in range(3):
                        layers.append(dense @ layers[-1])
                    expected = torch.stack(layers).mean(dim=0)
                    actual = model.get_embedding(graph)
                    torch.testing.assert_close(actual, expected)
                    edges = data.behavioral_val_edge_index
                    actual_scores = model(graph, edges)
                    expected_scores = (expected[edges[0]] * expected[edges[1]]).sum(dim=-1)
                    torch.testing.assert_close(actual_scores, expected_scores)
                    (actual.square().sum() + actual_scores.square().sum()).backward()
                    (expected.square().sum() + expected_scores.square().sum()).backward()
                    torch.testing.assert_close(model.gcn.embedding.weight.grad, initial.grad)
                    # Isolated nodes retain their layer-0 contribution.
                    torch.testing.assert_close(actual[10], initial[10] / 4)

    def test_bpr_regularizes_initial_embeddings_with_repeated_nodes(self):
        model = LightGCN(3, 8, 4, 2, 0.1)
        positive = torch.tensor([0.5, 1.0], requires_grad=True)
        negative = torch.tensor([0.2, -0.5], requires_grad=True)
        nodes = torch.tensor([0, 0, 3, 4, 5, 5])
        expected = F.softplus(negative - positive).mean()
        expected += 0.1 * sum(model.gcn.embedding.weight[node].square().sum() for node in nodes) / positive.size(0)
        loss = model.calculate_loss(positive, negative, nodes)
        torch.testing.assert_close(loss, expected)
        expected_gradients = torch.autograd.grad(
            expected, (positive, negative, model.gcn.embedding.weight), retain_graph=True
        )
        loss.backward()
        for actual, reference in zip(
            (positive.grad, negative.grad, model.gcn.embedding.weight.grad), expected_gradients
        ):
            torch.testing.assert_close(actual, reference)
        model.decay = 0
        torch.testing.assert_close(model.calculate_loss(positive, negative, nodes), F.softplus(negative - positive).mean())

    def test_zero_layers_and_graph_cache_lifecycle(self):
        torch.use_deterministic_algorithms(False)
        model = LightGCN(3, 8, 4, 0, 1e-4)
        graph = to_undirected(make_data().behavioral_train_edge_index)
        torch.testing.assert_close(model.get_embedding(graph), model.gcn.embedding.weight)
        old_adj = model._cached_adj
        self.assertIs(model._build_adj(graph), old_adj)
        new_graph = to_undirected(make_data().behavioral_edge_index)
        self.assertIsNot(model._build_adj(new_graph), old_adj)
        self.assertIs(model._cached_edge_index, new_graph)
        model.to('cpu')
        self.assertIsNone(model._cached_adj)
        self.assertIsNone(model._cached_edge_index)
        self.assertFalse(any('cached' in key for key in model.state_dict()))

    def test_training_evaluation_and_deployment_ignore_social_graph(self):
        for deterministic in [False, True]:
            with self.subTest(deterministic=deterministic):
                set_seed(42, deterministic=deterministic)
                data = make_data()
                del data.social_edge_index  # Any social graph access must fail.
                model = create_target_model(
                    42, deterministic, torch.device('cpu'), self.config.target.model.attribute, 3, 8
                )
                original = model.gcn.embedding.weight.detach().clone()
                train_graph = to_undirected(data.behavioral_train_edge_index)
                with patch.object(model, '_build_adj', wraps=model._build_adj) as build_adj:
                    model.train_model(data=data, writer=None, **self.config.target.model.train)
                    for call in build_adj.call_args_list:
                        torch.testing.assert_close(call.args[0], train_graph)
                self.assertFalse(torch.equal(original, model.gcn.embedding.weight))
                with patch.object(model, 'get_embedding', wraps=model.get_embedding) as get_embedding:
                    values = test_target_model(
                        42, deterministic, torch.device('cpu'), model, data, self.config.target.test
                    )
                    torch.testing.assert_close(get_embedding.call_args.args[0], train_graph)
                self.assertEqual(len(values), 2)
                self.assertTrue(all(0 <= value <= 1 for value in values))

                fingerprint = model_fingerprint(model)
                with patch.object(model, 'get_embedding', wraps=model.get_embedding) as get_embedding:
                    recommended = model.recommend(data=data, num_recommendations=2, batch_size=2)
                    torch.testing.assert_close(get_embedding.call_args.args[0], to_undirected(data.behavioral_edge_index))
                self.assertEqual(recommended.shape, (2, 6))
                self.assertTrue(torch.equal(torch.bincount(recommended[0], minlength=3), torch.full((3,), 2)))
                self.assertTrue(((recommended[1] >= 3) & (recommended[1] < 11)).all())
                history = set(map(tuple, data.behavioral_edge_index.t().tolist()))
                self.assertFalse(history.intersection(map(tuple, recommended.t().tolist())))
                self.assertEqual(model_fingerprint(model), fingerprint)

                restored = LightGCN(3, 8, 4, 3, 1e-4)
                restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                torch.testing.assert_close(restored.recommend(data=data, num_recommendations=2, batch_size=1), recommended)
                self.assertEqual(repr(model), repr(restored))

    def test_model_disk_cache_miss_and_hit(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as directory:
            previous_cwd = os.getcwd()
            try:
                os.chdir(directory)
                with patch.dict(os.environ, {'SRALT_CACHE_VERSION': 'lightgcn-test', 'SRALT_CACHE_REFRESH': '0'}):
                    model = load_target_model(42, False, torch.device('cpu'), self.config.target.model, data)
                    with patch.object(LightGCN, 'train_model', side_effect=AssertionError('Cache hit retrained model')):
                        cached = load_target_model(42, False, torch.device('cpu'), self.config.target.model, data)
                    self.assertEqual(repr(model), repr(cached))
                    torch.testing.assert_close(
                        model.recommend(data=data, num_recommendations=2, batch_size=2),
                        cached.recommend(data=data, num_recommendations=2, batch_size=2)
                    )
            finally:
                os.chdir(previous_cwd)


if __name__ == '__main__':
    unittest.main()

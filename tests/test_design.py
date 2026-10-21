import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from torch_geometric.utils import to_undirected

from sralt.target.data import TargetData
from sralt.target.model import create_target_model, load_target_model, test_target_model
from sralt.target.model.design import DESIGN
from sralt.target.model.design.layers import MeanGraph
from sralt.utils import model_fingerprint, set_seed


def make_data():
    data = TargetData(name='synthetic', directory='unused')
    data.num_users, data.num_items = 3, 8
    data.social_edge_index = torch.tensor([[0, 1], [1, 0]])
    data.behavioral_train_edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [3, 4, 4, 5, 5, 6]])
    data.behavioral_val_edge_index = torch.tensor([[0, 1, 2], [5, 6, 7]])
    data.behavioral_test_edge_index = torch.tensor([[0, 1, 2], [6, 7, 8]])
    return data


def config():
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        return compose(config_name='config', overrides=[
            'target/data=ciao', 'target/model=design', 'shadow/model=smia', 'attack/model=mlp',
            'target.model.attribute.embed_dim=4', 'target.model.train.num_epochs=2',
            'target.model.train.train_batch_size=4', 'target.model.train.val_batch_size=2',
            'target.model.train.val_num_negatives=3', 'target.model.train.metrics.0.k=2',
            'target.model.train.metrics.1.k=2', 'target.test.evaluation.test_batch_size=2',
            'target.test.evaluation.test_num_negatives=3', 'target.test.metrics.0.k=2',
            'target.test.metrics.1.k=2'
        ])


def author_reference(tables, social, behavior, targets, training, user_layers=(1, 2), item_layers=(0,)):
    """Independent dense equations from the DESIGN supplementary implementation.

    Delete batch edges physically, then compute simultaneous two-relation means.
    This reference does not use the production graph or convolution helpers.
    """
    num_users, num_nodes = 3, tables[0].size(0)
    s = torch.zeros(num_users, num_users, dtype=tables[0].dtype)
    s[social[1], social[0]] = 1
    s = s / s.sum(1, keepdim=True).clamp_min(1)
    a = torch.zeros(num_nodes, num_nodes, dtype=tables[0].dtype)
    a[behavior[1], behavior[0]] = 1
    if training:
        for u, v in targets.t().tolist():
            a[u, v] = a[v, u] = 0
    degrees = a.sum(1)
    self_weight = (1 - degrees.float() / (degrees.float() + 1e-8)).to(a.dtype)
    a = a / degrees[:, None].clamp_min(1)

    def rating_step(x):
        x = a @ x + self_weight[:, None] * x
        return a @ x + self_weight[:, None] * x

    social_table, rating_table, twin_table = tables
    social_layers = [social_table[:num_users]]
    rating_layers = [rating_table]
    twin_social_layers, twin_rating_layers = [twin_table[:num_users]], [twin_table]
    for _ in range(2):
        social_layers.append(s @ social_layers[-1])
        rating_layers.append(rating_step(rating_layers[-1]))
        twin_social_layers.append(s @ twin_social_layers[-1])
        twin_rating_layers.append(rating_step(twin_rating_layers[-1]))
    social_embeds = torch.cat([sum(social_layers[i] for i in user_layers), social_table[num_users:]])
    rating_embeds = torch.cat([sum(rating_layers[i][:num_users] for i in user_layers),
                               sum(rating_layers[i][num_users:] for i in item_layers)])
    twin_user_layers = [twin_table[:num_users]] + [
        .5 * twin_social_layers[i] + .5 * twin_rating_layers[i][:num_users] for i in (1, 2)]
    twin_embeds = torch.cat([sum(twin_user_layers[i] for i in user_layers), twin_table[num_users:]])
    u, v = targets
    scores = torch.stack([(embeds[u] * embeds[v]).sum(1)
                          for embeds in (social_embeds, rating_embeds, twin_embeds)], 1)
    return scores, twin_embeds


def author_loss(scores, labels, distill_weight=1.):
    probabilities = scores.sigmoid()
    loss = sum(F.binary_cross_entropy(probabilities[:, i], labels.to(scores.dtype)) for i in range(3))
    for teacher in range(3):
        for student in range(3):
            if teacher != student:
                target = probabilities[:, teacher].detach()
                prediction = probabilities[:, student]
                loss = loss - distill_weight * (target * prediction.log() +
                    (1 - target) * (1 - prediction).log()).mean()
    return loss


class DESIGNTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.are_deterministic_algorithms_enabled()
        set_seed(42, deterministic=True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous)

    def test_three_branches_masked_forward_and_gradients_match_author_equations(self):
        data = make_data()
        behavior = to_undirected(data.behavioral_train_edge_index)
        targets = torch.tensor([[0, 0, 1, 2, 0, 1], [3, 4, 5, 6, 8, 9]])
        for deterministic in (False, True):
            for training in (False, True):
                for user_layers, item_layers in (([1, 2], [0]), ([0, 2], [0, 1])):
                    with self.subTest(deterministic=deterministic, training=training, user_layers=user_layers):
                        torch.use_deterministic_algorithms(deterministic)
                        model = DESIGN(3, 8, 4, user_layers, item_layers, 1.).double().train(training)
                        tables = [branch.embeds.weight.detach().clone().requires_grad_()
                                  for branch in (model.social, model.rating, model.twin)]
                        expected, twin = author_reference(tables, data.social_edge_index, behavior, targets,
                                                         training, user_layers, item_layers)
                        actual = model(data.social_edge_index, behavior, targets)
                        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
                        actual.square().sum().backward()
                        expected.square().sum().backward()
                        for branch, table in zip((model.social, model.rating, model.twin), tables):
                            torch.testing.assert_close(branch.embeds.weight.grad, table.grad, rtol=1e-11, atol=1e-14)
                        if not training:
                            torch.testing.assert_close(model.get_embedding(data.social_edge_index, behavior), twin)

    def test_shared_mask_matches_physical_deletion_and_reuses_csr_indices(self):
        edges = to_undirected(make_data().behavioral_train_edge_index)
        template = MeanGraph(edges, 11, torch.float64)
        full = template.propagation()
        targets = torch.tensor([[0, 0, 1, 0, 0, 2], [3, 4, 5, 3, 10, 6]])
        before = edges.clone()
        masked = template.propagation(targets)
        self.assertEqual(full.adjacency.crow_indices().data_ptr(), masked.adjacency.crow_indices().data_ptr())
        self.assertEqual(full.adjacency.col_indices().data_ptr(), masked.adjacency.col_indices().data_ptr())
        dense = full.adjacency.to_dense()
        adjacency = torch.zeros(11, 11, dtype=torch.float64)
        adjacency[edges[1], edges[0]] = 1
        for u, v in targets.t().tolist():
            adjacency[u, v] = adjacency[v, u] = 0
        degree = adjacency.sum(1)
        expected = adjacency / degree[:, None].clamp_min(1)
        torch.testing.assert_close(masked.adjacency.to_dense(), expected)
        self.assertEqual(masked.self_weight[0], 1.)
        self.assertEqual(masked.self_weight[1], 0.)
        torch.testing.assert_close(full.adjacency.to_dense(), dense)
        torch.testing.assert_close(edges, before)
        self.assertIs(template.propagation(), full)
        empty = MeanGraph(torch.empty(2, 0, dtype=torch.long), 11, torch.float64).propagation(targets)
        torch.testing.assert_close(empty.self_weight, torch.ones(11, dtype=torch.float64))
        torch.testing.assert_close(empty.adjacency.to_dense(), torch.zeros(11, 11, dtype=torch.float64))

    def test_pointwise_bce_distillation_and_one_adam_match_author(self):
        data = make_data()
        behavior = to_undirected(data.behavioral_train_edge_index)
        targets = torch.tensor([[0, 1, 2, 0, 1, 2], [3, 4, 6, 8, 9, 10]])
        labels = torch.tensor([1., 1., 1., 0., 0., 0.], dtype=torch.float64)
        model = DESIGN(3, 8, 4, [1, 2], [0], 1.).double()
        references = [torch.nn.Parameter(branch.embeds.weight.detach().clone())
                      for branch in (model.social, model.rating, model.twin)]
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        author_optimizers = [torch.optim.Adam([table], lr=.01) for table in references]
        for _ in range(2):
            optimizer.zero_grad()
            for opt in author_optimizers:
                opt.zero_grad()
            scores = model(data.social_edge_index, behavior, targets)
            reference_scores, _ = author_reference(references, data.social_edge_index, behavior, targets, True)
            actual, expected = model.calculate_loss(scores, labels), author_loss(reference_scores, labels)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
            actual.backward()
            expected.backward()
            for branch, table in zip((model.social, model.rating, model.twin), references):
                torch.testing.assert_close(branch.embeds.weight.grad, table.grad, rtol=1e-10, atol=1e-14)
            optimizer.step()
            for opt in author_optimizers:
                opt.step()
            for branch, table in zip((model.social, model.rating, model.twin), references):
                torch.testing.assert_close(branch.embeds.weight, table, rtol=1e-10, atol=1e-14)
        extreme = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
        loss = model.calculate_loss(extreme, torch.ones(1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_isolated_branch_scaling_raw_twin_items_and_initialization(self):
        model = DESIGN(100, 200, 16, [1, 2], [0], 1.)
        for branch in (model.social, model.rating, model.twin):
            self.assertAlmostEqual(branch.embeds.weight.std().item(), .01, delta=.001)
        model = DESIGN(3, 8, 4, [1, 2], [0], 1.).double()
        data = make_data()
        empty = torch.empty(2, 0, dtype=torch.long)
        social, rating = model._prepare_graphs(data.social_edge_index, empty)
        output = model.rating.get_embedding(rating)
        torch.testing.assert_close(output[:3], model.rating.embeds.weight[:3] * 2)
        twin = model.twin.get_embedding(social, rating)
        torch.testing.assert_close(twin[3:], model.twin.embeds.weight[3:])
        # User 2 is isolated socially; its rating representation survives both layers.
        torch.testing.assert_close(twin[2], model.twin.embeds.weight[2])

    def test_training_validation_deployment_boundaries_and_strict_reload(self):
        cfg, data = config(), make_data()
        for deterministic in (False, True):
            with self.subTest(deterministic=deterministic):
                model = create_target_model(42, deterministic, torch.device('cpu'), cfg.target.model.attribute, 3, 8)
                before = model_fingerprint(model)
                contexts = []
                with patch.object(model, '_prepare_graphs', wraps=model._prepare_graphs) as prepare:
                    with patch.object(model.rating, 'get_embedding', wraps=model.rating.get_embedding) as rating_forward:
                        model.train_model(data=data, writer=None, **cfg.target.model.train)
                    self.assertEqual(rating_forward.call_count, 4)  # 2 epochs x ceil(6/4), one pass per batch.
                    contexts = list(prepare.call_args_list)
                train_graph = to_undirected(data.behavioral_train_edge_index)
                for call in contexts:
                    torch.testing.assert_close(call.args[1], train_graph)
                self.assertEqual(sum(len(call.args) == 3 and call.args[2] is not None for call in contexts), 4)
                self.assertNotEqual(before, model_fingerprint(model))
                with patch.object(model, '_prepare_graphs', wraps=model._prepare_graphs) as prepare:
                    values = test_target_model(42, deterministic, torch.device('cpu'), model, data, cfg.target.test)
                    self.assertTrue(all(0 <= value <= 1 for value in values))
                    torch.testing.assert_close(prepare.call_args.args[1], train_graph)
                    self.assertEqual(len(prepare.call_args.args), 2)
                fingerprint = model_fingerprint(model)
                with patch.object(model, '_prepare_graphs', wraps=model._prepare_graphs) as prepare:
                    recommended = model.recommend(data=data, num_recommendations=2, batch_size=2)
                    torch.testing.assert_close(prepare.call_args.args[1], to_undirected(data.behavioral_edge_index))
                self.assertEqual(recommended.shape, (2, 6))
                history = set(map(tuple, data.behavioral_edge_index.t().tolist()))
                self.assertFalse(history & set(map(tuple, recommended.t().tolist())))
                self.assertEqual(fingerprint, model_fingerprint(model))
                restored = DESIGN(3, 8, 4, [1, 2], [0], 1.)
                restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                torch.testing.assert_close(restored.recommend(data=data, num_recommendations=2, batch_size=1), recommended)
                model.to('cpu')
                self.assertFalse(model._graphs)
                self.assertFalse(any('graph' in key for key in model.state_dict()))

    def test_disk_cache_and_revision_preserve_unified_training_config(self):
        cfg = config()
        self.assertEqual(cfg.target.model.attribute.user_fusion_layers, [1, 2])
        self.assertEqual(cfg.target.model.attribute.item_fusion_layers, [0])
        self.assertNotIn('decay', cfg.target.model.attribute)
        data = make_data()
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                model = load_target_model(42, False, torch.device('cpu'), cfg.target.model, data)
                with patch.object(DESIGN, 'train_model', side_effect=AssertionError('Cache hit retrained')):
                    cached = load_target_model(42, False, torch.device('cpu'), cfg.target.model, data)
                self.assertEqual(model_fingerprint(model), model_fingerprint(cached))
                cfg.target.model.revision = 'another-revision'
                with patch.object(DESIGN, 'train_model', side_effect=AssertionError('Revision cache miss')):
                    with self.assertRaisesRegex(AssertionError, 'Revision cache miss'):
                        load_target_model(42, False, torch.device('cpu'), cfg.target.model, data)
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()

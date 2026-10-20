import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from torch_geometric.utils import to_undirected

from scripts.pipeline import shadow_stage, attack_stage
from sralt.attack.data import AttackData, EvalAttackDataset
from sralt.attack.model import load_attack_model
from sralt.attack.model.mlp import MLP
from sralt.shadow.data import load_shadow_data
from sralt.shadow.model import load_shadow_model
from sralt.shadow.model.smia import SMIA
from sralt.target.data import TargetData
from sralt.target.model.lightgcn import LightGCN
from sralt.utils import model_fingerprint, set_seed


def config():
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        return compose(config_name='config', overrides=[
            'target/data=ciao', 'target/model=lightgcn', 'shadow/model=smia', 'attack/model=mlp'
        ])


def data():
    result = TargetData(name='synthetic', directory='unused')
    result.num_users, result.num_items = 12, 20
    users = torch.arange(result.num_users)
    result.social_edge_index = to_undirected(torch.cat([
        torch.stack([users, (users + 1) % 12]), torch.stack([users, (users + 2) % 12])
    ], dim=1))
    items = 12 + (2 * users[:, None] + torch.arange(5)) % 20
    result.behavioral_train_edge_index = torch.stack([users.repeat_interleave(3), items[:, :3].flatten()])
    result.behavioral_val_edge_index = torch.stack([users, items[:, 3]])
    result.behavioral_test_edge_index = torch.stack([users, items[:, 4]])
    return result


def row_mean(edges, n, dtype=torch.float64):
    adj = torch.zeros(n, n, dtype=dtype)
    adj[edges[1], edges[0]] = 1
    return adj / adj.sum(1, keepdim=True).clamp(min=1)


def pairs(edges):
    return {tuple(sorted(pair)) for pair in edges.t().tolist()}


class SMIATests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.are_deterministic_algorithms_enabled()
        set_seed(42, deterministic=True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous)

    def test_author_row_propagation_raw_social_items_and_gradients(self):
        social = to_undirected(torch.tensor([[0, 1], [1, 2]]))
        behavior = to_undirected(torch.tensor([[0, 0, 1, 2], [4, 5, 4, 6]]))
        for deterministic in (False, True):
            with self.subTest(deterministic=deterministic):
                torch.use_deterministic_algorithms(deterministic)
                model = SMIA(4, 5, 3, 2, .1).double()
                initial_s = model.social_branch.embedding.weight.detach().clone().requires_grad_()
                initial_a = model.behavioral_branch.embedding.weight.detach().clone().requires_grad_()
                s, a = row_mean(social, 4), row_mean(behavior, 9)
                expected_s = torch.cat([(initial_s[:4] + s @ initial_s[:4] + s @ s @ initial_s[:4]) / 3,
                                        initial_s[4:]])
                expected_a = (initial_a + a @ initial_a + a @ a @ initial_a) / 3
                expected = torch.cat([expected_s, expected_a], dim=1)
                actual = model.get_embedding(social, behavior)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual[4:, :3], initial_s[4:])
                torch.testing.assert_close(actual[3, :3], initial_s[3] / 3)
                actual.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(model.social_branch.embedding.weight.grad, initial_s.grad)
                torch.testing.assert_close(model.behavioral_branch.embedding.weight.grad, initial_a.grad)

    def test_author_loss_scaling_and_separate_initialization(self):
        model = SMIA(4, 5, 3, 2, .1).double()
        for branch in (model.social_branch, model.behavioral_branch):
            self.assertLessEqual(branch.embedding.weight[:4].abs().max(), (6 / (4 + 3)) ** .5)
            self.assertLessEqual(branch.embedding.weight[4:].abs().max(), (6 / (5 + 3)) ** .5)
        positive = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        negative = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        nodes = torch.tensor([0, 0, 2, 4, 5, 5, 7, 8, 8])
        expected = sum(F.softplus(negative[i] - positive[i]).mean() for i in range(2))
        for branch in (model.social_branch, model.behavioral_branch):
            expected = expected + .1 * .5 * sum(
                branch.embedding.weight[chunk].square().mean() for chunk in nodes.chunk(3))
        actual = model.calculate_loss(positive, negative, nodes)
        torch.testing.assert_close(actual, expected)
        params = (positive, negative, *model.parameters())
        gradients = torch.autograd.grad(actual, params, retain_graph=True)
        references = torch.autograd.grad(expected, params)
        for gradient, reference in zip(gradients, references):
            torch.testing.assert_close(gradient, reference)

    def test_raw_attack_features_use_all_queries_and_preserve_state(self):
        cfg, target_data = config(), data()
        cfg.shadow.data.num_recommendations = 4
        target = LightGCN(12, 20, 4, 1, .001)
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                shadow_data = load_shadow_data(42, True, target_data, target, cfg.shadow.data)
            finally:
                os.chdir(previous)
        model = SMIA(12, 20, 4, 2, .01)
        before = model_fingerprint(model)
        with patch.object(model.social_branch, 'get_embedding', side_effect=AssertionError('Propagation')):
            features = model.get_user_features(shadow_data)
        social, behavior = model.social_branch.embedding.weight, model.behavioral_branch.embedding.weight
        for user in range(12):
            items = shadow_data.behavioral_edge_index[1, shadow_data.behavioral_edge_index[0] == user]
            fused = (social[items] + behavior[items]) / 2
            expected = torch.cat([behavior[user], social[user], fused.mean(0), fused.var(0)])
            torch.testing.assert_close(features[user], expected)
        self.assertFalse(features.requires_grad)
        self.assertEqual(before, model_fingerprint(model))

    def test_split_is_made_once_before_shadow_and_validation_excludes_all_public(self):
        cfg, target_data = config(), data()
        cfg.shadow.data.shadow_social_ratio = .5
        cfg.shadow.data.num_recommendations = 4
        cfg.attack.data.val_ratio = .25
        target = LightGCN(12, 20, 4, 1, .001)
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                shadow = load_shadow_data(42, True, target_data, target, cfg.shadow.data)
            finally:
                os.chdir(previous)
        public, training, validation = map(pairs, (shadow.public_social_edge_index,
            shadow.social_edge_index, shadow.attack_val_social_edge_index))
        self.assertEqual(len(public), 12)
        self.assertEqual(len(training), 9)
        self.assertEqual(len(validation), 3)
        self.assertFalse(training & validation)
        self.assertEqual(training | validation, public)
        attack = AttackData(target_data, shadow, SMIA(12, 20, 4, 2, .01), .25)
        self.assertEqual(pairs(attack.train_social_edge_index), training)
        self.assertEqual(pairs(attack.val_social_edge_index), validation)
        self.assertEqual(pairs(attack.test_social_edge_index), pairs(target_data.social_edge_index) - public)
        self.assertFalse(pairs(EvalAttackDataset(attack, 'val').edge_index[:, 3:]) & public)
        shadow.to(torch.device('cpu'))
        self.assertEqual(pairs(shadow.public_social_edge_index), public)
        with self.assertRaisesRegex(ValueError, 'validation ratio'):
            AttackData(target_data, shadow, SMIA(12, 20, 4, 2, .01), .1)

    def test_shadow_cache_depends_on_resolved_attack_validation_ratio(self):
        cfg, target_data = config(), data()
        cfg.shadow.data.shadow_social_ratio = .5
        cfg.shadow.data.num_recommendations = 4
        target = LightGCN(12, 20, 4, 1, .001)
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                first = load_shadow_data(42, True, target_data, target, cfg.shadow.data)
                with patch('sralt.shadow.data.data.ShadowData.__init__', side_effect=AssertionError('Cache miss')):
                    cached = load_shadow_data(42, True, target_data, target, cfg.shadow.data)
                torch.testing.assert_close(first.attack_val_social_edge_index, cached.attack_val_social_edge_index)
                cfg.attack.data.val_ratio = .5
                second = load_shadow_data(42, True, target_data, target, cfg.shadow.data)
                self.assertNotEqual(first.attack_val_social_edge_index.size(1), second.attack_val_social_edge_index.size(1))
                self.assertNotEqual(repr(first), repr(second))
            finally:
                os.chdir(previous)

    def test_author_mlp_probabilities_bce_and_clipped_sgd_update(self):
        cfg = config()
        model = MLP(8, .5, .2, 1.).double()
        reference = copy.deepcopy(model)
        self.assertFalse(any(isinstance(layer, torch.nn.BatchNorm1d) for layer in model.net))
        features = torch.randn(6, 8, dtype=torch.float64)
        edge = torch.tensor([[0, 2, 1, 4], [1, 1, 3, 5]])
        u, v = torch.minimum(*edge), torch.maximum(*edge)
        expected_pair = torch.cat([features[u], features[v], abs(features[u] - features[v]), features[u] * features[v]], 1)
        labels = torch.tensor([1., 0., 1., 0.], dtype=torch.float64)
        optimizer = instantiate(cfg.attack.model.train.optimizer, params=model.parameters())
        baseline = torch.optim.SGD(reference.parameters(), lr=.05, momentum=.9, weight_decay=0.)
        for _ in range(2):
            optimizer.zero_grad()
            baseline.zero_grad()
            torch.manual_seed(42)
            loss = model.loss_fn(model(edge, features), labels)
            torch.manual_seed(42)
            expected = F.binary_cross_entropy(torch.sigmoid(reference.net(expected_pair).squeeze(1)), labels)
            torch.testing.assert_close(loss, expected)
            loss.backward()
            expected.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.)
            optimizer.step()
            baseline.step()
            for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, expected_parameter)
        model.eval()
        torch.testing.assert_close(model.predict(edge, features), model.predict(edge.flip(0), features))
        self.assertEqual(cfg.attack.model.attribute.pos_weight, 1.)
        self.assertEqual(cfg.attack.model.train.train_batch_size, 512)
        self.assertEqual(cfg.attack.model.train.num_epochs, 50)

    def test_short_training_checkpoint_reload_and_cache_round_trip(self):
        cfg, target_data = config(), data()
        cfg.shadow.data.shadow_social_ratio = .5
        cfg.shadow.data.num_recommendations = 4
        cfg.attack.data.val_ratio = .25
        cfg.attack.tail_ratio = .5
        for stage in ('shadow', 'attack'):
            cfg[stage].model.train.num_epochs = 2
            cfg[stage].model.train.train_batch_size = 5
        cfg.shadow.model.attribute.embed_dim = 4
        cfg.shadow.model.train.val_num_negatives = 4
        cfg.shadow.test.evaluation.test_num_negatives = 4
        for metric in (*cfg.shadow.model.train.metrics, *cfg.shadow.test.metrics):
            metric.k = 2
        device = torch.device('cpu')
        for deterministic in (False, True):
            with self.subTest(deterministic=deterministic), tempfile.TemporaryDirectory() as directory:
                previous = os.getcwd()
                try:
                    os.chdir(directory)
                    target = LightGCN(12, 20, 4, 1, .001)
                    shadow_data, shadow = shadow_stage(42, deterministic, device, None, cfg.shadow, target_data, target)
                    with patch.object(SMIA, 'train_model', side_effect=AssertionError('Retrained shadow')):
                        cached_shadow = load_shadow_model(42, deterministic, device, cfg.shadow.model, shadow_data)
                    self.assertEqual(model_fingerprint(cached_shadow), model_fingerprint(shadow))
                    restored = SMIA(12, 20, 4, 2, .01)
                    restored.load_state_dict(shadow.state_dict(), strict=True)
                    torch.testing.assert_close(restored.get_user_features(shadow_data), shadow.get_user_features(shadow_data))
                    attack_data, model = attack_stage(42, deterministic, device, None, cfg.attack, target_data, shadow_data, shadow)
                    with patch.object(MLP, 'train_model', side_effect=AssertionError('Retrained attack')):
                        cached = load_attack_model(42, deterministic, device, cfg.attack.model, attack_data, .5, None)
                    self.assertEqual(model_fingerprint(cached), model_fingerprint(model))
                    restored_attack = MLP(16, .5, .2, 1.).eval()
                    restored_attack.load_state_dict(model.state_dict(), strict=True)
                    model.eval()
                    torch.testing.assert_close(restored_attack.predict(attack_data.test_social_edge_index, attack_data.user_features),
                                               model.predict(attack_data.test_social_edge_index, attack_data.user_features))
                finally:
                    os.chdir(previous)


if __name__ == '__main__':
    unittest.main()

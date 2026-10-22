import copy
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from torch_geometric.utils import to_undirected
from torchmetrics.classification import BinaryAUROC, BinaryRecall

from sralt.shadow.data import ShadowData
from sralt.shadow.model import load_shadow_model
from sralt.shadow.model.tail_proxy import TailProxyShadow
from sralt.attack.data import AttackData, load_attack_data
from sralt.attack.model import load_attack_model, test_attack_model as evaluate_attack
from sralt.attack.model.tail_proxy import TailProxyAttack
from sralt.attack.report import save_attack_report
from sralt.target.data import TargetData
from sralt.target.model.lightgcn import LightGCN
from sralt.utils import model_fingerprint, set_seed
from sralt.utils.attack_groups import grouped_attack_report


def configuration():
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        cfg = compose(config_name='config', overrides=[
            'target/data=ciao', 'target/model=design', 'shadow/model=tail_proxy', 'attack/model=tail_proxy'])
    cfg.shadow.model.attribute.embed_dim = 4
    cfg.shadow.model.attribute.num_quantizers = 2
    cfg.shadow.model.attribute.codebook_size = 4
    cfg.attack.model.attribute.hidden_dim = 16
    cfg.attack.data.val_ratio = .25
    for stage in ('shadow', 'attack'):
        cfg[stage].model.train.num_epochs = 2
        cfg[stage].model.train.train_batch_size = 16
        cfg[stage].model.train.val_batch_size = 8
        cfg[stage].test.evaluation.test_batch_size = 8
    cfg.shadow.model.train.val_num_negatives = 4
    cfg.shadow.test.evaluation.test_num_negatives = 4
    for metric in (*cfg.shadow.model.train.metrics, *cfg.shadow.test.metrics):
        metric.k = 2
    return cfg


def datasets():
    target = TargetData(name='synthetic', directory='unused')
    target.num_users, target.num_items = 12, 20
    users = torch.arange(12)
    target.social_edge_index = to_undirected(torch.cat([
        torch.stack([users, (users + 1) % 12]), torch.stack([users, (users + 2) % 12])], dim=1))
    items = 12 + (2 * users[:, None] + torch.arange(5)) % 20
    target.behavioral_train_edge_index = torch.stack([users.repeat_interleave(3), items[:, :3].flatten()])
    target.behavioral_val_edge_index = torch.stack([users, items[:, 3]])
    target.behavioral_test_edge_index = torch.stack([users, items[:, 4]])
    target.to(torch.device('cpu'))
    shadow = ShadowData(target, LightGCN(12, 20, 4, 1, .001), .5, 4, 8, .25)
    return target, shadow


class TailProxyTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.are_deterministic_algorithms_enabled()
        set_seed(42, deterministic=True)
        torch.set_num_threads(1)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous)

    def test_idf_weighted_graph_moments_and_conditioning(self):
        model = TailProxyShadow(3, 3, 2, 2, rvq_enabled=False)
        edges = torch.tensor([[0, 0, 1, 1, 2, 2], [3, 4, 3, 5, 3, 5]])
        idf = model.recommendation_idf(torch.cat([edges, edges[:, :1]], 1))
        torch.testing.assert_close(idf, torch.log(torch.tensor([1., 2., 4 / 3])))
        adj = model.behavior_adj(edges, idf)
        dense = torch.zeros(6, 6)
        dense[edges[0], edges[1]] = idf[edges[1] - 3]
        dense = dense + dense.T
        inverse = dense.sum(1).clamp_min(1e-12).rsqrt()
        expected = inverse[:, None] * dense * inverse[None, :]
        torch.testing.assert_close(adj.to_dense(), expected)
        initial = model.behavioral_branch.embedding.weight
        embeds = (initial + expected @ initial + expected @ expected @ initial) / 3
        torch.testing.assert_close(model.behavioral_branch(adj), embeds)
        mean, variance = model.aggregate(embeds, edges, idf)
        torch.testing.assert_close(mean, embeds[torch.tensor([4, 5, 5])])
        torch.testing.assert_close(variance, torch.zeros_like(variance))
        actual = model.behavior(adj, edges, idf)
        torch.testing.assert_close(actual[1], (embeds[:3] + mean) / 2)
        actual[0].square().sum().backward()
        self.assertTrue(torch.isfinite(initial.grad).all())
        uniform = TailProxyShadow(3, 3, 2, 2, rvq_enabled=False, idf_weighting=False)
        uniform.load_state_dict(model.state_dict())
        self.assertFalse(torch.allclose(uniform.behavior_adj(edges, idf).to_dense(), expected))
        mixed_edges = torch.tensor([[0, 0, 1, 1, 2], [3, 4, 4, 5, 5]])
        mixed_idf = model.recommendation_idf(mixed_edges)
        mixed_mean, mixed_var = model.aggregate(embeds, mixed_edges, mixed_idf)
        for user in range(3):
            items = mixed_edges[1, mixed_edges[0] == user]
            weights = mixed_idf[items - 3]
            expected_mean = (weights[:, None] * embeds[items]).sum(0) / weights.sum()
            expected_var = (weights[:, None] * (embeds[items] - expected_mean).square()).sum(0) / weights.sum()
            torch.testing.assert_close(mixed_mean[user], expected_mean)
            torch.testing.assert_close(mixed_var[user], expected_var)
        all_common = torch.tensor([[0, 1, 2], [3, 3, 3]])
        zero_idf = model.recommendation_idf(all_common)
        empty_information = model.behavior(model.behavior_adj(all_common, zero_idf), all_common, zero_idf)
        self.assertTrue(all(torch.isfinite(x).all() for x in empty_information))

    def initialized_shadow(self, shadow):
        model = TailProxyShadow(12, 20, 4, 2, num_quantizers=2, codebook_size=4)
        idf, edges = model.recommendation_idf(shadow.behavioral_edge_index), shadow.behavioral_train_edge_index
        model.behavior(model.behavior_adj(edges, idf), edges, idf)  # Initialize RVQ in training mode.
        model.eval()
        return model

    def test_semantic_sparse_graph_leakage_and_strict_reload(self):
        _, shadow = datasets()
        model = self.initialized_shadow(shadow)
        before = model_fingerprint(model)
        features, payload = model.get_attack_payload(shadow)
        self.assertEqual(model_fingerprint(model), before)
        self.assertEqual(payload['codes'].shape, (12, 2))
        self.assertEqual(model.last_diagnostics['semantic_edges'], 24)
        self.assertEqual(model.last_diagnostics['augmented_zero_degree'], 0)
        for usage in model.last_diagnostics['code_usage']:
            self.assertGreater(usage['active_codes'], 1)
        semantic = model.semantic_edges(payload['codes'])
        only_semantic = model.social_branch(model.social_adj(shadow.social_edge_index[:, :0], semantic))[:12]
        torch.testing.assert_close(only_semantic, payload['sparse_social'])
        altered = copy.deepcopy(shadow)
        altered.public_social_edge_index = torch.tensor([[0, 1], [6, 9]])
        altered.attack_val_social_edge_index = altered.public_social_edge_index
        # These held-out fields cannot affect any feature or semantic edge.
        changed, changed_payload = model.get_attack_payload(altered)
        torch.testing.assert_close(features, changed)
        for key in payload:
            torch.testing.assert_close(payload[key], changed_payload[key])
        clone = TailProxyShadow(12, 20, 4, 2, num_quantizers=2, codebook_size=4)
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        restored, restored_payload = clone.get_attack_payload(shadow)
        torch.testing.assert_close(features, restored)
        for key in payload:
            torch.testing.assert_close(payload[key], restored_payload[key])
        # Removing public edges changes only the social block, not behavior or codes.
        altered.social_edge_index = altered.social_edge_index[:, :0]
        sparse_features, _ = model.get_attack_payload(altered)
        torch.testing.assert_close(sparse_features[:, 4:8], payload['sparse_social'])
        torch.testing.assert_close(sparse_features[:, :4], features[:, :4])
        torch.testing.assert_close(sparse_features[:, 8:], features[:, 8:])

    def test_attack_inputs_normalization_symmetry_gradients_and_ablations(self):
        target, shadow = datasets()
        data = AttackData(target, shadow, self.initialized_shadow(shadow), .25)
        model = TailProxyAttack(16, num_quantizers=2, hidden_dim=16, dropout=0.)
        model.eval()
        edges = torch.tensor([[0, 1, 4], [1, 5, 8]])
        norms = model.endpoint_features(edges, data).norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))
        scaled = copy.deepcopy(data)
        scaled.user_features = (scaled.user_features.reshape(12, 4, 4) *
                                torch.rand(12, 4, 1).add(.1) * 100).reshape(12, 16)
        torch.testing.assert_close(model(edges, data), model(edges, scaled))
        torch.testing.assert_close(model(edges, data), model(edges.flip(0), data))
        sparse_mask = torch.tensor([[True, False, True], [False, True, True]])
        torch.testing.assert_close(model(edges, data, sparse_mask), model(edges.flip(0), data, sparse_mask.flip(0)))
        evidence = model.recommendation_evidence(edges, data)
        for j, (u, v) in enumerate(edges.T.tolist()):
            left, right = (set(data.side_features['recommendations'][x].tolist()) for x in (u, v))
            expected = sum(float(data.side_features['idf'][i]) for i in left & right)
            self.assertAlmostEqual(float(evidence[j, 1]), expected, places=5)
            self.assertEqual(float(evidence[j, 0]), len(left & right))
        agreement = (data.side_features['codes'][edges[0]] == data.side_features['codes'][edges[1]]).float()
        torch.testing.assert_close(model.code_evidence(edges, data), agreement)
        model(edges, data).sum().backward()
        self.assertGreater(model.alignment[0].weight.grad.abs().sum(), 0)
        self.assertGreater(model.continuous[0].weight.grad.abs().sum(), 0)
        # The two semantic channels actually enter the alignment layer.
        with patch.object(model.alignment, 'forward', wraps=model.alignment.forward) as forward:
            model(edges, data)
            torch.testing.assert_close(forward.call_args.args[0], torch.cat([agreement, evidence], -1))
        for normalization, codes, idf, sparse in itertools.product((False, True), repeat=4):
            attack = TailProxyAttack(16, num_quantizers=2, normalize_features=normalization,
                code_matching=codes, idf_pair_signal=idf, sparse_training=sparse, hidden_dim=16)
            attack(edges, data, sparse_mask if sparse else None).sum().backward()
        # Sampling is label-independent and honors probability endpoints exactly.
        model.sparse_probability = 1
        self.assertTrue(model.sample_sparse_mask(edges).all())
        model.sparse_probability = 0
        self.assertFalse(model.sample_sparse_mask(edges).any())
        model.sparse_probability = .5
        self.assertAlmostEqual(float(model.sample_sparse_mask(torch.zeros(2, 10000)).float().mean()), .5, delta=.02)

    def test_cpu_training_cache_payload_reports_and_rvq_off(self):
        cfg = configuration()
        target, shadow = datasets()
        device = torch.device('cpu')
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                model = load_shadow_model(42, True, device, cfg.shadow.model, shadow)
                restored = load_shadow_model(42, True, device, cfg.shadow.model, shadow)
                self.assertEqual(model_fingerprint(model), model_fingerprint(restored))
                before_test = model_fingerprint(model)
                with patch.object(model, 'behavior', wraps=model.behavior) as behavior_call:
                    model.test_model(data=shadow, metrics=instantiate(cfg.shadow.test.metrics),
                                     **cfg.shadow.test.evaluation)
                    torch.testing.assert_close(behavior_call.call_args.args[1], shadow.behavioral_train_edge_index)
                self.assertEqual(before_test, model_fingerprint(model))
                data = load_attack_data(42, True, target, shadow, model, cfg.attack.data)
                key = data.cache_key()
                self.assertIs(key[-1], data.side_features)
                attack = load_attack_model(42, True, device, cfg.attack.model, data, .5, None)
                restored_attack = load_attack_model(42, True, device, cfg.attack.model, data, .5, None)
                self.assertEqual(model_fingerprint(attack), model_fingerprint(restored_attack))
                self.assertGreater(sum(x['positive_masked_endpoints'] for x in attack.training_diagnostics), 0)
                self.assertGreater(sum(x['negative_masked_endpoints'] for x in attack.training_diagnostics), 0)
                result = evaluate_attack(42, True, device, attack, data, cfg.attack.test, .5)
                self.assertTrue(all(torch.isfinite(torch.tensor(group)).all() for group in result))
                save_attack_report(data, attack, directory)
                report = json.loads(Path(directory, 'attack_diagnostics.json').read_text())
                self.assertIn('TT', report['groups'])
                self.assertIn('isolated_Tail', report['groups'])
                features, payload = model.get_attack_payload(shadow)
                loaded_data = load_attack_data(42, True, target, shadow, model, cfg.attack.data).to(device)
                torch.testing.assert_close(features, loaded_data.user_features)
                for key in payload:
                    torch.testing.assert_close(payload[key], loaded_data.side_features[key])
                clone = instantiate(cfg.attack.model.attribute, user_feature_dim=16)
                clone.load_state_dict(attack.state_dict(), strict=True)
                clone.eval()
                torch.testing.assert_close(clone.predict(data.train_social_edge_index, data),
                                           attack.predict(data.train_social_edge_index, data))
                # RVQ-off automatically disables its dependent graph and attack inputs.
                cfg.shadow.model.attribute.rvq_enabled = False
                cfg.shadow.model.train.num_epochs = 1
                plain = load_shadow_model(42, True, device, cfg.shadow.model, shadow)
                plain_data = AttackData(target, shadow, plain, .25)
                self.assertEqual(plain_data.side_features['codes'].shape, (12, 0))
                self.assertEqual(plain.last_diagnostics['semantic_edges'], 0)
                plain_attack = instantiate(cfg.attack.model.attribute, user_feature_dim=16)
                plain_attack(plain_data.train_social_edge_index, plain_data).sum().backward()
                # Semantic-only ablation preserves direct RVQ codes.
                cfg.shadow.model.attribute.rvq_enabled = True
                cfg.shadow.model.attribute.semantic_augmentation = False
                no_semantic = load_shadow_model(42, True, device, cfg.shadow.model, shadow)
                _, no_semantic_payload = no_semantic.get_attack_payload(shadow)
                self.assertEqual(no_semantic_payload['codes'].shape, (12, 2))
                self.assertEqual(no_semantic.last_diagnostics['semantic_edges'], 0)
            finally:
                os.chdir(previous)

    def test_group_weights_empty_groups_and_undefined_roc(self):
        data = SimpleNamespace(num_users=4, shadow_social_edge_index=torch.tensor([[0, 1], [1, 0]]))
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        report = grouped_attack_report(edges, torch.tensor([.9, .5, .8]), torch.tensor([1, 0, 1]),
            data, torch.tensor([0]), torch.tensor([1, 2, 3]), [BinaryAUROC(), BinaryRecall()])
        self.assertEqual(report['HH']['samples'], 0)
        self.assertIsNone(report['HH']['AUROC'])
        self.assertEqual(report['TT']['samples'], 2)
        self.assertEqual(report['Tail']['samples'], 5)
        self.assertEqual(report['isolated_Tail']['samples'], 3)
        self.assertEqual(report['non_isolated_Tail']['samples'], 2)
        self.assertIsNone(report['HT']['AUROC'])
        self.assertAlmostEqual(report['All']['predicted_positive_rate'], 2 / 3)


if __name__ == '__main__':
    unittest.main()

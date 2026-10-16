import copy
import io
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from torch_geometric.utils import to_undirected

from scripts.pipeline import target_stage, shadow_stage, attack_stage
from sralt.attack.data import load_attack_data
from sralt.attack.model import create_attack_model, load_attack_model
from sralt.attack.model.miner import MINER, SGDWithGradientClipping
from sralt.shadow.data import ShadowData
from sralt.shadow.model import create_shadow_model, load_shadow_model
from sralt.shadow.model.miner import MINER as ShadowMINER
from sralt.target.data import TargetData
from sralt.utils import model_fingerprint, set_seed


def make_target_data() -> TargetData:
    data = TargetData(name='synthetic', directory='unused')
    data.num_users, data.num_items = 12, 20
    users = torch.arange(data.num_users)
    data.social_edge_index = to_undirected(torch.cat([
        torch.stack([users, (users + 1) % data.num_users]),
        torch.stack([users, (users + 2) % data.num_users])
    ], dim=1))
    items = data.num_users + (2 * users[:, None] + torch.arange(5)) % data.num_items
    data.behavioral_train_edge_index = torch.stack([users.repeat_interleave(3), items[:, :3].flatten()])
    data.behavioral_val_edge_index = torch.stack([users, items[:, 3]])
    data.behavioral_test_edge_index = torch.stack([users, items[:, 4]])
    return data


def make_shadow_data() -> ShadowData:
    data = ShadowData.__new__(ShadowData)
    TargetData.__init__(data, name='synthetic', directory='unused')
    data.num_users, data.num_items, data.num_recommendations = 3, 5, 3
    data.social_edge_index = torch.tensor([[0, 1], [1, 0]])
    # Deliberately interleave users and item IDs across the three splits.
    data.behavioral_train_edge_index = torch.tensor([[2, 0, 1], [7, 5, 3]])
    data.behavioral_val_edge_index = torch.tensor([[1, 2, 0], [6, 4, 7]])
    data.behavioral_test_edge_index = torch.tensor([[0, 2, 1], [3, 5, 4]])
    return data


def reference_distances(items: list[list[float]], user: list[float]) -> list[float]:
    """Scalar distance definitions, independent of the batched tensor code."""
    l1, l2, cosine, bray = [], [], [], []
    for item in items:
        distance = sum(abs(a - b) for a, b in zip(item, user))
        l1.append(distance)
        l2.append(math.dist(item, user))
        dot = sum(a * b for a, b in zip(item, user))
        norms = math.sqrt(sum(a * a for a in item) * sum(b * b for b in user))
        cosine.append(1 - dot / norms)
        bray.append(distance / sum(abs(a + b) for a, b in zip(item, user)))
    return l1 + l2 + cosine + bray


def compose_config(dataset: str = 'ciao', mode: str = 'miner'):
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        return compose(config_name='config', overrides=[
            f'target/data={dataset}', 'target/model=lightgcn',
            'shadow/model=miner', f'attack/model={mode}'
        ])


class MINERTests(unittest.TestCase):
    def setUp(self):
        self.previous_determinism = torch.are_deterministic_algorithms_enabled()
        set_seed(42, deterministic=True)
        self.data = make_shadow_data()
        self.shadow = ShadowMINER(3, 5, 2, 2, 1e-2).double()
        self.embeddings = torch.tensor([
            [0.5, -1.0], [1.5, 2.0], [-0.5, 3.0],
            [2.0, 1.0], [-1.0, 0.5], [3.0, -2.0], [1.0, 4.0], [-2.0, 1.5]
        ], dtype=torch.float64)
        with torch.no_grad():
            self.shadow.social_branch.embedding.weight.copy_(self.embeddings)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous_determinism)

    def test_packing_uses_raw_social_embeddings_and_all_public_recommendations(self):
        fingerprint = model_fingerprint(self.shadow)
        with patch.object(self.shadow.social_branch, 'get_embedding', side_effect=AssertionError('Propagated features')):
            features = self.shadow.get_user_features(self.data)
        expected_items = [[3, 5, 7], [3, 4, 6], [4, 5, 7]]
        for user, items in enumerate(expected_items):
            expected = torch.cat([self.embeddings[user], self.embeddings[items].flatten()])
            torch.testing.assert_close(features[user], expected)
        self.assertEqual(features.shape, (3, 8))
        self.assertFalse(features.requires_grad)
        self.assertEqual(model_fingerprint(self.shadow), fingerprint)

        # Feature order is independent of the leave-one-out split and edge order.
        reordered = copy.deepcopy(self.data)
        reordered.behavioral_train_edge_index = self.data.behavioral_test_edge_index.flip(1)
        reordered.behavioral_test_edge_index = self.data.behavioral_train_edge_index.flip(1)
        reordered.behavioral_val_edge_index = self.data.behavioral_val_edge_index.flip(1)
        with torch.no_grad():
            self.shadow.behavioral_branch.embedding.weight.add_(100)
        torch.testing.assert_close(self.shadow.get_user_features(reordered), features)

    def test_missing_recommendations_and_wrong_feature_contract_fail(self):
        self.data.behavioral_test_edge_index = self.data.behavioral_test_edge_index[:, :-1]
        with self.assertRaisesRegex(ValueError, 'exactly K'):
            self.shadow.get_user_features(self.data)
        with self.assertRaisesRegex(ValueError, 'shadow/model=miner'):
            MINER(4 * 64, 64, 30, False, 0.5, 0.2)

    def test_distance_features_match_scalar_reference_for_both_modes(self):
        packed = self.shadow.get_user_features(self.data)
        recommendations = [[3, 5, 7], [3, 4, 6], [4, 5, 7]]
        pairs = torch.tensor([[0, 2, 1], [1, 0, 2]])
        for bidirectional in [False, True]:
            with self.subTest(bidirectional=bidirectional):
                model = MINER(8, 2, 3, bidirectional, 0.5, 0.2).double().eval()
                expected = []
                for first, second in pairs.t().tolist():
                    first, second = sorted([first, second])
                    row = reference_distances(
                        self.embeddings[recommendations[first]].tolist(), self.embeddings[second].tolist()
                    )
                    if bidirectional:
                        row += reference_distances(
                            self.embeddings[recommendations[second]].tolist(), self.embeddings[first].tolist()
                        )
                    expected.append(row)
                actual = model.get_attack_features(pairs, packed)
                torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))
                self.assertEqual(actual.shape, (3, 24 if bidirectional else 12))
                torch.testing.assert_close(model.predict(pairs, packed), model.predict(pairs.flip(0), packed))
                self.assertEqual(model.predict(pairs[:, :1], packed).shape, (1,))
                self.assertFalse(any(isinstance(layer, torch.nn.BatchNorm1d) for layer in model.net))

    def test_zero_norm_and_zero_bray_denominator_are_finite(self):
        for bidirectional in [False, True]:
            model = MINER(6, 2, 2, bidirectional, 0.0, 0.2)
            packed = torch.tensor([[0., 0., 0., 0., 0., 0.], [0., 0., 1., -1., -1., 1.]])
            pairs = torch.tensor([[0], [1]])
            features = model.get_attack_features(pairs, packed)
            self.assertTrue(torch.isfinite(features).all())
            torch.testing.assert_close(features[0, :8], torch.tensor([0., 0., 0., 0., 1., 1., 0., 0.]))
            loss = model.loss_fn(model(pairs, packed), torch.ones(1))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        # Opposite nonzero vectors also have a zero Bray-Curtis denominator.
        distances = MINER._distances(torch.tensor([[[-1., 2.]]]), torch.tensor([[1., -2.]]))
        self.assertTrue(torch.isfinite(distances).all())

    def test_sgd_updates_match_authors_clipped_momentum_sgd(self):
        actual = torch.nn.Parameter(torch.tensor([10., -20.]))
        expected = torch.nn.Parameter(actual.detach().clone())
        optimizer = SGDWithGradientClipping([actual], lr=0.05, momentum=0.9, weight_decay=0., grad_clip_norm=5.)
        reference = torch.optim.SGD([expected], lr=0.05, momentum=0.9, weight_decay=0.)
        for gradient in [torch.tensor([300., -400.]), torch.tensor([-60., 80.])]:
            actual.grad = gradient.clone()
            expected.grad = gradient.clone()
            torch.nn.utils.clip_grad_norm_([expected], 5.)
            reference.step()
            optimizer.step()
            torch.testing.assert_close(actual, expected)

    def test_config_factories_and_dataset_specific_learning_rates(self):
        for dataset, learning_rate in [('ciao', 0.01), ('flickr', 0.05), ('yelp', 0.05)]:
            for mode in ['miner', 'enminer']:
                with self.subTest(dataset=dataset, mode=mode):
                    config = compose_config(dataset, mode)
                    config.shadow.model.attribute.embed_dim = 2
                    config.shadow.data.num_recommendations = 3
                    shadow = create_shadow_model(42, True, torch.device('cpu'), config.shadow.model.attribute, 3, 5)
                    self.assertIsInstance(shadow, ShadowMINER)
                    model = create_attack_model(42, True, torch.device('cpu'), config.attack.model.attribute, 8)
                    self.assertEqual(model.bidirectional, mode == 'enminer')
                    self.assertEqual(model.attack_feature_dim, 24 if mode == 'enminer' else 12)
                    optimizer = instantiate(config.attack.model.train.optimizer, params=model.parameters())
                    self.assertEqual(optimizer.param_groups[0]['lr'], learning_rate)
                    self.assertIsNone(model.loss_fn.pos_weight)

    def test_experiment_preset_pairs_miner_shadow_with_both_attack_modes(self):
        with initialize_config_module(config_module='sralt.configs', version_base=None):
            config = compose(config_name='config', overrides=[
                'target/data=ciao', 'target/model=design', 'shadow/model=miner',
                'attack/model=miner', 'experiment=miner'
            ], return_hydra_config=True)
        params = config.hydra.sweeper.params
        self.assertEqual(params['shadow/model'], 'miner')
        self.assertEqual(params['attack/model'], 'miner,enminer')
        self.assertEqual(len(params['target/data'].split(',')) * len(params['target/model'].split(',')) *
                         len(params['attack/model'].split(',')) * len(params['seed'].split(',')), 90)

    def test_three_stages_cache_round_trip_and_strict_checkpoint_restore(self):
        config = compose_config()
        for stage in ['target', 'shadow']:
            model_config = config[stage].model
            model_config.attribute.embed_dim = 4
            model_config.attribute.num_layers = 1
            model_config.train.num_epochs = 2
            model_config.train.train_batch_size = 16
            model_config.train.val_batch_size = 12
            model_config.train.val_num_negatives = 4
            model_config.train.metrics[0].k = 2
            model_config.train.metrics[1].k = 2
            config[stage].test.metrics[0].k = 2
            config[stage].test.metrics[1].k = 2
            config[stage].test.evaluation.test_num_negatives = 4
        config.shadow.data.num_recommendations = 4
        config.shadow.data.shadow_social_ratio = 0.5
        config.attack.data.val_ratio = 0.25
        config.attack.tail_ratio = 0.5
        config.attack.model.train.num_epochs = 2
        config.attack.model.train.train_batch_size = 5

        for deterministic in [False, True]:
            with self.subTest(deterministic=deterministic), tempfile.TemporaryDirectory() as directory:
                previous_cwd = os.getcwd()
                try:
                    os.chdir(directory)
                    with patch.dict(os.environ, {'SRALT_CACHE_VERSION': 'miner-test', 'SRALT_CACHE_REFRESH': '0'}):
                        device = torch.device('cpu')
                        with patch('scripts.pipeline.load_target_data', return_value=make_target_data()):
                            target_data, target = target_stage(42, deterministic, device, None, config.target)
                        shadow_data, shadow = shadow_stage(
                            42, deterministic, device, None, config.shadow, target_data, target
                        )
                        with patch.object(ShadowMINER, 'train_model', side_effect=AssertionError('Shadow cache miss')):
                            cached_shadow = load_shadow_model(42, deterministic, device, config.shadow.model, shadow_data)
                        self.assertEqual(repr(shadow), repr(cached_shadow))

                        for mode in ['miner', 'enminer']:
                            attack_config = compose_config(mode=mode).attack
                            attack_config.data.val_ratio = config.attack.data.val_ratio
                            attack_config.tail_ratio = config.attack.tail_ratio
                            attack_config.model.attribute.embed_dim = 4
                            attack_config.model.attribute.num_recommendations = 4
                            attack_config.model.train.num_epochs = 2
                            attack_config.model.train.train_batch_size = 5
                            attack_data, attack = attack_stage(
                                42, deterministic, device, None, attack_config, target_data, shadow_data, shadow
                            )
                            self.assertEqual(attack_data.user_features.shape, (12, 20))
                            self.assertEqual(attack.attack_feature_dim, 32 if mode == 'enminer' else 16)
                            initial = create_attack_model(42, deterministic, device, attack_config.model.attribute, 20)
                            self.assertNotEqual(model_fingerprint(initial), model_fingerprint(attack))
                            with patch.object(ShadowMINER, 'get_user_features', side_effect=AssertionError('Data cache miss')):
                                cached_data = load_attack_data(
                                    42, deterministic, target_data, shadow_data, shadow, attack_config.data
                                )
                            torch.testing.assert_close(cached_data.user_features, attack_data.user_features)
                            with patch.object(MINER, 'train_model', side_effect=AssertionError('Attack cache miss')):
                                cached_attack = load_attack_model(
                                    42, deterministic, device, attack_config.model, cached_data,
                                    attack_config.tail_ratio, None
                                )
                            pairs = attack_data.test_social_edge_index
                            attack.eval()
                            expected = attack.predict(pairs, attack_data.user_features)
                            self.assertTrue(torch.isfinite(expected).all())
                            torch.testing.assert_close(cached_attack.eval().predict(pairs, cached_data.user_features), expected)

                            checkpoint = io.BytesIO()
                            torch.save({'shadow': shadow.state_dict(), 'attack': attack.state_dict()}, checkpoint)
                            checkpoint.seek(0)
                            state = torch.load(checkpoint, weights_only=True)
                            restored_shadow = create_shadow_model(
                                7, deterministic, device, config.shadow.model.attribute, 12, 20
                            )
                            restored_shadow.load_state_dict(state['shadow'], strict=True)
                            restored_features = restored_shadow.get_user_features(shadow_data)
                            torch.testing.assert_close(restored_features, attack_data.user_features)
                            restored_attack = create_attack_model(
                                7, deterministic, device, attack_config.model.attribute, 20
                            )
                            restored_attack.load_state_dict(state['attack'], strict=True)
                            fingerprint = model_fingerprint(restored_attack)
                            torch.testing.assert_close(restored_attack.eval().predict(pairs, restored_features), expected)
                            self.assertEqual(model_fingerprint(restored_attack), fingerprint)
                finally:
                    os.chdir(previous_cwd)


if __name__ == '__main__':
    unittest.main()

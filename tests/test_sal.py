import copy
import os
import tempfile
import unittest
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from torch_geometric.utils import to_undirected

from scripts import comparison_report, pipeline, pipeline_low_memory, reproduce_sal
from sralt.target.data import TargetData
from sralt.target.defense import SocialPairSampler
from sralt.shadow.data import load_shadow_data
from sralt.target.model import create_target_model, load_target_model
from sralt.target.model.design import DESIGN, DefendedDESIGN
from sralt.utils import model_fingerprint, set_seed


def make_data():
    data = TargetData(name='synthetic', directory='unused')
    data.num_users, data.num_items = 12, 20
    users = torch.arange(12)
    data.social_edge_index = to_undirected(torch.cat([
        torch.stack([users, (users + 1) % 12]), torch.stack([users, (users + 2) % 12])
    ], dim=1))
    items = 12 + (2 * users[:, None] + torch.arange(5)) % 20
    data.behavioral_train_edge_index = torch.stack([users.repeat_interleave(3), items[:, :3].flatten()])
    data.behavioral_val_edge_index = torch.stack([users, items[:, 3]])
    data.behavioral_test_edge_index = torch.stack([users, items[:, 4]])
    return data.to(torch.device('cpu'))


def configuration(target='design_sal'):
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        config = compose(config_name='config', overrides=[
            'target/data=ciao', f'target/model={target}', 'shadow/model=smia', 'attack/model=mlp'
        ])
    for stage in ('target', 'shadow', 'attack'):
        config[stage].model.train.num_epochs = 2
        config[stage].model.train.train_batch_size = 7
        config[stage].model.train.val_batch_size = 5
        config[stage].test.evaluation.test_batch_size = 5
    for stage in ('target', 'shadow'):
        config[stage].model.attribute.embed_dim = 4
        config[stage].model.train.val_num_negatives = 4
        config[stage].test.evaluation.test_num_negatives = 4
        for metric in (*config[stage].model.train.metrics, *config[stage].test.metrics):
            metric.k = 2
    config.shadow.data.shadow_social_ratio = .5
    config.shadow.data.num_recommendations = 4
    config.attack.data.val_ratio = .25
    config.attack.tail_ratio = .5
    if target == 'design_sal':
        config.target.model.attribute.defense.surrogate_warmup_epochs = 1
        config.target.model.attribute.defense.surrogate_epochs = 1
        config.target.model.attribute.defense.surrogate_batch_size = 9
    return config


def make_defense():
    return instantiate(configuration().target.model.attribute.defense, embed_dim=4).double()


class SALTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.are_deterministic_algorithms_enabled()
        set_seed(42, deterministic=True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous)

    def test_probability_moments_gradients_and_target_adam_update(self):
        defense = make_defense()
        reference = copy.deepcopy(defense.surrogate)
        embeds = torch.nn.Parameter(torch.randn(12, 4, dtype=torch.float64))
        expected_embeds = torch.nn.Parameter(embeds.detach().clone())
        positive = torch.tensor([[0, 2, 4], [1, 3, 5]])
        negative = torch.tensor([[0, 1, 3], [6, 8, 10]])
        optimizer = torch.optim.Adam([embeds], lr=.01)
        expected_optimizer = torch.optim.Adam([expected_embeds], lr=.01)
        for step in range(2):
            optimizer.zero_grad()
            expected_optimizer.zero_grad()
            torch.manual_seed(43 + step)
            actual = defense.distribution_loss(embeds, positive, negative)
            torch.manual_seed(43 + step)
            p = reference(torch.cat([expected_embeds[positive[0]], expected_embeds[positive[1]]], -1)).sigmoid()
            n = reference(torch.cat([expected_embeds[negative[0]], expected_embeds[negative[1]]], -1)).sigmoid()
            expected = ((p.mean() - n.mean()).square() + (p.var() - n.var()).square()).sqrt()
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
            actual.backward()
            expected.backward()
            torch.testing.assert_close(embeds.grad, expected_embeds.grad, rtol=1e-11, atol=1e-13)
            self.assertTrue(all(p.grad is None for p in defense.surrogate.parameters()))
            optimizer.step()
            expected_optimizer.step()
            torch.testing.assert_close(embeds, expected_embeds)

    def test_surrogate_fitting_detaches_target_and_preserves_dropout_and_optimizer_isolation(self):
        defense = make_defense()
        data = make_data()
        sampler = SocialPairSampler(data.social_edge_index, 12, 1.)
        embeds = torch.nn.Parameter(torch.randn(12, 4, dtype=torch.float64))
        before_embeds = embeds.detach().clone()
        before = model_fingerprint(defense)
        loss = defense.fit_surrogate(embeds, sampler, defense.create_optimizer())
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertNotEqual(before, model_fingerprint(defense))
        torch.testing.assert_close(embeds, before_embeds)
        self.assertIsNone(embeds.grad)
        self.assertTrue(defense.surrogate.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in defense.parameters()))
        before = model_fingerprint(defense)
        positive, negative = sampler.sample()
        defense.distribution_loss(embeds, positive, negative).backward()
        self.assertGreater(embeds.grad.abs().sum(), 0)
        self.assertEqual(before, model_fingerprint(defense))

    def test_balanced_full_graph_non_edges_singleton_batches_and_zero_distance(self):
        data = make_data()
        sampler = SocialPairSampler(data.social_edge_index, 12, .3)  # 7 pairs; tail of 1 is merged.
        positives, negatives = sampler.sample()
        truth = set(map(tuple, sampler.positive_pairs.t().tolist()))
        sampled_positive = set(map(tuple, positives.t().tolist()))
        sampled_negative = set(map(tuple, negatives.t().tolist()))
        self.assertEqual(positives.shape, negatives.shape)
        self.assertEqual(len(sampled_negative), 7)
        self.assertTrue(sampled_positive <= truth)
        self.assertFalse(sampled_negative & truth)
        self.assertTrue((positives[0] < positives[1]).all())
        self.assertTrue((negatives[0] < negatives[1]).all())
        batches = list(sampler.batches(3))
        self.assertEqual([p.size(1) for p, _ in batches], [2, 2, 3])
        repeating = cycle(batches)
        self.assertEqual([next(repeating)[0].size(1) for _ in range(5)], [2, 2, 3, 2, 2])
        defense = make_defense().eval()
        embeds = torch.randn(12, 4, dtype=torch.float64, requires_grad=True)
        loss = defense.distribution_loss(embeds, positives, positives)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(embeds.grad).all())
        self.assertEqual(embeds.grad.abs().sum(), 0.)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            SocialPairSampler(data.social_edge_index[:, :1], 12, 1.)

    def test_zero_weight_preserves_initialization_training_and_rng(self):
        base_config, zero_config, data = configuration('design'), configuration(), make_data()
        zero_config.target.model.attribute.defense.loss_weight = 0.
        models, states = [], []
        for config in (base_config, zero_config):
            model = create_target_model(42, True, torch.device('cpu'), config.target.model.attribute, 12, 20)
            if isinstance(model, DefendedDESIGN):
                self.assertIsNone(model.defense)
            else:
                self.assertFalse(hasattr(model, 'defense'))
            set_seed(42, deterministic=True)
            model.train_model(data=data, writer=None, **config.target.model.train)
            states.append(torch.get_rng_state())
            models.append(model)
        self.assertIs(type(models[0]), DESIGN)
        self.assertIs(type(models[1]), DefendedDESIGN)
        self.assertEqual(models[0].state_dict().keys(), models[1].state_dict().keys())
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(states[0], states[1])

    def test_legacy_baseline_cache_reaches_report_and_shadow_without_constructor_or_retraining(self):
        config, data = configuration('design'), make_data()
        config.device = {'_target_': 'torch.device', 'device': 'cpu'}
        config.deterministic = True
        del config.writer
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                config.target.data.directory = directory
                original = load_target_model(42, True, torch.device('cpu'), config.target.model, data)
                self.assertIs(type(original), DESIGN)
                self.assertFalse(hasattr(original, 'defense'))
                original_repr = repr(original)
                with patch.object(DESIGN, '__init__', side_effect=AssertionError('Reconstructed cached baseline')), \
                        patch.object(DESIGN, 'train_model', side_effect=AssertionError('Retrained cached baseline')):
                    cached = load_target_model(42, True, torch.device('cpu'), config.target.model, data)
                    self.assertEqual(repr(cached), original_repr)
                    self.assertEqual(model_fingerprint(cached), model_fingerprint(original))
                    with patch.object(comparison_report, 'compose_case', return_value=config), \
                            patch.object(comparison_report, 'CACHE_DIR', Path(directory) / 'metrics'), \
                            patch('sralt.target.load_target_data', return_value=data):
                        result = reproduce_sal.run_case('ciao', 42, None, 'legacy-baseline-test', False)
                    self.assertTrue(all(result['results'][stage] for stage in ('target', 'shadow', 'attack')))
                self.assertEqual(len(list(Path('.cache/target/model/DESIGN').glob('*.pkl'))), 1)
                self.assertTrue(list(Path('.cache/shadow/data/synthetic/DESIGN').glob('*.pkl')))
                self.assertNotIn('defense', repr(cached))
            finally:
                os.chdir(previous)

    def test_warmup_graphs_full_epoch_optimizer_groups_and_strict_reload(self):
        config, data = configuration(), make_data()
        model = create_target_model(42, True, torch.device('cpu'), config.target.model.attribute, 12, 20)
        defense = model.defense
        writer = Mock()
        optimized_parameters = []

        def record_optimizer(config, **kwargs):
            if 'params' in kwargs:
                kwargs['params'] = list(kwargs['params'])
                optimized_parameters.extend(kwargs['params'])
            return instantiate(config, **kwargs)

        with patch.object(defense, 'fit_surrogate', wraps=defense.fit_surrogate) as fit, \
                patch.object(model, '_prepare_graphs', wraps=model._prepare_graphs) as graphs, \
                patch.object(defense, 'distribution_loss', wraps=defense.distribution_loss) as distance, \
                patch('sralt.target.model.design.defended.instantiate', side_effect=record_optimizer):
            model.train_model(data=data, writer=writer, **config.target.model.train)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(distance.call_count, 12)  # ceil(36 / 7) x two complete epochs.
        self.assertEqual({id(p) for p in optimized_parameters},
                         {id(p) for branch in (model.social, model.rating, model.twin) for p in branch.parameters()})
        masked = [call for call in graphs.call_args_list if len(call.args) == 3]
        self.assertEqual(len(masked), 12)
        training_graph = to_undirected(data.behavioral_train_edge_index)
        for call in graphs.call_args_list:
            torch.testing.assert_close(call.args[1], training_graph)
        tags = [call.args[0] for call in writer.add_scalar.call_args_list]
        self.assertEqual(tags.count('DefendedDESIGN/SAL/surrogate_bce'), 1)
        self.assertEqual(tags.count('DefendedDESIGN/SAL/defense_loss'), 2)
        restored = create_target_model(43, True, torch.device('cpu'), config.target.model.attribute, 12, 20)
        restored.load_state_dict(model.state_dict(), strict=True)
        before = model_fingerprint(restored)
        torch.testing.assert_close(restored.recommend(data=data, num_recommendations=4, batch_size=5),
                                   model.recommend(data=data, num_recommendations=4, batch_size=5))
        self.assertEqual(before, model_fingerprint(restored))

    def test_both_pipelines_retrain_downstream_and_cache_round_trip(self):
        config, data, device = configuration(), make_data(), torch.device('cpu')
        config.device = {'_target_': 'torch.device', 'device': 'cpu'}
        config.writer = None
        config.deterministic = True
        for deterministic in (False, True):
            config.deterministic = deterministic
            with self.subTest(deterministic=deterministic), tempfile.TemporaryDirectory() as directory:
                previous = os.getcwd()
                try:
                    os.chdir(directory)
                    with patch.object(pipeline, 'load_target_data', return_value=data):
                        target_data, target = pipeline.target_stage(42, deterministic, device, None, config.target)
                    shadow_data, shadow = pipeline.shadow_stage(42, deterministic, device, None, config.shadow, target_data, target)
                    attack_data, attack = pipeline.attack_stage(42, deterministic, device, None, config.attack, target_data, shadow_data, shadow)
                    self.assertEqual(attack_data.user_features.size(1), 16)  # No target/defender features.
                    self.assertFalse(any('surrogate' in name for name in shadow.state_dict()))
                    with patch.object(type(target), 'train_model', side_effect=AssertionError('Retrained target')):
                        cached = load_target_model(42, deterministic, device, config.target.model, data)
                    self.assertEqual(model_fingerprint(cached), model_fingerprint(target))
                    writer = Mock()
                    # Execute the checkpoint-writing entry point as well as individual stages.
                    with patch.object(pipeline, 'load_target_data', return_value=data), \
                            patch.object(pipeline, 'instantiate', side_effect=lambda cfg: writer if cfg is None else instantiate(cfg)), \
                            patch.object(pipeline.HydraConfig, 'get', return_value=SimpleNamespace(
                                runtime=SimpleNamespace(output_dir=directory))):
                        pipeline.main.__wrapped__(config)
                    writer.close.assert_called_once_with()
                    checkpoint = torch.load(Path(directory) / 'target_model.pth', weights_only=True)
                    self.assertTrue(any(name.startswith('defense.surrogate.') for name in checkpoint))
                    target.load_state_dict(checkpoint, strict=True)
                    low_memory_directory = Path(directory) / 'low-memory'
                    low_memory_directory.mkdir()
                    os.chdir(low_memory_directory)
                    writer = Mock()
                    with patch.object(pipeline_low_memory, 'load_target_data', return_value=data), \
                            patch.object(pipeline_low_memory, 'instantiate', side_effect=lambda cfg: writer if cfg is None else instantiate(cfg)):
                        pipeline_low_memory.run_pipeline(config)
                    writer.close.assert_called_once_with()
                    os.chdir(directory)
                    # A changed defense configuration creates a new target cache.
                    changed = copy.deepcopy(config.target.model)
                    changed.attribute.defense.loss_weight = .02
                    updated = load_target_model(42, deterministic, device, changed, data)
                    self.assertNotEqual(model_fingerprint(updated), model_fingerprint(target))
                    load_shadow_data(42, deterministic, data, updated, config.shadow.data)
                    self.assertEqual(len(list(Path('.cache/shadow/data').rglob('*.pkl'))), 2)
                    self.assertEqual(len(list(Path('.cache/target/model/DESIGN_SAL').glob('*.pkl'))), 2)
                finally:
                    os.chdir(previous)

    def test_report_keeps_defenses_and_weights_distinct(self):
        args = comparison_report.parse_args([
            '--datasets', 'ciao', '--targets', 'design', 'design_sal', '--pairs', 'smia:mlp',
            '--seeds', '42', '--sal-weights', '.01', '.02', '--dry-run'
        ])
        with tempfile.TemporaryDirectory() as directory, patch.object(comparison_report, 'CACHE_DIR', Path(directory)):
            payload = comparison_report.collect(args)
        self.assertEqual([r['defense'] for r in payload['rows']], ['None', 'SAL λ=0.01', 'SAL λ=0.02'])
        self.assertEqual({r['target'] for r in payload['rows']}, {'design'})
        configs = [comparison_report.compose_case('ciao', 'design_sal', 'smia', 'mlp', 42, weight)
                   for weight in (.01, .02)]
        for stage in ('target', 'shadow', 'attack'):
            self.assertNotEqual(comparison_report.cache_identity(configs[0], stage, 'test'),
                                comparison_report.cache_identity(configs[1], stage, 'test'))

    def test_experiment_phases_and_paired_relative_loss(self):
        for phase, count in (('minimal', 2), ('scan', 14), ('formal', 40)):
            self.assertEqual(len(list(reproduce_sal.cases(reproduce_sal.parse_args(['--phase', phase])))), count)
        rows = [dict(dataset='ciao', defense=defense, seed=seed,
                     results={'target': {'all': {'Recall@10': recall}}, 'attack': {'all': {'AUROC': auc}}})
                for defense, seed, recall, auc in [('None', 42, .3, .8), ('None', 43, .4, .6),
                                                   ('SAL λ=0.01', 42, .27, .6), ('SAL λ=0.01', 43, .32, .54)]]
        summary = reproduce_sal.paired_summary(rows)
        attack = next(row for row in summary if row['metric'] == 'AUROC')
        self.assertAlmostEqual(attack['relative_loss_mean'], .175)
        self.assertEqual(attack['n'], 2)
        recommendation = next(row for row in summary if row['metric'] == 'Recall@10')
        self.assertAlmostEqual(recommendation['relative_loss_mean'], .15)

    def test_experiment_runner_evaluates_and_resumes_without_training(self):
        config, data = configuration(), make_data()
        config.device = {'_target_': 'torch.device', 'device': 'cpu'}
        config.deterministic = True
        del config.writer
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                config.target.data.directory = directory
                with patch.object(comparison_report, 'compose_case', return_value=config), \
                        patch.object(comparison_report, 'CACHE_DIR', Path(directory) / 'metrics'), \
                        patch('sralt.target.load_target_data', return_value=data):
                    first = reproduce_sal.run_case('ciao', 42, .01, 'synthetic-test', False)
                    self.assertEqual(set(first['status'].values()), {'computed'})
                    self.assertIsNone(first['peak_cuda_allocated_bytes'])
                    self.assertTrue(all(first['results'][stage] for stage in ('target', 'shadow', 'attack')))
                    with patch.object(comparison_report, 'evaluate_missing', side_effect=AssertionError('Recomputed metrics')):
                        cached = reproduce_sal.run_case('ciao', 42, .01, 'synthetic-test', False)
                    self.assertEqual(cached['results'], first['results'])
                    self.assertEqual(set(cached['status'].values()), {'cached'})
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()

"""Training, gradient and deployment checks for the GDMSR/EGCL SAL adapters."""
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from hydra.utils import instantiate
from torch_geometric.utils import to_undirected

from test_sal import configuration, make_data
from scripts import comparison_report, defense_report, pipeline, pipeline_low_memory, reproduce_sal
from sralt.target.defense import SocialPairSampler
from sralt.target.model import create_target_model, load_target_model
from sralt.utils import model_fingerprint, set_seed


def adapter_configuration(target):
    config = configuration(target)
    attribute = config.target.model.attribute
    attribute.num_layers = 1
    if target.startswith('gdmsr'):
        attribute.dim_feedforward = 8
        attribute.sequence_length = 3
        attribute.curriculum_interval = 1
    else:
        attribute.contrastive_batch_size = 5
    if target.endswith('_sal'):
        attribute.defense.surrogate_warmup_epochs = 1
        attribute.defense.surrogate_epochs = 1
        attribute.defense.surrogate_batch_size = 9
    return config


def create(config):
    return create_target_model(42, True, torch.device('cpu'), config.target.model.attribute, 12, 20)


class SALAdapterTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.are_deterministic_algorithms_enabled()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        set_seed(42, deterministic=True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous)
        torch.set_num_threads(self.threads)

    def test_zero_weight_preserves_parameters_training_and_rng(self):
        for target in ('gdmsr', 'egcl'):
            with self.subTest(target=target):
                configs = [adapter_configuration(target), adapter_configuration(target + '_sal')]
                configs[1].target.model.attribute.defense.loss_weight = 0.
                models, states = [], []
                for config in configs:
                    model = create(config)
                    set_seed(42, deterministic=True)
                    model.train_model(data=make_data(), writer=None, **config.target.model.train)
                    models.append(model)
                    states.append(torch.get_rng_state())
                self.assertIsNone(models[1].defense)
                self.assertEqual(models[0].state_dict().keys(), models[1].state_dict().keys())
                for name, value in models[0].state_dict().items():
                    torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)
                torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)

    def test_nonzero_training_readout_gradients_steps_and_strict_restore(self):
        data = make_data()
        behavioral = to_undirected(data.behavioral_train_edge_index)
        for target in ('gdmsr', 'egcl'):
            with self.subTest(target=target):
                config = adapter_configuration(target + '_sal')
                model = create(config)
                distribution = model.defense.distribution_loss
                calls = []

                def checked_loss(embeds, positives, negatives):
                    expected = model.get_embedding(
                        social_edge_index=data.social_edge_index, behavioral_edge_index=behavioral
                    )[:data.num_users]
                    torch.testing.assert_close(embeds, expected)
                    self.assertEqual(embeds.size(1), 8 if target == 'gdmsr' else 4)
                    self.assertTrue(all(not p.requires_grad for p in model.defense.parameters()))
                    loss = distribution(embeds, positives, negatives)
                    params = list((model.denoiser if target == 'gdmsr' else model.gap).parameters())
                    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
                    if target == 'egcl':
                        weight_params = [*model.behavioral_weight.parameters(), *model.social_weight.parameters()]
                        self.assertTrue(all(g is None for g in torch.autograd.grad(
                            loss, weight_params, retain_graph=True, allow_unused=True
                        )))
                    calls.append(positives.size(1))
                    return loss

                original_fit = model.defense.fit_surrogate

                def checked_fit(*args):
                    before = model_fingerprint(model.defense)
                    result = original_fit(*args)
                    self.assertNotEqual(before, model_fingerprint(model.defense))
                    return result

                with patch.object(model.defense, 'distribution_loss', side_effect=checked_loss), \
                        patch.object(model.defense, 'fit_surrogate', side_effect=checked_fit) as fit, \
                        patch.object(SocialPairSampler, 'batches', autospec=True, wraps=None) as batches:
                    # Force few SAL pairs to exercise cycling without truncating GDMSR epochs.
                    batches.side_effect = lambda sampler, steps: iter([sampler.sample()])
                    model.train_model(data=data, writer=None, **config.target.model.train)
                    self.assertEqual(fit.call_count, 1)
                self.assertEqual(len(calls), 12 if target == 'gdmsr' else 2)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.defense.parameters()))
                restored = create(config)
                restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                model.eval()
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model.get_embedding(
                        social_edge_index=data.social_edge_index, behavioral_edge_index=behavioral
                    ), restored.get_embedding(social_edge_index=data.social_edge_index, behavioral_edge_index=behavioral))
                torch.testing.assert_close(model.recommend(data=data, num_recommendations=3, batch_size=5),
                                           restored.recommend(data=data, num_recommendations=3, batch_size=5))
                if target == 'gdmsr':
                    self.assertEqual(restored.social_edge_mask.numel(), data.social_edge_index.size(1))

    def test_both_pipelines_and_cached_black_box_evaluation(self):
        for target in ('gdmsr', 'egcl'):
            for runner in (pipeline, pipeline_low_memory):
                with self.subTest(target=target, pipeline=runner.__name__), tempfile.TemporaryDirectory() as directory:
                    config = adapter_configuration(target + '_sal')
                    config.device = {'_target_': 'torch.device', 'device': 'cpu'}
                    config.deterministic = True
                    config.writer = None
                    config.target.data.directory = directory
                    previous = os.getcwd()
                    try:
                        os.chdir(directory)
                        with patch.object(runner, 'load_target_data', return_value=make_data()):
                            if runner is pipeline_low_memory:
                                writer = Mock()
                                with patch.object(runner, 'instantiate', side_effect=lambda cfg: writer if cfg is None else instantiate(cfg)):
                                    runner.run_pipeline(config)
                                writer.close.assert_called_once_with()
                            else:
                                device = torch.device('cpu')
                                data, target_model = runner.target_stage(42, True, device, None, config.target)
                                shadow_data, shadow = runner.shadow_stage(42, True, device, None, config.shadow, data, target_model)
                                attack_data, _ = runner.attack_stage(42, True, device, None, config.attack, data, shadow_data, shadow)
                                self.assertEqual(attack_data.user_features.size(1), 16)
                        with patch('sralt.target.load_target_data', return_value=make_data()), \
                                patch.object(comparison_report, 'compose_case', return_value=config), \
                                patch.object(comparison_report, 'CACHE_DIR', Path(directory) / 'metrics'), \
                                patch.object(type(create(config)), 'train_model', side_effect=AssertionError('Retraining')):
                            row = reproduce_sal.run_case('ciao', 42, .01, 'adapter-test', False, target)
                            self.assertEqual(row['target'], target)
                            self.assertTrue(all(row['results'][stage] for stage in ('target', 'shadow', 'attack')))
                        changed = copy.deepcopy(config.target.model)
                        changed.attribute.defense.loss_weight = .02
                        load_target_model(42, True, torch.device('cpu'), changed, make_data())
                        self.assertEqual(len(list(Path(f'.cache/target/model/{target.upper()}_SAL').glob('*.pkl'))), 2)
                    finally:
                        os.chdir(previous)

    def test_configs_reports_and_experiment_plans(self):
        for target in ('gdmsr', 'egcl'):
            args = reproduce_sal.parse_args(['--target', target, '--phase', 'formal', '--datasets', 'ciao'])
            self.assertEqual(len(list(reproduce_sal.cases(args))), 10)
            self.assertEqual({weight for _, _, weight in reproduce_sal.cases(args)}, {None, .01})
            args = comparison_report.parse_args(['--datasets', 'ciao', '--targets', target, target + '_sal',
                                                '--pairs', 'smia:mlp', '--seeds', '42', '--sal-weights', '.01', '.02', '--dry-run'])
            with tempfile.TemporaryDirectory() as directory, patch.object(comparison_report, 'CACHE_DIR', Path(directory)):
                rows = comparison_report.collect(args)['rows']
            self.assertEqual([r['defense'] for r in rows], ['None', 'SAL λ=0.01', 'SAL λ=0.02'])
            self.assertEqual({r['target'] for r in rows}, {target})
            plans = list(defense_report.plans(defense_report.parse_args(['--targets', target])))
            self.assertTrue(all(p['adapter'] == target + '_sal' and p['setting_status'] == 'provisional' for p in plans))
        # Same dataset/seed for different targets must never form a pair.
        rows = [dict(dataset='ciao', target='gdmsr', seed=42, defense='None', results={}),
                dict(dataset='ciao', target='egcl', seed=42, defense='SAL λ=0.01', results={})]
        self.assertEqual(reproduce_sal.paired_summary(rows), [])


if __name__ == '__main__':
    unittest.main()

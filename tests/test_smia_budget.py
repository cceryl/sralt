import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from scripts.tune_smia import arguments, cases, settings, attack_config, train_trial
from sralt.shadow.data import ShadowData
from sralt.shadow.model import load_shadow_model
from sralt.shadow.model.smia import SMIA
from sralt.target.model.lightgcn import LightGCN
from sralt.attack.data import AttackData, EvalAttackDataset
from sralt.attack.model import create_attack_model
from sralt.utils import model_fingerprint, set_seed
from test_smia import config, data


def shadow_config(fixed=False):
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        cfg = compose(config_name='config', overrides=[
            'target/data=ciao', 'target/model=lightgcn',
            'shadow/model=' + ('smia_fixed' if fixed else 'smia'), 'attack/model=mlp',
        ])
    cfg.shadow.model.attribute.embed_dim = 4
    cfg.shadow.model.train.num_epochs = 3
    cfg.shadow.model.train.train_batch_size = 12
    cfg.shadow.model.train.val_batch_size = 5
    cfg.shadow.model.train.val_num_negatives = 3
    for metric in cfg.shadow.model.train.metrics:
        metric.k = 2
    if not fixed:
        cfg.shadow.model.train.early_stopping.patience = 1
    return cfg


class SMIABudgetTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.are_deterministic_algorithms_enabled()
        set_seed(42, True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous)

    def make_shadow(self):
        return ShadowData(data(), LightGCN(12, 20, 4, 1, .001), .5, 4, 12, .25)

    def test_final_epoch_is_retained_even_when_validation_degrades(self):
        sd = self.make_shadow()
        models = []
        for fixed in (False, True):
            cfg = shadow_config(fixed)
            snapshots = []
            model = SMIA(12, 20, 4, 2, .01)
            def diagnostic(**kwargs):
                snapshots.append(copy.deepcopy(model.state_dict()))
                return [1. / len(snapshots)] * 2
            # Only shadow recommendation validation is replaced. Training,
            # optimizers and saved state restoration execute normally.
            set_seed(42, True)
            with patch('sralt.shadow.model.smia.compute_recommendation_metrics', side_effect=diagnostic), \
                    patch('sralt.shadow.model.smia.instantiate', wraps=instantiate) as factory:
                model.train_model(data=sd, writer=None, **cfg.shadow.model.train)
            summary = model.training_summary
            self.assertEqual(summary['best_recommendation_epoch'], 1)
            if fixed:
                self.assertEqual(summary['selected_epoch'], 3)
                self.assertEqual(summary['stop_epoch'], 3)
                self.assertEqual(summary['optimizer_steps'], 6)  # 24 training edges / batch12 x 3 epochs.
                expected = snapshots[-1]
                self.assertFalse(any(call.kwargs.get('model') is model for call in factory.call_args_list))
            else:
                self.assertEqual(summary['selected_epoch'], 1)
                self.assertEqual(summary['stop_epoch'], 2)
                expected = snapshots[0]
            for key, value in expected.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            models.append(model)
        self.assertNotEqual(model_fingerprint(models[0]), model_fingerprint(models[1]))

    def test_fixed_budget_cache_reload_and_changed_budget_are_separate(self):
        cfg, sd = shadow_config(True), self.make_shadow()
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                first = load_shadow_model(42, True, torch.device('cpu'), cfg.shadow.model, sd)
                with patch.object(SMIA, 'train_model', side_effect=AssertionError('Retrained fixed shadow')):
                    second = load_shadow_model(42, True, torch.device('cpu'), cfg.shadow.model, sd)
                self.assertEqual(first.training_summary, second.training_summary)
                self.assertEqual(model_fingerprint(first), model_fingerprint(second))
                cfg.shadow.model.train.num_epochs = 2
                third = load_shadow_model(42, True, torch.device('cpu'), cfg.shadow.model, sd)
                self.assertEqual(third.training_summary['selected_epoch'], 2)
                self.assertNotEqual(model_fingerprint(first), model_fingerprint(third))
                self.assertEqual(len(list(Path('.cache/shadow/model/SMIA_FIXED').glob('*.pkl'))), 2)
            finally:
                os.chdir(previous)

    def test_weighted_layernorm_preset_handles_singletons_and_strict_reload(self):
        with initialize_config_module(config_module='sralt.configs', version_base=None):
            cfg = compose(config_name='config', overrides=[
                'target/data=ciao', 'target/model=design_smia', 'shadow/model=smia_fixed',
                'attack/model=mlp_stable',
            ])
        self.assertEqual(cfg.attack.model.attribute.pos_weight, 2.)
        self.assertEqual(cfg.attack.model.train.optimizer.lr, .01)
        self.assertEqual(cfg.shadow.model.train.checkpoint_selection, 'last')
        model = create_attack_model(42, True, torch.device('cpu'), cfg.attack.model.attribute, 8)
        self.assertEqual(sum(isinstance(m, torch.nn.LayerNorm) for m in model.modules()), 2)
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules()))
        features = torch.randn(4, 8)
        edge = torch.tensor([[0], [1]])
        logits = model(edge, features)
        loss = model.loss_fn(logits, torch.ones(1))
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(1), pos_weight=torch.tensor(2.))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        other = create_attack_model(43, True, torch.device('cpu'), cfg.attack.model.attribute, 8)
        other.load_state_dict(model.state_dict(), strict=True)
        model.eval(); other.eval()
        torch.testing.assert_close(model.predict(edge, features), other.predict(edge.flip(0), features))

    def test_budget_weight_norm_grid_preserves_original_control(self):
        args = arguments(['--sweep', 'shadow', '--datasets', 'flickr', '--targets', 'design_smia', 'design_smia_sal',
                          '--seeds', '42', '--shadow-lrs', '.01', '--shadow-decays', '.01',
                          '--shadow-checkpoint', 'last', '--shadow-epoch-grid', '5', '10', '20',
                          '--norms', 'none', 'layer', '--pos-weights', '1', '2', '--lrs', '.05', '.01'])
        inputs, grid = list(cases(args)), list(settings(args))
        self.assertEqual(len(inputs), 6)
        self.assertEqual(len(grid), 8)
        self.assertEqual(len({name for _, _, name in inputs}), 6)
        for cfg, identity, _ in inputs:
            self.assertEqual(cfg.shadow.model.train.checkpoint_selection, 'last')
            self.assertIsNone(cfg.shadow.model.train.early_stopping)
            self.assertIn(cfg.shadow.model.train.num_epochs, (5, 10, 20))
            self.assertEqual(cfg.shadow.data.revision, 'attack-train-only-v1')
        original = config()
        old_args = arguments(['--sweep', 'shadow', '--lrs', '.05', '--epochs', '50', '--patience', '5'])
        self.assertEqual(attack_config(original, next(settings(old_args)), 'original'), original.attack.model)
        for setting in grid:
            attack = attack_config(original, setting, 'original')
            self.assertEqual(attack.attribute.pos_weight, setting.get('pos_weight', 1.))
        with self.assertRaises(SystemExit):
            arguments(['--shadow-epoch-grid', '10', '--shadow-epochs', '20'])

    def test_validation_trials_keep_candidates_identical_across_weight_and_norm(self):
        cfg, sd = shadow_config(True), self.make_shadow()
        cfg.attack.model.train.val_batch_size = 5
        cfg.attack.tail_ratio = .5
        ad = AttackData(data(), sd, SMIA(12, 20, 4, 2, .01), .25)
        original_eval = EvalAttackDataset.__init__
        def validation_only(instance, data, split='val'):
            self.assertEqual(split, 'val')
            return original_eval(instance, data, split)
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                with patch.object(EvalAttackDataset, '__init__', validation_only):
                    for index, (norm, weight) in enumerate((('none', 1.), ('layer', 2.))):
                        folder = Path(directory) / str(index)
                        folder.mkdir()
                        setting = dict(lr=.01, batch=5, epochs=2, patience=2, norm=norm, pos_weight=weight)
                        rows.append(train_trial(ad, cfg, setting, folder, torch.device('cpu'), 'original'))
                self.assertEqual(rows[0]['validation_pairs'], rows[1]['validation_pairs'])
                self.assertNotEqual(rows[0]['model_fingerprint'], rows[1]['model_fingerprint'])
                self.assertEqual(len(list(Path('.cache/attack/model/MLP').glob('*.pkl'))), 2)
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()

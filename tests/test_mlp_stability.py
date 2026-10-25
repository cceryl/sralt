import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_module

from scripts.tune_smia import arguments, cases, settings, attack_config, train_trial
from sralt.attack.data import AttackData, TrainAttackDataset, EvalAttackDataset, BalancedAttackBatchSampler
from sralt.attack.data.dataset import sample_negative_pairs
from sralt.attack.model import create_attack_model
from sralt.shadow.data import ShadowData
from sralt.shadow.model.smia import SMIA
from sralt.target.model.lightgcn import LightGCN
from sralt.utils import DataLoader, model_fingerprint, set_seed
from test_smia import config, data


def attack_data():
    target_data = data()
    shadow = ShadowData(target_data, LightGCN(12, 20, 4, 1, .001), .5, 4, 12, .25)
    return AttackData(target_data, shadow, SMIA(12, 20, 4, 2, .01), .25)


class MLPStabilityTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.are_deterministic_algorithms_enabled()
        set_seed(42, True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous)

    def test_balanced_sampler_covers_all_rows_keeps_tail_and_resamples_each_epoch(self):
        ad = attack_data()
        dataset = TrainAttackDataset(ad, 1.)
        sampler = BalancedAttackBatchSampler(dataset, 8)
        batches = list(sampler)
        self.assertEqual(len(sampler), 2)  # Tiny 1+1 tail merges into the previous batch.
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(18)))
        self.assertEqual([len(batch) for batch in batches], [8, 10])
        for batch in batches:
            self.assertEqual(dataset.labels[batch].sum().item() * 2, len(batch))
        with patch('sralt.attack.data.dataset.sample_negative_pairs', wraps=sample_negative_pairs) as sample:
            loader = DataLoader(dataset, batch_sampler=sampler)
            for _ in range(2):
                count = 0
                for users, friends, labels in loader:
                    self.assertEqual(labels.sum().item() * 2, labels.numel())
                    count += labels.numel()
                self.assertEqual(count, len(dataset))
            self.assertEqual(sample.call_count, 2)
            for call in sample.call_args_list:
                self.assertIs(call.args[0], ad.train_social_edge_index)
        with self.assertRaises(ValueError):
            BalancedAttackBatchSampler(dataset, 7)

    def test_diagnostics_are_observational_and_capture_real_class_balance(self):
        cfg, ad = config(), attack_data()
        cfg.attack.model.train.num_epochs = 3
        cfg.attack.model.train.train_batch_size = 8
        cfg.attack.model.train.val_batch_size = 4
        cfg.attack.tail_ratio = .5
        models = []
        for diagnostics in (False, True):
            model = create_attack_model(42, True, torch.device('cpu'), cfg.attack.model.attribute, ad.user_feature_dim)
            set_seed(42, True)
            model.train_model(data=ad, tail_ratio=.5, writer=None, **cfg.attack.model.train,
                              logit_diagnostics=diagnostics)
            models.append(model)
        self.assertEqual(model_fingerprint(models[0]), model_fingerprint(models[1]))
        self.assertEqual(models[0].training_summary, models[1].training_summary)
        self.assertFalse(hasattr(models[0], 'training_diagnostics'))
        self.assertEqual(len(models[1].training_diagnostics), 3)
        first = models[1].training_diagnostics[0]
        self.assertIsNone(first['validation_prediction_flip_rate'])
        self.assertIn('logit_positive_mean', first)
        self.assertTrue(all(0 <= row['predicted_positive_rate'] <= 1 for row in models[1].training_diagnostics))

    def test_balanced_validation_trials_keep_candidates_and_restore_selection(self):
        cfg, ad = config(), attack_data()
        cfg.attack.tail_ratio = .5
        original = EvalAttackDataset.__init__
        def validation_only(instance, dataset, split='val'):
            self.assertEqual(split, 'val')
            return original(instance, dataset, split)
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                with patch.object(EvalAttackDataset, '__init__', validation_only):
                    for index, balanced in enumerate((False, True)):
                        folder = Path(directory) / str(index)
                        folder.mkdir()
                        setting = dict(lr=.01, batch=8, epochs=2, patience=2,
                                       balanced_batches=balanced, logit_diagnostics=True)
                        rows.append(train_trial(ad, cfg, setting, folder, torch.device('cpu'), 'original'))
                self.assertEqual(rows[0]['validation_pairs'], rows[1]['validation_pairs'])
                for diagnostic in rows[1]['training_diagnostics']:
                    for key in ('batch_positive_rate_min', 'batch_positive_rate_max', 'last_batch_positive_rate'):
                        self.assertEqual(diagnostic[key], .5)
                self.assertEqual(len(list(Path('.cache/attack/model/MLP').glob('*.pkl'))), 2)
            finally:
                os.chdir(previous)

    def test_defaults_and_presets_keep_behavior_options_explicit(self):
        cfg = config()
        self.assertEqual(cfg.shadow.model.train.num_epochs, 10)
        self.assertNotIn('checkpoint_selection', cfg.shadow.model.train)
        self.assertNotIn('balanced_batches', cfg.attack.model.train)
        with initialize_config_module(config_module='sralt.configs', version_base=None):
            preset = compose(config_name='config', overrides=[
                'target/data=flickr', 'target/model=design_smia', 'shadow/model=smia', 'attack/model=mlp_balanced',
            ])
        self.assertTrue(preset.attack.model.train.balanced_batches)
        self.assertTrue(preset.attack.model.train.logit_diagnostics)
        self.assertEqual(preset.attack.model.attribute.pos_weight, 2.)
        args = arguments(['--sweep', 'shadow', '--datasets', 'flickr', '--targets', 'design_smia',
                          '--seeds', '42', '--shadow-lrs', '.01', '--shadow-decays', '.01',
                          '--balanced-batches', '--logit-diagnostics'])
        case, _, _ = next(cases(args))
        setting = next(settings(args))
        changed = attack_config(case, setting, 'original')
        self.assertTrue(changed.train.balanced_batches)
        self.assertTrue(changed.train.logit_diagnostics)
        self.assertEqual(case.shadow.model.train.num_epochs, 10)


if __name__ == '__main__':
    unittest.main()

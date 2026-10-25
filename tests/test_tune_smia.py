import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.tune_smia import arguments, cases, settings, reports, train_trial, attack_config, freeze_input, main
from test_smia import config, data
from sralt.target.model.lightgcn import LightGCN
from sralt.shadow.data import ShadowData
from sralt.shadow.model.smia import SMIA
from sralt.attack.data import AttackData, EvalAttackDataset
from sralt.attack.model import load_attack_model
from sralt.attack.model.mlp import MLP
from sralt.utils import set_seed


class TuningTests(unittest.TestCase):
    def test_plan_is_finite_and_has_original_baseline(self):
        args = arguments([])
        self.assertEqual(len(list(cases(args))) * len(list(settings(args))), 128)
        self.assertEqual(list(settings(args))[0], dict(label='baseline', lr=.05, batch=512, epochs=50, patience=5))
        cfg = config()
        self.assertEqual(attack_config(cfg, list(settings(args))[0], 'original'), cfg.attack.model)

    def test_training_uses_only_fixed_public_validation_and_restores_checkpoint(self):
        cfg = config()
        cfg.shadow.data.shadow_social_ratio = .5
        cfg.shadow.data.num_recommendations = 4
        cfg.attack.data.val_ratio = .25
        cfg.attack.tail_ratio = .5
        set_seed(42, deterministic=False)
        td = data()
        sd = ShadowData(td, LightGCN(12, 20, 4, 1, .001), .5, 4, 12, .25)
        attack = AttackData(td, sd, SMIA(12, 20, 4, 2, .01), .25)
        original_init = EvalAttackDataset.__init__

        def val_only(instance, attack_data, split='val'):
            self.assertEqual(split, 'val')
            original_init(instance, attack_data, split)

        with tempfile.TemporaryDirectory() as temporary, patch.object(EvalAttackDataset, '__init__', val_only):
            previous = os.getcwd()
            try:
                os.chdir(temporary)
                fingerprints = []
                for batch, norm, selection in [(4, 'none', 'original'), (8, 'layer', 'auc-f1')]:
                    folder = Path(temporary) / str(batch)
                    folder.mkdir()
                    setting = dict(label='smoke', lr=.003, batch=batch, epochs=2, patience=2)
                    if norm != 'none':
                        setting['norm'] = norm
                    row = train_trial(attack, cfg, setting, folder, torch.device('cpu'), selection)
                    self.assertFalse((folder / 'model.pt').exists())
                    self.assertIn(row['best_epoch'], [1, 2])
                    self.assertTrue(row['training_summary_available'])
                    self.assertEqual(set(row['validation']['all']), {'AUROC', 'F1', 'TPR01'})
                    fingerprints.append(row['validation_pairs'])
                    # Both the normal pipeline and a different sweep directory hit
                    # the same single weight cache, with no second training pass.
                    other = Path(temporary) / f'repeat-{batch}'
                    other.mkdir()
                    with patch.object(MLP, 'train_model', side_effect=AssertionError('Retrained attack')):
                        cached = load_attack_model(42, cfg.deterministic, torch.device('cpu'),
                                                   attack_config(cfg, setting, selection), attack, .5, None)
                        repeated = train_trial(attack, cfg, setting, other, torch.device('cpu'), selection)
                    self.assertEqual(row['model_fingerprint'], repeated['model_fingerprint'])
                    self.assertEqual(row['validation'], repeated['validation'])
                    self.assertEqual(row['best_epoch'], repeated['best_epoch'])
                    self.assertEqual(getattr(cached, 'norm', 'none'), norm)
                self.assertEqual(*fingerprints)
                self.assertEqual(len(list(Path('.cache/attack/model/MLP').glob('*.pkl'))), 2)
                self.assertEqual(list(Path(temporary).rglob('*.pt')), [])
            finally:
                os.chdir(previous)

    def test_shadow_plan_and_norm_grid_keep_current_control(self):
        args = arguments(['--sweep', 'shadow', '--datasets', 'ciao', '--targets', 'design',
                          '--seeds', '42', '--norms', 'none', 'layer'])
        inputs = list(cases(args))
        self.assertEqual(len(inputs), 6)
        self.assertEqual(len({name for _, _, name in inputs}), 6)
        self.assertEqual(len(list(settings(args))), 2)
        for cfg, identity, _ in inputs:
            self.assertEqual(cfg.shadow.model.attribute.embed_dim, 64)
            self.assertEqual(cfg.shadow.model.attribute.num_layers, 2)
            self.assertEqual(cfg.shadow.model.train.train_batch_size, 8192)
            self.assertEqual(cfg.shadow.model.train.num_epochs, 10)
            self.assertEqual(identity['shadow_setting']['attribute.decay'], cfg.shadow.model.attribute.decay)
        control = next(cfg for cfg, _, _ in inputs if cfg.shadow.model.train.optimizer.lr == .01
                       and cfg.shadow.model.attribute.decay == .01)
        self.assertEqual(control.shadow.model, config().shadow.model)
        for setting in settings(args):
            self.assertEqual(setting['lr'], .05)
            self.assertEqual(setting['batch'], 512)
            self.assertEqual(setting['epochs'], 150)

    def test_shadow_preparation_uses_shared_cache_and_never_trains_target(self):
        from comparison_report import load_report_model
        from omegaconf import OmegaConf

        cfg, td = config(), data()
        cfg.shadow.data.shadow_social_ratio = .5
        cfg.shadow.data.num_recommendations = 4
        cfg.attack.data.val_ratio = .25
        cfg.shadow.model.attribute.embed_dim = 4
        cfg.shadow.model.train.num_epochs = 1
        cfg.shadow.model.train.train_batch_size = 5
        cfg.shadow.model.train.val_num_negatives = 4
        for metric in cfg.shadow.model.train.metrics:
            metric.k = 2
        target = LightGCN(12, 20, 4, 1, .001)

        def reader(stage, loader, allow_training, **kwargs):
            if stage == 'target':
                self.assertFalse(allow_training)
                return target
            return load_report_model(stage, loader, allow_training, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            previous = os.getcwd()
            try:
                os.chdir(temporary)
                folder = Path(temporary)
                with patch('sralt.target.load_target_data', return_value=td), \
                     patch('sralt.target.load_target_model', side_effect=AssertionError('Trained target')), \
                     patch('comparison_report.load_report_model', side_effect=reader):
                    # Frozen-shadow mode must reject a cache miss.
                    with self.assertRaises(RuntimeError):
                        freeze_input(cfg, torch.device('cpu'), folder)
                    first = freeze_input(cfg, torch.device('cpu'), folder, allow_shadow_training=True)
                    with patch.object(SMIA, 'train_model', side_effect=AssertionError('Retrained shadow')):
                        cached = freeze_input(cfg, torch.device('cpu'), folder, allow_shadow_training=True)
                    torch.testing.assert_close(first.user_features, cached.user_features)
                    OmegaConf.update(cfg.shadow.model, 'train.optimizer.lr', .003)
                    second = freeze_input(cfg, torch.device('cpu'), folder, allow_shadow_training=True)
                    self.assertNotEqual(first.shadow_model_repr, second.shadow_model_repr)
                    torch.testing.assert_close(first.val_social_edge_index, second.val_social_edge_index)
                self.assertEqual(len(list(Path('.cache/shadow/model/SMIA').glob('*.pkl'))), 2)
                self.assertFalse((folder / 'input.pkl').exists())
                self.assertEqual(list(folder.rglob('*.pt')), [])
            finally:
                os.chdir(previous)

    def test_layer_norm_handles_single_pair_and_preserves_original_state_keys(self):
        original = MLP(4, .5, .2, 1.)
        self.assertEqual(set(original.state_dict()), {'loss_fn.pos_weight', 'net.0.weight', 'net.0.bias',
                                                     'net.3.weight', 'net.3.bias', 'net.6.weight', 'net.6.bias'})
        model = MLP(4, .5, .2, 1., norm='layer')
        self.assertEqual(sum(isinstance(layer, torch.nn.LayerNorm) for layer in model.net), 2)
        edges, features = torch.tensor([[0], [1]]), torch.randn(2, 4)
        loss = model.loss_fn(model(edges, features), torch.ones(1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        torch.testing.assert_close(model.predict(edges, features), model.predict(edges.flip(0), features))

    def test_prepare_then_train_and_resume_without_private_checkpoints(self):
        cfg = config()
        cfg.shadow.data.shadow_social_ratio = .5
        cfg.shadow.data.num_recommendations = 4
        cfg.attack.data.val_ratio = .25
        cfg.attack.tail_ratio = .5
        td = data()
        sd = ShadowData(td, LightGCN(12, 20, 4, 1, .001), .5, 4, 12, .25)
        attack = AttackData(td, sd, SMIA(12, 20, 4, 2, .01), .25)
        cli = ['--datasets', 'ciao', '--targets', 'design', '--seeds', '42', '--no-baseline',
               '--lrs', '.003', '--batches', '4', '--epochs', '2', '--patience', '2',
               '--norms', 'none', 'layer', '--selection', 'auc-f1']
        identity = next(cases(arguments(cli)))[1]
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.getcwd()
            try:
                root = Path(temporary)
                output = root / 'sweep'
                cli.extend(['--output', str(output), '--device', 'cpu'])
                with patch('scripts.tune_smia.ROOT', root), \
                     patch('scripts.tune_smia.cases', return_value=[(cfg, identity, 'synthetic')]), \
                     patch('scripts.tune_smia.freeze_input', return_value=attack):
                    main([*cli, '--prepare-only'])
                    self.assertFalse((root / '.cache/attack/model/MLP').exists())
                    main(cli)
                    first = json.loads((output / 'results.json').read_text())
                    self.assertEqual(len(first), 2)
                    with patch('scripts.tune_smia.freeze_input', side_effect=AssertionError('Reloaded inputs')), \
                         patch.object(MLP, 'train_model', side_effect=AssertionError('Retrained attack')):
                        main(cli)
                    self.assertEqual(first, json.loads((output / 'results.json').read_text()))
                self.assertEqual(len(list((root / '.cache/attack/model/MLP').glob('*.pkl'))), 2)
                self.assertEqual(list(output.rglob('*.pkl')), [])
                self.assertEqual(list(output.rglob('*.pt')), [])
                self.assertEqual(json.loads((output / 'status.json').read_text())['completed'], 2)
            finally:
                os.chdir(previous)

    def test_summary_does_not_pool_targets_and_marks_round_finished(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [dict(dataset='ciao', target=target, seed=seed,
                         setting=dict(label='baseline', lr=.05, batch=512, epochs=50, patience=5),
                         selection_score=1.5, validation={'all': dict(AUROC=.8, F1=.6, TPR01=.1)})
                    for target in ['design', 'gdmsr'] for seed in [42, 43]]
            reports(output, rows, len(rows))
            self.assertEqual(len(json.loads((output / 'summary.json').read_text())), 2)
            self.assertEqual(json.loads((output / 'status.json').read_text())['state'], 'complete_waiting_for_user')

    def test_summary_separates_shadow_configs_norms_and_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = []
            for lr, norm, selection in [(lr, norm, selection) for lr in [.001, .01]
                                         for norm in ['none', 'layer'] for selection in ['original', 'auc-f1']]:
                for seed in [42, 43]:
                    rows.append(dict(dataset='ciao', target='design', seed=seed, selection=selection,
                                     shadow_setting={'train.optimizer.lr': lr},
                                     setting={'label': 'grid', **({'norm': norm} if norm != 'none' else {})},
                                     selection_score=1.5, validation={'all': dict(AUROC=.8, F1=.6, TPR01=.1)}))
            reports(Path(temporary), rows, len(rows))
            summary = json.loads((Path(temporary) / 'summary.json').read_text())
            self.assertEqual(len(summary), 8)
            self.assertTrue(all(row['n'] == 2 for row in summary))


if __name__ == '__main__':
    unittest.main()

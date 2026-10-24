import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.tune_smia import arguments, cases, settings, reports, train_trial
from test_smia import config, data
from sralt.target.model.lightgcn import LightGCN
from sralt.shadow.data import ShadowData
from sralt.shadow.model.smia import SMIA
from sralt.attack.data import AttackData, EvalAttackDataset
from sralt.utils import set_seed


class TuningTests(unittest.TestCase):
    def test_plan_is_finite_and_has_original_baseline(self):
        args = arguments([])
        self.assertEqual(len(list(cases(args))) * len(list(settings(args))), 128)
        self.assertEqual(list(settings(args))[0], dict(label='baseline', lr=.05, batch=512, epochs=50, patience=5))

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
            fingerprints = []
            for batch in (4, 8):
                folder = Path(temporary) / str(batch)
                folder.mkdir()
                row = train_trial(attack, cfg, dict(label='smoke', lr=.003, batch=batch, epochs=2, patience=2),
                                  folder, torch.device('cpu'), 'original')
                self.assertTrue((folder / 'model.pt').exists())
                self.assertIn(row['best_epoch'], [1, 2])
                self.assertEqual(set(row['validation']['all']), {'AUROC', 'F1', 'TPR01'})
                fingerprints.append(row['validation_pairs'])
            self.assertEqual(*fingerprints)

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


if __name__ == '__main__':
    unittest.main()

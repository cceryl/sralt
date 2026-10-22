import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import tune_egcl
from sralt.target.data import TargetData


class TuneEGCLTests(unittest.TestCase):
    def test_grid_has_no_attack_config_and_preserves_base(self):
        args = tune_egcl.parse_args(['--seeds', '42', '43', '--grid',
                                    'train.optimizer.lr=0.01,0.003', 'attribute.num_layers=1,2'])
        cases = list(tune_egcl.make_cases(args))
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({name for _, _, name in cases}), 8)
        for config, identity, _ in cases:
            self.assertNotIn('shadow', config)
            self.assertNotIn('attack', config)
            self.assertEqual(config.target.model.train.early_stopping.patience, 50)
            self.assertEqual(identity['target']['model']['name'], 'EGCL')

    def test_embedding_diagnostics(self):
        identical = tune_egcl.embedding_stats(torch.ones(4, 3))
        orthogonal = tune_egcl.embedding_stats(torch.eye(4))
        self.assertAlmostEqual(identical['user_mean_cosine'], 1, places=6)
        self.assertAlmostEqual(identical['user_pc1_energy'], 1, places=6)
        self.assertAlmostEqual(orthogonal['user_mean_cosine'], 0, places=6)
        self.assertAlmostEqual(orthogonal['user_pc1_energy'], .25, places=6)

    def test_train_report_and_cache_resume(self):
        data = TargetData(name='synthetic', directory='unused')
        data.num_users, data.num_items = 3, 8
        data.social_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data.behavioral_train_edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [3, 4, 4, 5, 5, 6]])
        data.behavioral_val_edge_index = torch.tensor([[0, 1, 2], [5, 6, 7]])
        data.behavioral_test_edge_index = torch.tensor([[0, 1, 2], [6, 7, 8]])
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            os.chdir(temporary)
            try:
                args = ['--cpu', '--seeds', '42', '--grid', 'train.optimizer.lr=0.003',
                        '--num-recommendations', '2', '--set', 'attribute.embed_dim=4',
                        'train.num_epochs=2', 'train.val_num_negatives=3',
                        'train.metrics.0.k=2', 'train.metrics.1.k=2']
                with patch('scripts.tune_egcl.os.chdir'), patch('sralt.target.load_target_data', return_value=data):
                    tune_egcl.main(args)
                    first = json.loads(Path('runs/egcl_tuning/results.json').read_text())
                    with patch('sralt.target.model.egcl.EGCL.train_model', side_effect=AssertionError('cache miss')):
                        tune_egcl.main(args)
                    second = json.loads(Path('runs/egcl_tuning/results.json').read_text())
                self.assertEqual(first, second)
                self.assertEqual(first[0]['validation_score'], sum(first[0]['validation'].values()))
                self.assertNotIn('test', first[0])
                self.assertGreater(first[0]['diagnostics']['recommended_unique_items'], 0)
                self.assertTrue(Path('runs/egcl_tuning/summary.csv').exists())
                self.assertEqual(len(list(Path('.cache/target/model/EGCL').glob('*.pkl'))), 1)
                self.assertFalse(Path('.cache/shadow').exists())
                self.assertFalse(Path('.cache/attack').exists())
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()

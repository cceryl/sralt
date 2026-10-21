import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sralt.target.data.ciao import Ciao
from sralt.target.data.epinions import Epinions
from sralt.utils import set_seed


class EpinionsTests(unittest.TestCase):
    def test_mat_filtering_reindexing_and_random_leave_one_out(self):
        # Sparse raw IDs, duplicate ratings, a low rating, and a social cascade:
        # user 40 has only four interactions; removing it isolates user 50.
        ratings = [[u, i, 1, 2] for u in [10, 20, 30, 50] for i in [100, 200, 300, 400, 500]]
        ratings += [[40, i, 1, 5] for i in [100, 200, 300, 400]]
        ratings += [[10, 100, 1, 5], [10, 600, 1, 1]]
        social = np.array([[10, 20], [20, 30], [30, 10], [30, 40], [40, 50], [10, 20]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            scipy.io.savemat(path / 'trustnetwork.mat', {'trustnetwork': social})
            scipy.io.savemat(path / 'rating.mat', {'rating': np.array(ratings)})
            kwargs = dict(name='Epinions', directory=directory, min_rating=2,
                          min_social_connections=1, min_user_interactions=5, min_item_interactions=1)
            set_seed(42, True)
            data = Epinions(**kwargs)
            self.assertEqual((data.num_users, data.num_items, data.num_social_edges), (3, 5, 6))
            self.assertEqual(data.num_behavioral_edges, 15)
            expected = torch.cartesian_prod(torch.arange(3), torch.arange(3, 8)).t()
            actual = data.behavioral_edge_index
            torch.testing.assert_close(torch.sort(actual[0] * 8 + actual[1]).values,
                                       expected[0] * 8 + expected[1])
            splits = [data.behavioral_train_edge_index, data.behavioral_val_edge_index,
                      data.behavioral_test_edge_index]
            for split, count in zip(splits, [3, 1, 1]):
                self.assertEqual(split.dtype, torch.long)
                self.assertEqual(torch.bincount(split[0]).tolist(), [count] * 3)
            set_seed(42, True)
            repeat = Epinions(**kwargs)
            for field in ['social_edge_index', 'behavioral_train_edge_index',
                          'behavioral_val_edge_index', 'behavioral_test_edge_index']:
                torch.testing.assert_close(getattr(data, field), getattr(repeat, field))
            set_seed(43, True)
            other = Epinions(**kwargs)
            self.assertFalse(torch.equal(data.behavioral_train_edge_index, other.behavioral_train_edge_index))
            # Ciao's extra helpfulness column must not change the shared pipeline.
            scipy.io.savemat(path / 'rating.mat', {'rating': np.column_stack([ratings, np.ones(len(ratings))])})
            set_seed(42, True)
            ciao = Ciao(**(kwargs | {'name': 'Ciao'}))
            for field in ['social_edge_index', 'behavioral_train_edge_index',
                          'behavioral_val_edge_index', 'behavioral_test_edge_index']:
                torch.testing.assert_close(getattr(data, field), getattr(ciao, field))
            self.assertTrue(repr(data).startswith('Epinions('))
            self.assertTrue(repr(ciao).startswith('Ciao('))

    def test_all_model_configurations_resolve_and_sweeps_include_epinions(self):
        with initialize_config_module(config_module='sralt.configs', version_base=None):
            for target in ['design', 'gdmsr', 'egcl', 'lightgcn']:
                for shadow, attack in [('smia', 'mlp'), ('csa', 'mlp'), ('dsa', 'mlp'),
                                       ('miner', 'miner'), ('miner', 'enminer')]:
                    with self.subTest(target=target, shadow=shadow, attack=attack):
                        config = compose(config_name='config', overrides=[
                            'target/data=epinions', f'target/model={target}',
                            f'shadow/model={shadow}', f'attack/model={attack}'
                        ])
                        for stage in ['target', 'shadow', 'attack']:
                            OmegaConf.to_container(config[stage], resolve=True)
                        self.assertEqual(config.target.data.name, 'Epinions')
                        if shadow == 'miner':
                            parameter = torch.nn.Parameter(torch.zeros(1))
                            optimizer = instantiate(config.attack.model.train.optimizer, params=[parameter])
                            self.assertEqual(optimizer.param_groups[0]['lr'], 0.05)
            for experiment in ['comparison', 'miner']:
                config = compose(config_name='config', overrides=[
                    f'experiment={experiment}', 'target/data=epinions', 'target/model=design',
                    'shadow/model=miner', 'attack/model=miner'
                ], return_hydra_config=True)
                self.assertIn('epinions', config.hydra.sweeper.params['target/data'].split(','))


if __name__ == '__main__':
    unittest.main()

import unittest

import torch

from sralt.utils import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_restore_best_epoch_after_stop_or_epoch_limit(self):
        for patience, stopped in [(2, True), (10, False)]:
            with self.subTest(patience=patience):
                model = torch.nn.Linear(1, 1, bias=False)
                stopping = EarlyStopping(model, patience=patience, greater_is_better=True)
                for epoch, metric in enumerate([.5, .8, .7, .6], start=1):
                    with torch.no_grad():
                        model.weight.fill_(epoch)
                    result = stopping.step(metric)
                self.assertEqual(result, stopped)
                self.assertEqual(stopping.epoch, 4)
                self.assertEqual(stopping.best_epoch, 2)
                with self.assertLogs(level='INFO') as logs:
                    stopping.load_best_state()
                self.assertIn('epoch=2', logs.output[0])
                self.assertEqual(model.weight.item(), 2)

    def test_reset_epoch_and_epsilon_with_minimization(self):
        model = torch.nn.Linear(1, 1)
        stopping = EarlyStopping(model, patience=2, epsilon=.01)
        self.assertFalse(stopping.step(1.0))
        self.assertFalse(stopping.step(.995))
        self.assertEqual(stopping.best_epoch, 1)
        self.assertFalse(stopping.step(.9))
        self.assertEqual(stopping.best_epoch, 3)
        self.assertFalse(stopping.step(.91))
        with self.assertLogs(level='INFO') as logs:
            self.assertTrue(stopping.step(.92))
        self.assertIn('best epoch=3, current epoch=5', logs.output[0])
        stopping.reset()
        self.assertEqual((stopping.epoch, stopping.best_epoch, stopping.counter), (0, 0, 0))
        self.assertFalse(stopping.step(2.0))
        self.assertEqual(stopping.best_epoch, 1)


if __name__ == '__main__':
    unittest.main()

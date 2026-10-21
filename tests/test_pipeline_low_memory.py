import unittest
import weakref
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf

from scripts import pipeline_low_memory as pipeline


class Resource:
    def to(self, device):
        self.device = device
        return self


class LowMemoryPipelineTests(unittest.TestCase):
    def setUp(self):
        recommendation_test = dict(metrics=[dict(_target_='example.LinkPredRecall', k=10)])
        self.config = OmegaConf.create(dict(
            seed=42, deterministic=False, filter_warnings=[],
            device=dict(kind='device'), writer=dict(kind='writer'),
            target=dict(data=dict(name='synthetic'), model=dict(name='target'), test=recommendation_test),
            shadow=dict(data={}, model=dict(name='shadow'), test=recommendation_test),
            attack=dict(data={}, model=dict(name='attack'), tail_ratio=0.8,
                        test=dict(metrics=[dict(_target_='example.BinaryAUROC')]))
        ))
        self.writer = Mock()

    def instantiate(self, config):
        if isinstance(config, list) or not hasattr(config, 'kind'):
            return None
        return torch.device('cpu') if config.kind == 'device' else self.writer

    def test_models_are_collected_before_next_training_and_no_checkpoints_are_saved(self):
        references = {}
        events = []

        def resource(name, cycle=False):
            value = Resource()
            if cycle:
                # Training helpers can leave model/optimizer/early-stopping cycles.
                value.training_helper = SimpleNamespace(model=value)
            references[name] = weakref.ref(value)
            events.append(name)
            return value

        def target_data(**kwargs):
            return resource('target_data')

        def target_model(**kwargs):
            self.assertIs(kwargs['data'], references['target_data']())
            self.assertEqual(kwargs['config'], self.config.target.model)
            return resource('target_model', cycle=True)

        def shadow_data(**kwargs):
            self.assertIs(kwargs['target_model'], references['target_model']())
            return resource('shadow_data')

        def shadow_model(**kwargs):
            self.assertIsNone(references['target_model']())
            self.assertIsNotNone(references['target_data']())
            self.assertIs(kwargs['data'], references['shadow_data']())
            return resource('shadow_model', cycle=True)

        def attack_data(**kwargs):
            for name in ('target_data', 'shadow_data', 'shadow_model'):
                self.assertIs(kwargs[name], references[name]())
            return resource('attack_data')

        def attack_model(**kwargs):
            for name in ('target_model', 'shadow_model', 'target_data', 'shadow_data'):
                self.assertIsNone(references[name](), name)
            self.assertEqual(kwargs['tail_ratio'], self.config.attack.tail_ratio)
            self.assertIs(kwargs['data'], references['attack_data']())
            return resource('attack_model', cycle=True)

        def test_model(stage, **kwargs):
            self.assertIs(kwargs['model'], references[f'{stage}_model']())
            self.assertIs(kwargs['data'], references[f'{stage}_data']())
            self.assertEqual(kwargs['seed'], 42)
            self.assertFalse(kwargs['deterministic'])
            self.assertEqual(kwargs['config'], self.config[stage].test)
            events.append(f'test_{stage}')
            return [[0.5]] * 3 if stage == 'attack' else [0.5]

        replacements = dict(
            instantiate=self.instantiate,
            load_target_data=target_data, load_target_model=target_model,
            load_shadow_data=shadow_data, load_shadow_model=shadow_model,
            load_attack_data=attack_data, load_attack_model=attack_model,
            test_target_model=lambda **kw: test_model('target', **kw),
            test_shadow_model=lambda **kw: test_model('shadow', **kw),
            test_attack_model=lambda **kw: test_model('attack', **kw),
        )
        with ExitStack() as stack:
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(pipeline, name, new=replacement))
            save = stack.enter_context(patch.object(torch, 'save'))
            pipeline.run_pipeline(self.config)
            save.assert_not_called()

        self.assertEqual(events, [
            'target_data', 'target_model', 'test_target',
            'shadow_data', 'shadow_model', 'test_shadow',
            'attack_data', 'attack_model', 'test_attack',
        ])
        for name, reference in references.items():
            self.assertIsNone(reference(), name)
        self.writer.close.assert_called_once_with()

    def test_writer_is_closed_on_failure(self):
        with patch.object(pipeline, 'instantiate', new=self.instantiate), \
                patch.object(pipeline, 'load_target_data', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                pipeline.run_pipeline(self.config)
        self.writer.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()

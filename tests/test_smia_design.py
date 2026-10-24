"""Compare the new target directly against the preserved SocialPP source."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import scipy.sparse as sp
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from torch_geometric.utils import to_undirected

from scripts import comparison_report, pipeline, pipeline_low_memory, reproduce_sal
from sralt.target.model import create_target_model, load_target_model
from sralt.target.model.smia_design import SMIADESIGN, DefendedSMIADESIGN
from sralt.utils import model_fingerprint, set_seed
from test_sal import make_data, configuration


AUTHOR = Path(__file__).resolve().parent / 'fixtures/socialpp_design'


def load_source(name):
    if name == 'utils':
        # Import only the original normalization helper, avoiding the unrelated
        # legacy pytz dependency used for log timestamps. Its AST is unchanged.
        tree = ast.parse((AUTHOR / 'utils.py').read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'normalize_sp']
        namespace = dict(np=np, sp=sp)
        exec(compile(tree, str(AUTHOR / 'utils.py'), 'exec'), namespace)
        return SimpleNamespace(normalize_sp=namespace['normalize_sp'])
    spec = importlib.util.spec_from_file_location('socialpp_reference_' + name, AUTHOR / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_model(model, social, behavior):
    """Load unmodified author methods with explicit matrices and shared raw tables."""
    with patch.dict(sys.modules, {'layers': load_source('layers'), 'utils': load_source('utils')}):
        source = load_source('models')
    n = model.num_users
    def row_graph(edges, count):
        matrix = torch.zeros(count, count, dtype=model.user_embs.weight.dtype)
        matrix[edges[0], edges[1]] = 1
        return matrix / matrix.sum(1, keepdim=True).clamp_min(1)
    social_matrix, rating_matrix = row_graph(social, n), row_graph(behavior, n + model.num_items)
    args = SimpleNamespace(hidden=model.embed_dim, neg=1, std=.1, hop=model.hop,
                           dropout=.5, decay=model.decay, ssl_temp=.2, ssl_reg=1e-6,
                           recon_reg=.2, recon_drop=.8, kl_reg=model.kl_reg)
    config = dict(n_users=n, n_target_users=n, n_items=model.num_items,
                  S=social_matrix.numpy(), A_tr=sp.coo_matrix(rating_matrix.numpy()), RRT_tr=None)
    reference = source.DESIGN(config, args, torch.device('cpu')).to(dtype=model.user_embs.weight.dtype)
    reference.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('defense.')})
    # Author constructors always cast matrices to float32. Supply double matrices
    # explicitly for the float64 equation/gradient check; no methods are replaced.
    reference.S, reference.A = social_matrix, rating_matrix.to_sparse()
    return reference


def small_config(target):
    config = configuration('design')
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        target_config = compose(config_name='config', overrides=[
            'target/data=ciao', f'target/model={target}', 'shadow/model=smia', 'attack/model=mlp',
        ]).target.model
    config.target.model = target_config
    config.target.model.attribute.embed_dim = 4
    config.target.model.train.num_epochs = 2
    config.target.model.train.train_batch_size = 7
    config.target.model.train.val_batch_size = 5
    config.target.model.train.val_num_negatives = 4
    for metric in config.target.model.train.metrics:
        metric.k = 2
    if target.endswith('_sal'):
        defense = config.target.model.attribute.defense
        defense.surrogate_warmup_epochs = 1
        defense.surrogate_epochs = 1
        defense.surrogate_batch_size = 9
    config.device = {'_target_': 'torch.device', 'device': 'cpu'}
    config.deterministic = True
    config.writer = None
    return config


class SMIADESIGNTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.are_deterministic_algorithms_enabled()
        set_seed(42, deterministic=True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous)

    def test_source_manifest_and_author_initialization_order(self):
        manifest = json.loads((AUTHOR / 'manifest.json').read_text())
        for name, digest in manifest['sha256'].items():
            self.assertEqual(hashlib.sha256((AUTHOR / name).read_bytes()).hexdigest(), digest)
        social = torch.tensor([[0, 0, 1], [1, 2, 2]])
        behavior = to_undirected(torch.tensor([[0, 1, 1], [4, 4, 5]]))
        set_seed(42, True)
        model = SMIADESIGN(4, 5, 3)
        # reference_model constructs the actual author tables before loading ours.
        set_seed(42, True)
        with patch.object(torch.nn.Module, 'load_state_dict'):
            reference = reference_model(model, social, behavior)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)

    def test_author_forward_loss_gradients_adam_and_distillation(self):
        # Directed social input catches transposed aggregation. Node 3 is isolated.
        social = torch.tensor([[0, 0, 0, 1, 2], [1, 2, 3, 2, 1]])
        behavior = to_undirected(torch.tensor([[0, 0, 1, 1, 2], [4, 5, 4, 6, 6]]))
        users, pos, neg = torch.tensor([0, 1, 1]), torch.tensor([4, 6, 4]), torch.tensor([7, 5, 8])
        for dtype in (torch.float32, torch.float64):
            for kl in (0., .7):
                with self.subTest(dtype=dtype, kl=kl):
                    model = SMIADESIGN(4, 5, 3, decay=.04, kl_reg=kl).to(dtype=dtype)
                    reference = reference_model(model, social, behavior)
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                    ref_optimizer = torch.optim.Adam(reference.parameters(), lr=.001)
                    for _ in range(2):
                        optimizer.zero_grad()
                        ref_optimizer.zero_grad()
                        branches = model.get_branch_embeddings(social, behavior)
                        outputs = reference(users, pos - 4, neg - 4)
                        for branch, offset in zip(branches, (0, 5, 8), strict=True):
                            torch.testing.assert_close(branch[users], outputs[offset])
                            torch.testing.assert_close(branch[pos], outputs[offset + 1])
                            torch.testing.assert_close(branch[neg], outputs[offset + 2])
                        torch.testing.assert_close(model.get_embedding(social, behavior), branches[0])
                        torch.testing.assert_close(branches[0][3], model.user_embs.weight[3] / 3)
                        logits = model(social, behavior, torch.stack([users, pos]))
                        torch.testing.assert_close(logits[:, 0], (outputs[0] * outputs[1]).sum(-1))
                        loss = model.calculate_loss(users, pos, neg, branches)
                        expected = reference.calculate_loss(users, pos - 4, neg - 4, 0)
                        torch.testing.assert_close(loss, expected)
                        loss.backward()
                        expected.backward()
                        for key, value in model.named_parameters():
                            torch.testing.assert_close(value.grad, dict(reference.named_parameters())[key].grad)
                        optimizer.step()
                        ref_optimizer.step()
                        for key, value in model.state_dict().items():
                            torch.testing.assert_close(value, reference.state_dict()[key])

    def test_reused_sal_matches_author_distance_gradients_and_update(self):
        cfg = small_config('design_smia_sal')
        model = create_target_model(42, True, torch.device('cpu'), cfg.target.model.attribute, 12, 20).double()
        data = make_data()
        behavior = to_undirected(data.behavioral_train_edge_index)
        reference = reference_model(model, data.social_edge_index, behavior)
        defender = load_source('attacker').Defender(8).double()
        defender.net.load_state_dict(model.defense.surrogate.net.state_dict())
        defender.requires_grad_(False)
        optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith('defense.')], lr=.001)
        ref_optimizer = torch.optim.Adam(reference.parameters(), lr=.001)
        positive = torch.tensor([[0, 2, 4], [1, 3, 5]])
        negative = torch.tensor([[0, 1, 3], [6, 8, 10]])
        for step in range(2):
            optimizer.zero_grad()
            ref_optimizer.zero_grad()
            torch.manual_seed(100 + step)
            actual = model.defense.distribution_loss(model.get_embedding(data.social_edge_index, behavior)[:12], positive, negative)
            torch.manual_seed(100 + step)
            expected = reference.pp_loss(*positive, *negative, defender)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-14)
            actual.backward()
            expected.backward()
            for key in ('user_embs.weight', 'item_embs.weight'):
                torch.testing.assert_close(dict(model.named_parameters())[key].grad,
                                           dict(reference.named_parameters())[key].grad, rtol=1e-9, atol=1e-14)
            self.assertTrue(all(p.grad is None for p in model.defense.parameters()))
            optimizer.step()
            ref_optimizer.step()
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=1e-10, atol=1e-13)

    def test_training_is_test_independent_and_uses_complete_training_graph(self):
        for target in ('design_smia', 'design_smia_sal'):
            cfg, data = small_config(target), make_data()
            changed = copy.deepcopy(data)
            changed.behavioral_test_edge_index[1] = 31  # Hidden labels must not affect training/candidates.
            model = create_target_model(42, True, torch.device('cpu'), cfg.target.model.attribute, 12, 20)
            with patch.object(model, 'get_branch_embeddings', wraps=model.get_branch_embeddings) as graphs:
                set_seed(42, True)
                model.train_model(data=data, writer=None, **cfg.target.model.train)
            self.assertEqual(graphs.call_count, 12)
            for call in graphs.call_args_list:
                torch.testing.assert_close(call.args[1], to_undirected(data.behavioral_train_edge_index))
            other = create_target_model(42, True, torch.device('cpu'), cfg.target.model.attribute, 12, 20)
            set_seed(42, True)
            other.train_model(data=changed, writer=None, **cfg.target.model.train)
            self.assertEqual(model_fingerprint(model), model_fingerprint(other))
            self.assertEqual(model.training_summary, other.training_summary)
            if target.endswith('_sal'):
                self.assertIsNone(model.training_history[0]['surrogate_bce'])
                self.assertIsNotNone(model.training_history[1]['surrogate_bce'])
                self.assertTrue(model.training_summary['selection'] == 'validation-only')
            self.assertTrue(model._graphs)
            restored = pickle.loads(pickle.dumps(model))
            self.assertEqual(restored._graphs, {})
            restored.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(restored.recommend(data=data, num_recommendations=4, batch_size=5),
                                       model.recommend(data=data, num_recommendations=4, batch_size=5))
            known = set(map(tuple, data.behavioral_edge_index.t().tolist()))
            self.assertFalse(known & set(map(tuple, model.recommend(data=data, num_recommendations=4, batch_size=5).t().tolist())))
            model.double()
            self.assertEqual(model._graphs, {})

    def test_zero_sal_matches_baseline_without_sampling_or_rng_drift(self):
        base, defended, data = small_config('design_smia'), small_config('design_smia_sal'), make_data()
        defended.target.model.attribute.defense.loss_weight = 0
        models, states = [], []
        with patch('sralt.target.model.smia_design.model.SocialPairSampler', side_effect=AssertionError('Zero SAL sampled')):
            for cfg in (base, defended):
                model = create_target_model(42, True, torch.device('cpu'), cfg.target.model.attribute, 12, 20)
                set_seed(42, True)
                model.train_model(data=data, writer=None, **cfg.target.model.train)
                models.append(model)
                states.append(torch.get_rng_state())
        self.assertIsNone(models[1].defense)
        self.assertEqual(models[0].state_dict().keys(), models[1].state_dict().keys())
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(*states, rtol=0, atol=0)

    def test_full_chain_both_pipeline_entries_and_cache_isolation(self):
        data, device = make_data(), torch.device('cpu')
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                for target in ('design_smia', 'design_smia_sal'):
                    cfg = small_config(target)
                    with patch.object(pipeline, 'load_target_data', return_value=data):
                        td, model = pipeline.target_stage(42, True, device, None, cfg.target)
                    sd, sm = pipeline.shadow_stage(42, True, device, None, cfg.shadow, td, model)
                    ad, am = pipeline.attack_stage(42, True, device, None, cfg.attack, td, sd, sm)
                    self.assertEqual(ad.user_features.size(1), 16)
                    self.assertTrue(all('surrogate' not in key for key in sm.state_dict()))
                    with patch.object(type(model), 'train_model', side_effect=AssertionError('Target cache miss')):
                        cached = load_target_model(42, True, device, cfg.target.model, data)
                        with patch.object(pipeline_low_memory, 'load_target_data', return_value=data), \
                                patch.object(pipeline_low_memory, 'instantiate', side_effect=lambda c: Mock() if c is None else instantiate(c)):
                            pipeline_low_memory.run_pipeline(cfg)
                    self.assertEqual(model_fingerprint(model), model_fingerprint(cached))
                    self.assertEqual(model.training_summary, cached.training_summary)
                self.assertEqual(len(list(Path('.cache/target/model/DESIGN_SMIA').glob('*.pkl'))), 1)
                self.assertEqual(len(list(Path('.cache/target/model/DESIGN_SMIA_SAL').glob('*.pkl'))), 1)
                self.assertEqual(len(list(Path('.cache/shadow/data').rglob('*.pkl'))), 2)
                self.assertEqual(len(list(Path('.cache/shadow/model').rglob('*.pkl'))), 2)
                self.assertEqual(len(list(Path('.cache/attack/model').rglob('*.pkl'))), 2)
            finally:
                os.chdir(previous)

    def test_configs_report_and_runner_keep_new_and_old_models_separate(self):
        for dataset in ('ciao', 'flickr'):
            cfg = comparison_report.compose_case(dataset, 'design_smia_sal', 'smia', 'mlp', 42)
            self.assertEqual(cfg.target.model.attribute.defense.loss_weight, .01)
            self.assertEqual(cfg.target.model.train.train_batch_size, 1024)
            self.assertEqual(cfg.target.model.train.val_batch_size, 2048)
            self.assertEqual(cfg.target.model.train.optimizer.lr, .001)
            self.assertEqual(cfg.target.model.train.num_epochs, 200)
            self.assertEqual([m.k for m in cfg.target.model.train.metrics], [10, 10])
            self.assertEqual([m._target_.rsplit('.', 1)[-1] for m in cfg.target.model.train.metrics],
                             ['LinkPredRecall', 'LinkPredNDCG'])
            self.assertEqual(cfg.target.model.train.early_stopping.patience, 11)
            self.assertEqual(cfg.shadow.data.revision, 'attack-train-only-v1')
        args = reproduce_sal.parse_args(['--target', 'design_smia', '--datasets', 'ciao', 'flickr'])
        self.assertEqual(list(reproduce_sal.cases(args)), [('ciao', 42, None), ('ciao', 42, .01),
                                                         ('flickr', 42, None), ('flickr', 42, .01)])
        args = comparison_report.parse_args(['--datasets', 'ciao', '--targets', 'design_smia', 'design_smia_sal',
                                            '--pairs', 'smia:mlp', '--seeds', '42', '--sal-weights', '.01', '.1', '--dry-run'])
        with tempfile.TemporaryDirectory() as directory, patch.object(comparison_report, 'CACHE_DIR', Path(directory)):
            rows = comparison_report.collect(args)['rows']
        self.assertEqual([r['defense'] for r in rows], ['None', 'SAL λ=0.01', 'SAL λ=0.1'])
        self.assertEqual({r['target'] for r in rows}, {'design_smia'})
        self.assertEqual(comparison_report.compose_case('ciao', 'design', 'smia', 'mlp', 42).target.model.revision,
                         'author-aligned-v1')


if __name__ == '__main__':
    unittest.main()

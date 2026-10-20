import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from torch_geometric.utils import to_undirected

from sralt.target.data import TargetData
from sralt.target.model import create_target_model, load_target_model, test_target_model
from sralt.target.model.egcl.dcl import DualContrastiveLearning
from sralt.target.model.egcl.gap import AdaptivePropagation, AttentiveAggregation
from sralt.target.model.egcl.weight import AdaptiveWeight, l2_loss
from sralt.target.model.gdmsr.data import build_item_history, sample_social_relations
from sralt.target.model.gdmsr.denoiser import Denoiser, GCNLayer
from sralt.utils import model_fingerprint, set_seed


def make_data() -> TargetData:
    data = TargetData(name='synthetic', directory='unused')
    data.num_users, data.num_items = 3, 8
    # User 2 is isolated, including after curriculum sampling.
    data.social_edge_index = torch.tensor([[0, 1], [1, 0]])
    data.behavioral_train_edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [3, 4, 4, 5, 5, 6]])
    data.behavioral_val_edge_index = torch.tensor([[0, 1, 2], [5, 6, 7]])
    data.behavioral_test_edge_index = torch.tensor([[0, 1, 2], [6, 7, 8]])
    return data.to(torch.device('cpu'))


def make_denoiser(**kwargs) -> Denoiser:
    attributes = dict(num_users=3, num_items=8, embed_dim=4, num_layers=2,
                      alpha=0.2, beta=0.5, gamma=1.0, epsilon=2, r=1.0, decay=0.1)
    return Denoiser(**(attributes | kwargs))


def compose_config(name: str):
    with initialize_config_module(config_module='sralt.configs', version_base=None):
        overrides = [
            'target/data=ciao', f'target/model={name}', 'shadow/model=csa', 'attack/model=mlp',
            'target.model.attribute.embed_dim=4', 'target.model.train.num_epochs=2',
            'target.model.train.train_batch_size=4', 'target.model.train.val_batch_size=2',
            'target.model.train.val_num_negatives=3', 'target.model.train.metrics.0.k=2',
            'target.model.train.metrics.1.k=2', 'target.test.evaluation.test_batch_size=2',
            'target.test.evaluation.test_num_negatives=3', 'target.test.metrics.0.k=2',
            'target.test.metrics.1.k=2'
        ]
        if name == 'gdmsr':
            overrides += ['target.model.attribute.sequence_length=2', 'target.model.attribute.curriculum_interval=1']
        else:
            overrides += ['target.model.attribute.contrastive_batch_size=2']
        return compose(config_name='config', overrides=overrides)


class GDMSRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_relation_convolution_matches_dense_with_asymmetric_mask(self):
        layer = GCNLayer(4).double()
        initial = torch.randn(11, 4, dtype=torch.double, requires_grad=True)
        social = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        behavior = to_undirected(make_data().behavioral_train_edge_index)
        mask = torch.tensor([True, False, True, True])

        def reference(edges, linear, n, edge_mask=None):
            adj = torch.zeros(n, n, dtype=torch.double)
            for src, dst in edges.t().tolist():
                adj[dst, src] = 1
            out_degree = adj.sum(0).clamp(min=1)
            in_degree = adj.sum(1).clamp(min=1)
            if edge_mask is not None:
                for (src, dst), keep in zip(edges.t().tolist(), edge_mask.tolist()):
                    if not keep:
                        adj[dst, src] = 0
            normalized = in_degree.rsqrt()[:, None] * adj * out_degree.rsqrt()[None, :]
            return F.relu(F.linear(normalized @ initial[:n], linear.weight, linear.bias))

        expected_social = reference(social, layer.social, 3, mask)
        expected_user = reference(behavior[:, behavior[0] >= 3], layer.item_to_user, 11)
        expected_item = reference(behavior[:, behavior[0] < 3], layer.user_to_item, 11)
        expected = torch.cat([(expected_social + expected_user[:3]) / 2, expected_item[3:]])
        actual = layer(embeds=initial, num_users=3, social_edge_index=social,
                       behavioral_edge_index=behavior, social_edge_mask=mask)
        torch.testing.assert_close(actual, expected)
        params = (initial, *layer.parameters())
        actual_grads = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), params)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_concatenated_readout_and_author_regularization(self):
        denoiser = make_denoiser().eval()
        data = make_data()
        behavior = to_undirected(data.behavioral_train_edge_index)
        mask = torch.ones(2, dtype=torch.bool)
        history, padding = build_item_history(data.behavioral_train_edge_index, 3, 8, 2)
        users, pos, neg = torch.tensor([0, 1]), torch.tensor([3, 4]), torch.tensor([7, 8])
        peers, negative_peers = torch.tensor([1, 0]), torch.tensor([2, 2])
        embeds = denoiser.get_embedding(social_edge_index=data.social_edge_index,
                                       behavioral_edge_index=behavior, social_edge_mask=mask)
        self.assertEqual(embeds.shape, (11, 12))
        torch.testing.assert_close(embeds[:, :4], denoiser.embeds.weight)
        positive = (embeds[users] * embeds[pos]).sum(1)
        negative = (embeds[users] * embeds[neg]).sum(1)
        probabilities = torch.cat([
            denoiser._relation_confidence_forward(history, padding, torch.stack([users, peers])),
            denoiser._relation_confidence_forward(history, padding, torch.stack([users, negative_peers]))
        ])
        expected = 0.8 * F.softplus(negative - positive).mean()
        expected += 0.1 * (embeds[users].square().sum() + embeds[pos].square().sum()) / 2
        expected += 0.2 * F.binary_cross_entropy(probabilities, torch.tensor([1., 1., 0., 0.]))
        actual = denoiser(social_edge_index=data.social_edge_index, behavioral_edge_index=behavior,
                          social_edge_mask=mask, item_history=history, item_history_mask=padding,
                          users=users, pos_items=pos, neg_items=neg, pos_peers=peers, neg_peers=negative_peers)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in denoiser.parameters() if p.grad is not None))
        self.assertEqual(denoiser.transformer.linear1.out_features, 2048)

    def test_curriculum_ema_cumulative_mask_and_degree_boundary(self):
        denoiser = make_denoiser()
        edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        first = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        history = torch.zeros(3, 1, dtype=torch.long)
        with patch.object(denoiser, '_relation_confidence_forward', return_value=first):
            confidence, mask = denoiser.denoise_social_graph(history, history.bool(), edges,
                                                            torch.zeros(6), 6, initialize=True)
            torch.testing.assert_close(confidence, first)
            self.assertTrue(mask.all())
            confidence, mask = denoiser.denoise_social_graph(history, history.bool(), edges, confidence, 6,
                                                            social_edge_mask=mask)
        self.assertEqual(mask.tolist(), [False, True, False, True, False, True])
        second = torch.tensor([0.9, 0.0, 0.8, 0.0, 0.7, 0.0])
        with patch.object(denoiser, '_relation_confidence_forward', return_value=second):
            smoothed, mask = denoiser.denoise_social_graph(history, history.bool(), edges, confidence, 6,
                                                          social_edge_mask=mask)
        torch.testing.assert_close(smoothed, (first + second) / 2)
        self.assertFalse(mask.any())
        users = torch.tensor([0, 1, 2, 2])
        positives, negatives = sample_social_relations(users, torch.empty(2, 0, dtype=torch.long), 3)
        torch.testing.assert_close(positives, users)
        self.assertTrue(((negatives >= 0) & (negatives < 3)).all())
        positives, _ = sample_social_relations(users, make_data().social_edge_index, 3)
        self.assertEqual(positives.tolist(), [1, 0, 2, 2])

    def test_curriculum_refreshes_sampling_after_training_in_eval_mode(self):
        config = compose_config('gdmsr')
        config.target.model.train.num_epochs = 3
        data = make_data()
        model = create_target_model(42, False, torch.device('cpu'), config.target.model.attribute, 3, 8)
        updates = []

        def denoise(**kwargs):
            self.assertFalse(model.denoiser.training)
            updates.append(kwargs['initialize'])
            mask = torch.full((2,), kwargs['initialize'], dtype=torch.bool)
            return torch.zeros(2), mask

        with patch.object(model.denoiser, 'denoise_social_graph', side_effect=denoise), patch(
            'sralt.target.model.gdmsr.data.sample_social_relations', wraps=sample_social_relations
        ) as sample, patch(
            'sralt.target.model.gdmsr.gdmsr.compute_recommendation_metrics', return_value=[0.5, 0.4]
        ):
            model.train_model(data=data, writer=None, **config.target.model.train)
        self.assertEqual(updates, [True, False, False])
        self.assertEqual(len(sample.call_args_list), 3)
        self.assertEqual([call.kwargs['edge_index'].size(1) for call in sample.call_args_list], [2, 2, 0])

    def test_padding_is_ignored_and_history_prefers_popular_items(self):
        denoiser = make_denoiser().eval()
        edges = torch.tensor([[0, 0, 0, 1, 2], [3, 4, 5, 4, 4]])
        history, padding = build_item_history(edges, 3, 8, 2)
        self.assertEqual(history[0].tolist(), [4, 3])
        pairs = torch.tensor([[1, 2], [2, 1]])
        expected = denoiser._relation_confidence_forward(history, padding, pairs)
        modified = history.clone()
        modified[padding] = 10
        actual = denoiser._relation_confidence_forward(modified, padding, pairs)
        torch.testing.assert_close(actual, expected)


class EGCLTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_weighted_square_loss_matches_explicit_all_pairs_and_gradients(self):
        weight = AdaptiveWeight(3, offset=8).double()
        left = torch.randn(3, 3, dtype=torch.double, requires_grad=True)
        right = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
        edges = torch.tensor([[0, 0, 1, 2], [0, 2, 3, 4]])
        for include_constant in [True, False]:
            with self.subTest(include_constant=include_constant):
                scores = left @ right.t()
                weights = weight(left[edges[0]], right[edges[1]])
                observed = scores[edges[0], edges[1]]
                expected = scores.square().sum() - observed.square().sum()
                expected += (weights * (observed - 1).square()).sum()
                if not include_constant:
                    expected -= weights.sum()
                actual = weight.square_loss(left, right, edges, 2, include_constant)
                torch.testing.assert_close(actual, expected)
                params = (left, right, *weight.parameters())
                actual_grads = torch.autograd.grad(actual, params, retain_graph=True)
                expected_grads = torch.autograd.grad(expected, params, retain_graph=True)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_infonce_chunks_match_full_node_loss_and_gradients(self):
        dcl = DualContrastiveLearning(0.05, 1e-6, 1e-8, batch_size=2)
        first = torch.randn(7, 4, dtype=torch.double, requires_grad=True)
        second = torch.randn(7, 4, dtype=torch.double, requires_grad=True)
        actual = dcl.info_nce(first, second)
        expected = F.cross_entropy(F.normalize(first) @ F.normalize(second).t() / 0.05,
                                   torch.arange(7), reduction='sum')
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual, (first, second), retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (first, second))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_propagation_attention_neighbor_normalization_and_cache_lifecycle(self):
        data = make_data()
        edges = to_undirected(data.behavioral_train_edge_index)
        initial = torch.randn(11, 4)
        propagate = AdaptivePropagation(3)
        dense = torch.zeros(11, 11)
        dense[edges[0], edges[1]] = 1
        degrees = dense.sum(1)
        normalized = dense * degrees.clamp(min=1).rsqrt()[:, None] * degrees.clamp(min=1).rsqrt()[None, :]
        time = (degrees + 1e-7).log().sigmoid()
        layers = []
        current = initial
        factorial = 1
        for k in range(3):
            current = normalized @ current
            if k > 0:
                factorial *= k
            layers.append(current * ((-time).exp() * time.pow(k) / factorial)[:, None])
        torch.testing.assert_close(propagate(initial, edges), torch.stack(layers, 1))
        readout = AttentiveAggregation(4)
        torch.testing.assert_close(readout(initial, layers[0][:, None]), layers[0])
        propagate.to('cpu')
        self.assertIsNone(propagate._edge_index)
        self.assertFalse(any('degrees' in key or 'weights' in key for key in propagate.state_dict()))
        dcl = DualContrastiveLearning(0.05, 1e-6, 1e-8, 2)
        social_embeds = torch.randn(3, 4)
        social_edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        actual_behavior, actual_social = dcl.neighbor_views(social_edges, edges, initial, social_embeds)
        rating = torch.zeros(3, 8)
        rating[data.behavioral_train_edge_index[0], data.behavioral_train_edge_index[1] - 3] = 1
        rating /= rating.sum(0).clamp(min=1)
        torch.testing.assert_close(actual_behavior[:3], rating @ initial[3:])
        torch.testing.assert_close(actual_behavior[3:], rating.t() @ initial[:3])
        social = torch.zeros(3, 3)
        social[social_edges[0], social_edges[1]] = 1
        social /= social.sum(0).clamp(min=1)
        expected_social = social @ social_embeds
        torch.testing.assert_close(actual_social, expected_social)

    def test_full_objective_and_optimizer_parameter_isolation(self):
        config = compose_config('egcl')
        model = create_target_model(42, False, torch.device('cpu'), config.target.model.attribute, 3, 8)
        data = make_data()
        behavior = to_undirected(data.behavioral_train_edge_index)
        behavioral_embeds, social_embeds = model.gap(data.social_edge_index, behavior)
        torch.testing.assert_close(model.get_embedding(data.social_edge_index, behavior), behavioral_embeds)
        weight_params = [*model.behavioral_weight.parameters(), *model.social_weight.parameters()]
        embedding_params = list(model.gap.parameters())
        weight_optimizer = torch.optim.Adam(weight_params, lr=0.01)
        embedding_optimizer = torch.optim.Adam(embedding_params, lr=0.01)
        before = [p.detach().clone() for p in embedding_params]
        model(data.social_edge_index, behavior, 2, weight_phase=True).backward()
        self.assertTrue(all(p.grad is None for p in embedding_params))
        weight_optimizer.step()
        for previous, current in zip(before, embedding_params):
            torch.testing.assert_close(previous, current)
        weight_optimizer.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in weight_params]
        loss = model(data.social_edge_index, behavior, 2)
        behavioral_embeds, social_embeds = model.gap(data.social_edge_index, behavior)
        interaction_edges = data.behavioral_train_edge_index.clone()
        interaction_edges[1] -= 3
        expected = model.behavioral_weight.square_loss(behavioral_embeds[:3], behavioral_embeds[3:], interaction_edges, 99, False)
        expected += model.beta * model.social_weight.square_loss(social_embeds, social_embeds, data.social_edge_index, 99, False)
        expected += model.embedding_reg * l2_loss([model.gap.user_embeds.weight, model.gap.item_embeds.weight])
        expected += model.weight_reg * l2_loss([*model.gap.behavioral_user_readout.parameters(), *model.gap.behavioral_item_readout.parameters()])
        expected += model.beta * model.weight_reg * l2_loss(model.gap.social_readout.parameters())
        expected += model.dcl(data.social_edge_index, behavior, behavioral_embeds, social_embeds)
        torch.testing.assert_close(loss, expected)
        torch.autograd.backward(loss, inputs=embedding_params)
        self.assertTrue(all(p.grad is None for p in weight_params))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in embedding_params))
        embedding_optimizer.step()
        for previous, current in zip(before, weight_params):
            torch.testing.assert_close(previous, current)


class SocialTargetIntegrationTests(unittest.TestCase):
    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def test_cpu_training_protocol_checkpoint_and_disk_cache(self):
        for name in ['gdmsr', 'egcl']:
            for deterministic in [False, True]:
                with self.subTest(name=name, deterministic=deterministic):
                    set_seed(42, deterministic=deterministic)
                    config = compose_config(name)
                    data = make_data()
                    model = create_target_model(42, deterministic, torch.device('cpu'), config.target.model.attribute, 3, 8)
                    untrained_fingerprint = model_fingerprint(model)
                    model.eval()
                    model.recommend(data=data, num_recommendations=2, batch_size=2)
                    self.assertEqual(model_fingerprint(model), untrained_fingerprint)
                    before = copy.deepcopy(model.state_dict())
                    training_graph = to_undirected(data.behavioral_train_edge_index)
                    with patch.object(model, 'get_embedding', wraps=model.get_embedding) as get_embedding:
                        model.train_model(data=data, writer=None, **config.target.model.train)
                        for call in get_embedding.call_args_list:
                            graph = call.kwargs.get('behavioral_edge_index') if name == 'gdmsr' else call.args[1]
                            torch.testing.assert_close(graph, training_graph)
                    self.assertTrue(any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
                    with patch.object(model, 'get_embedding', wraps=model.get_embedding) as get_embedding:
                        values = test_target_model(42, deterministic, torch.device('cpu'), model, data, config.target.test)
                        call = get_embedding.call_args
                        graph = call.kwargs.get('behavioral_edge_index') if name == 'gdmsr' else call.args[1]
                        torch.testing.assert_close(graph, training_graph)
                    self.assertTrue(all(0 <= v <= 1 for v in values))
                    fingerprint = model_fingerprint(model)
                    with patch.object(model, 'get_embedding', wraps=model.get_embedding) as get_embedding:
                        recommended = model.recommend(data=data, num_recommendations=2, batch_size=2)
                        call = get_embedding.call_args
                        graph = call.kwargs.get('behavioral_edge_index') if name == 'gdmsr' else call.args[1]
                        torch.testing.assert_close(graph, to_undirected(data.behavioral_edge_index))
                    self.assertEqual(recommended.shape, (2, 6))
                    torch.testing.assert_close(torch.bincount(recommended[0], minlength=3), torch.full((3,), 2))
                    self.assertTrue(((recommended[1] >= 3) & (recommended[1] < 11)).all())
                    history = set(map(tuple, data.behavioral_edge_index.t().tolist()))
                    self.assertFalse(history.intersection(map(tuple, recommended.t().tolist())))
                    self.assertEqual(model_fingerprint(model), fingerprint)
                    restored = create_target_model(42, deterministic, torch.device('cpu'), config.target.model.attribute, 3, 8)
                    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                    torch.testing.assert_close(restored.recommend(data=data, num_recommendations=2, batch_size=1), recommended)
                    self.assertEqual(repr(model), repr(restored))
                    with tempfile.TemporaryDirectory() as directory:
                        previous_cwd = os.getcwd()
                        try:
                            os.chdir(directory)
                            with patch.dict(os.environ, {'SRALT_CACHE_VERSION': 'social-target-test', 'SRALT_CACHE_REFRESH': '0'}):
                                with patch.object(type(model), 'train_model') as train:
                                    cached = load_target_model(42, deterministic, torch.device('cpu'), config.target.model, data)
                                    self.assertEqual(train.call_count, 1)
                                with patch.object(type(model), 'train_model', side_effect=AssertionError('cache hit retrained')):
                                    hit = load_target_model(42, deterministic, torch.device('cpu'), config.target.model, data)
                                self.assertEqual(repr(cached), repr(hit))
                                legacy = copy.deepcopy(config.target.model)
                                del legacy.revision
                                with patch.object(type(model), 'train_model') as train:
                                    load_target_model(42, deterministic, torch.device('cpu'), legacy, data)
                                    self.assertEqual(train.call_count, 1)
                        finally:
                            os.chdir(previous_cwd)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

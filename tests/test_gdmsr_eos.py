import copy
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_module

from sralt.target.model import create_target_model
from sralt.target.model.gdmsr.denoiser import Denoiser


def make_denoiser(dim_feedforward: int = 2048) -> Denoiser:
    return Denoiser(3, 8, 4, 2, 0.2, 0.5, 1.0, 2, 1.0, 0.1, dim_feedforward)


def full_relation_confidence(
    model: Denoiser, history: torch.Tensor, padding: torch.Tensor, pairs: torch.Tensor
) -> torch.Tensor:
    """Independent reference using the original full-sequence layer forward."""
    left, right = pairs
    batch = pairs.size(1)
    sequence = torch.cat([
        model.embeds(history[left]), model.sep_token.expand(batch, -1, -1),
        model.embeds(history[right]), model.eos_token.expand(batch, -1, -1)
    ], dim=1)
    special = torch.zeros(batch, 1, dtype=torch.bool)
    mask = torch.cat([padding[left], special, padding[right], special], dim=1)
    encoded = model.transformer(sequence, src_key_padding_mask=mask)
    return model.scorer(encoded[:, -1]).squeeze(-1)


class GDMSREOSTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_full_sequence_scores_and_all_parameter_gradients_match(self):
        for width in [16, 2048]:
            for training in [False, True]:
                with self.subTest(width=width, training=training):
                    optimized = make_denoiser(width).double().train(training)
                    # Disable all dropout, including MHA's functional dropout.
                    optimized.transformer.self_attn.dropout = 0.0
                    for module in optimized.modules():
                        if isinstance(module, torch.nn.Dropout):
                            module.p = 0.0
                    reference = copy.deepcopy(optimized)
                    history = torch.tensor([[3, 4, 5], [6, 3, 3], [3, 3, 3]])
                    padding = torch.tensor([[False, False, False], [False, True, True], [True, True, True]])
                    pairs = torch.tensor([[0, 1, 2, 2, 0], [1, 2, 0, 2, 0]])
                    actual = optimized._relation_confidence_forward(history, padding, pairs)
                    expected = full_relation_confidence(reference, history, padding, pairs)
                    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
                    weights = torch.arange(1, 6, dtype=torch.double)
                    (actual * weights).sum().backward()
                    (expected * weights).sum().backward()
                    for (name, parameter), (reference_name, reference_parameter) in zip(
                        optimized.named_parameters(), reference.named_parameters()
                    ):
                        self.assertEqual(name, reference_name)
                        self.assertEqual(parameter.grad is None, reference_parameter.grad is None, name)
                        if parameter.grad is not None:
                            torch.testing.assert_close(parameter.grad, reference_parameter.grad,
                                                       rtol=1e-9, atol=1e-11, msg=name)

    def test_single_query_keeps_all_keys_and_padding_mask(self):
        model = make_denoiser(16).eval()
        sequence = torch.randn(3, 8, 4, requires_grad=True)
        padding = torch.tensor([[False] * 8, [True] * 6 + [False] * 2, [False] * 8])
        with patch.object(model.transformer.self_attn, 'forward', wraps=model.transformer.self_attn.forward) as attention, patch.object(
            model.transformer.linear1, 'forward', wraps=model.transformer.linear1.forward
        ) as ffn:
            actual = model._encode_eos(sequence, padding)
        self.assertEqual(attention.call_args.kwargs['query'].shape, (3, 1, 4))
        self.assertIs(attention.call_args.kwargs['key'], sequence)
        self.assertIs(attention.call_args.kwargs['value'], sequence)
        self.assertIs(attention.call_args.kwargs['key_padding_mask'], padding)
        self.assertFalse(attention.call_args.kwargs['need_weights'])
        self.assertEqual(ffn.call_args.args[0].shape, (3, 1, 4))
        modified = sequence.detach().clone()
        modified[padding] = 100
        with torch.no_grad():
            torch.testing.assert_close(model._encode_eos(modified, padding), actual)
        actual.square().sum().backward()
        self.assertGreater(sequence.grad[0, :-1].abs().sum().item(), 0)
        self.assertEqual(sequence.grad[padding].abs().sum().item(), 0)

    def test_training_dropout_backward_and_default_checkpoint_compatibility(self):
        model = make_denoiser().train()
        history = torch.tensor([[3, 4], [5, 3], [3, 3]])
        padding = torch.tensor([[False, False], [False, True], [True, True]])
        pairs = torch.tensor([[0, 1, 2], [1, 2, 0]])
        loss = model._relation_confidence_forward(history, padding, pairs).sum()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        # A layer constructed with the former implicit default has exactly the same state.
        old_layer = torch.nn.TransformerEncoderLayer(d_model=4, nhead=1, batch_first=True)
        restored = make_denoiser()
        state = copy.deepcopy(model.state_dict())
        state.update({f'transformer.{key}': value for key, value in old_layer.state_dict().items()})
        restored.load_state_dict(state, strict=True)
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored._relation_confidence_forward(history, padding, pairs),
                                       full_relation_confidence(restored, history, padding, pairs))

    def test_config_exposes_ffn_width_and_passes_it_to_the_layer(self):
        for width in [2048, 256]:
            with self.subTest(width=width), initialize_config_module(config_module='sralt.configs', version_base=None):
                config = compose(config_name='config', overrides=[
                    'target/data=ciao', 'target/model=gdmsr', 'shadow/model=smia', 'attack/model=mlp',
                    f'target.model.attribute.dim_feedforward={width}'
                ])
                model = create_target_model(42, False, torch.device('cpu'), config.target.model.attribute, 3, 8)
                self.assertEqual(model.dim_feedforward, width)
                self.assertEqual(model.denoiser.transformer.linear1.out_features, width)
                self.assertEqual(model.denoiser.transformer.linear2.in_features, width)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

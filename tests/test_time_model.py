"""Exercise the actual JAX transformer; only the vision encoder is reduced for CPU tests."""
import dataclasses
import unittest
from unittest import mock

import flax.linen as linen
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import nnx_utils


class TinyVision(linen.Module):
    num_classes: int
    variant: str
    pool_type: str
    scan: bool
    dtype_mm: str

    @linen.compact
    def __call__(self, images, *, train=False):
        pooled = jnp.mean(images, axis=(1, 2))[:, None, :]
        return linen.Dense(self.num_classes)(pooled), {}


class ContextModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = pi0_config.Pi0Config(
            pi05=True, action_dim=7, action_horizon=10, max_token_len=4,
            paligemma_variant='dummy', action_expert_variant='dummy',
            action_context_horizon=9, dtype='float32',
        )
        with mock.patch.object(pi0._siglip, 'Module', TinyVision):
            cls.model = cls.config.create(jax.random.key(0))
        # Fresh PI0.5 AdaRMS gates are zero, which deliberately disables attention
        # residuals. Open the action-expert gates in this random-weight fixture;
        # production validation below uses the unmodified trained checkpoint.
        gates = 0
        state = nnx.state(cls.model)
        for path, variable in state.flat_state().items():
            if path[-1] == 'bias' and any(name in path for name in ('pre_attention_norm_1', 'pre_ffw_norm_1')):
                width = variable.value.shape[-1] // 3
                variable.value = variable.value.at[..., -width:].set(0.5)
                gates += 1
        if gates == 0:
            raise AssertionError('No action-expert AdaRMS gates found in test model')
        nnx.update(cls.model, state)
        obs = cls.config.fake_obs()
        cls.obs = dataclasses.replace(
            obs, action_context=jnp.zeros((1, 9, 7)),
            action_context_mask=jnp.arange(9)[None, :] < 4,
        )
        cls.noise = jax.random.normal(jax.random.key(8), (1, 10, 7))
        cls.sample = nnx_utils.module_jit(cls.model.sample_actions)

    def sample_with(self, obs):
        return np.asarray(type(self).sample(jax.random.key(7), obs, noise=self.noise, num_steps=2))

    def test_valid_context_changes_generated_actions(self):
        base = self.sample_with(self.obs)
        changed = dataclasses.replace(self.obs, action_context=self.obs.action_context.at[:, :4].set(0.5))
        delta = np.max(np.abs(base-self.sample_with(changed)))
        self.assertTrue(np.isfinite(base).all())
        self.assertGreater(float(delta), 1e-5)

    def test_masked_padding_does_not_change_generated_actions(self):
        changed = dataclasses.replace(self.obs, action_context=self.obs.action_context.at[:, 4:].set(17.0))
        np.testing.assert_allclose(self.sample_with(self.obs), self.sample_with(changed), atol=1e-5, rtol=1e-5)

    def test_context_length_changes_generated_actions(self):
        shorter = dataclasses.replace(self.obs, action_context_mask=jnp.arange(9)[None, :] < 1)
        self.assertGreater(float(np.max(np.abs(self.sample_with(self.obs)-self.sample_with(shorter)))), 1e-5)

    def test_empty_and_maximum_context_are_finite(self):
        for k in (0, 9):
            with self.subTest(k=k):
                obs = dataclasses.replace(self.obs, action_context_mask=jnp.arange(9)[None, :] < k)
                actions = self.sample_with(obs)
                self.assertEqual(actions.shape, (1, 10, 7))
                self.assertTrue(np.isfinite(actions).all())

    def test_context_attention_precedes_generated_block(self):
        tokens, mask, ar_mask, _ = self.model.embed_suffix(self.obs, self.noise, jnp.ones((1,)))
        self.assertEqual(tokens.shape[1], 19)
        attn = np.asarray(pi0.make_attn_mask(mask, ar_mask))[0]
        self.assertTrue(attn[:4, :4].all())
        self.assertFalse(attn[:9, 9:].any())
        self.assertTrue(attn[9:, :4].all())
        self.assertFalse(attn[:, 4:9].any())

    def test_missing_and_wrong_context_are_rejected(self):
        for obs in (
            dataclasses.replace(self.obs, action_context=None),
            dataclasses.replace(self.obs, action_context_mask=None),
            dataclasses.replace(self.obs, action_context=jnp.zeros((1, 8, 7)), action_context_mask=jnp.ones((1, 8), dtype=bool)),
        ):
            with self.subTest(shape=None if obs.action_context is None else obs.action_context.shape):
                with self.assertRaises(ValueError):
                    self.model.embed_suffix(obs, self.noise, jnp.ones((1,)))

    def test_strict_loading_rejects_missing_and_extra_parameters(self):
        params = nnx.state(self.model).to_pure_dict()
        missing = dict(params)
        missing.pop('action_context_in_proj')
        extra = dict(params, unused_parameter=jnp.zeros((1,)))
        with mock.patch.object(pi0._siglip, 'Module', TinyVision):
            for tree in (missing, extra):
                with self.subTest(extra='unused_parameter' in tree):
                    with self.assertRaises((ValueError, KeyError)):
                        self.config.load(tree, remove_extra_params=False)
            restored = self.config.load(params, remove_extra_params=False)
        np.testing.assert_array_equal(
            np.asarray(restored.action_context_in_proj.kernel.value),
            np.asarray(self.model.action_context_in_proj.kernel.value),
        )

    def test_context_disabled_preserves_original_suffix(self):
        config = dataclasses.replace(self.config, action_context_horizon=0)
        with mock.patch.object(pi0._siglip, 'Module', TinyVision):
            model = config.create(jax.random.key(0))
        obs = config.fake_obs()
        self.assertIsNone(obs.action_context)
        self.assertFalse(hasattr(model, 'action_context_in_proj'))
        tokens, _, _, _ = model.embed_suffix(obs, self.noise, jnp.ones((1,)))
        self.assertEqual(tokens.shape[1], 10)


if __name__ == '__main__':
    unittest.main()

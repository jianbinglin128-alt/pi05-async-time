"""Check real Future weights with fixed images/noise and controlled context changes."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from serve_time import load_future_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    policy, metadata = load_future_policy(args.checkpoint)
    rng = np.random.default_rng(7)
    observation = {
        'observation/image': rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        'observation/wrist_image': rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        'observation/state': np.zeros(8, dtype=np.float32),
        'prompt': 'pick up the black bowl and place it on the plate',
    }
    noise = rng.standard_normal((10, 32)).astype(np.float32)

    def infer(context, mask=None):
        obs = dict(observation, action_context=context)
        if mask is not None:
            obs['action_context_mask'] = mask
        policy._rng = jax.random.key(7)
        actions = np.asarray(policy.infer(obs, noise=noise)['actions'])
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError('Invalid real-model actions')
        return actions

    base_context = np.zeros((4, 7), dtype=np.float32)
    base = infer(base_context)
    repeated = infer(base_context)
    changed = infer(np.full((4, 7), 0.25, dtype=np.float32))
    shorter = infer(np.zeros((1, 7), dtype=np.float32))
    maximum = infer(np.zeros((9, 7), dtype=np.float32))
    empty = infer(np.zeros((0, 7), dtype=np.float32))
    padded = np.zeros((9, 7), dtype=np.float32)
    padded[4:] = 0.75
    masked = infer(padded, np.arange(9) < 4)
    report = dict(
        metadata=metadata,
        backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
        model_source=__import__('inspect').getfile(type(policy._model)),
        repeated_max_abs=float(np.max(np.abs(base-repeated))),
        context_value_max_abs=float(np.max(np.abs(base-changed))),
        context_length_max_abs=float(np.max(np.abs(base-shorter))),
        masked_padding_max_abs=float(np.max(np.abs(base-masked))),
        finite_empty_and_maximum=bool(np.isfinite(empty).all() and np.isfinite(maximum).all()),
        note='Synthetic fixed observations; verifies conditioning, not LIBERO success rate.',
    )
    np.testing.assert_allclose(base, repeated, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(base, masked, atol=1e-5, rtol=1e-5)
    if report['context_value_max_abs'] <= 1e-5 or report['context_length_max_abs'] <= 1e-5:
        raise AssertionError('Real Future output did not respond to context')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

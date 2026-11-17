"""Serve the existing unpruned after-context JAX checkpoint for adaptive timing."""
import argparse
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path

import jax

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config


class SeededPolicy:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, observation):
        observation = dict(observation)
        seed = observation.pop('_eval_seed', None)
        if seed is not None:
            self.policy._rng = jax.random.key(int(seed))
        return self.policy.infer(observation)

    def reset(self):
        self.policy.reset()


def load_future_policy(checkpoint, *, seed=7):
    """Verify files and require the complete training-time JAX parameter tree."""
    checkpoint = Path(checkpoint).resolve()
    manifest = json.loads((checkpoint/'checkpoint_manifest.json').read_text())
    if manifest.get('repo_id') != 'Dream-LIN/pi05_libero_async_future' or manifest.get('format') != 'Orbax JAX':
        raise ValueError('Require the original Future JAX checkpoint')
    for item in manifest['files']:
        p = (checkpoint/item['path']).resolve()
        if not p.is_relative_to(checkpoint) or p.stat().st_size != item['bytes']:
            raise ValueError(f'Invalid checkpoint file: {p}')
        digest = hashlib.sha256()
        with p.open('rb') as f:
            while block := f.read(8*1024*1024):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            raise ValueError(f'Checkpoint SHA256 mismatch: {p}')
    # Target shift is a training property. The existing context architecture has
    # the same inference inputs/parameter shapes; use the FUTURE weights above.
    cfg = config.get_config('pi05_libero_async_context_from_libero')
    cfg = dataclasses.replace(cfg, name='pi05_libero_async_future_time')
    policy = policy_config.create_trained_policy(
        cfg, checkpoint, sample_kwargs={'num_steps': 10}, strict_checkpoint=True,
    )
    if getattr(policy._model, 'action_context_horizon', None) != 9 or not hasattr(policy._model, 'action_context_in_proj'):
        raise ValueError('Future inference requires the trained JAX action-context projection')
    policy._rng = jax.random.key(seed)
    metadata = dict(config=cfg.name, action_target_alignment='after_context',
                    predicted_steps=10, context_horizon=9, output_skip_steps=0,
                    checkpoint=str(checkpoint), checkpoint_step=manifest['checkpoint_step'],
                    backend='jax', num_denoise_steps=10, timing_scheduler='client',
                    context_conditioning='jax_action_context_v1', strict_checkpoint=True)
    return policy, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=os.environ.get('TIME_CHECKPOINT', 'checkpoints/pi05_libero_async_future'))
    parser.add_argument('--port', type=int, default=19930)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    policy, metadata = load_future_policy(args.checkpoint, seed=args.seed)
    logging.info('TIME_SERVER_METADATA %s', json.dumps(metadata))
    WebsocketPolicyServer(SeededPolicy(policy), host='127.0.0.1', port=args.port,
                          metadata=metadata).serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    main()

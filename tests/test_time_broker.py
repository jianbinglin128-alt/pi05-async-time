import unittest

import numpy as np

from openpi_client.adaptive_k_scheduler import Decision
from openpi_client.adaptive_future_broker import AdaptiveFutureBroker


class ScriptedScheduler:
    executed_steps = 10
    candidates = (2, 3, 4)

    def __init__(self):
        self.index = 0

    def choose_for_next_chunk(self):
        k = (2, 4, 3, 2, 4, 3)[self.index % 6]
        self.index += 1
        return Decision(k, 10-k, False, True, .1, .05, .11, 0., 0.)

    def observe_request(self, duration):
        assert duration > 0

    def observe_action(self, duration):
        assert duration > 0


class ContinuationClient:
    def __init__(self):
        self.contexts = []

    def infer(self, request):
        start = request['at_step']
        context = request['action_context']
        np.testing.assert_array_equal(context[:, 0], np.arange(start, start+len(context)))
        self.contexts.append(len(context))
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:, 0] = np.arange(start+len(context), start+len(context)+10)
        return {'actions': actions}


class BrokerTests(unittest.TestCase):
    def test_real_thread_dynamic_context_alignment_and_terminal_drain(self):
        client = ContinuationClient()
        broker = AdaptiveFutureBroker(client, ScriptedScheduler())
        try:
            for step in range(49):
                action = broker.next_action(lambda: {'at_step': step})
                self.assertEqual(action[0], step)
                broker.observe_action(.01)
        finally:
            broker.close()
        self.assertEqual(client.contexts, [0, 2, 4, 3, 2, 4])
        self.assertEqual([d['k'] for d in broker.decisions], [2, 4, 3, 2, 4])
        self.assertTrue(broker.requests[-1]['terminal_prefetch'])
        self.assertEqual(len(broker.handovers), 4)
        self.assertTrue(all(r['output_skip_steps'] == 0 for r in broker.requests))
        self.assertTrue(all(h['wait_s'] >= 0 for h in broker.handovers))

    def test_fixed_comparator_preserves_context_alignment(self):
        client = ContinuationClient()
        broker = AdaptiveFutureBroker(client, ScriptedScheduler(), fixed_k=3)
        try:
            for step in range(35):
                self.assertEqual(broker.next_action(lambda: {'at_step': step})[0], step)
        finally:
            broker.close()
        self.assertEqual(client.contexts, [0, 3, 3, 3])
        self.assertTrue(all(d['k'] == 3 for d in broker.decisions))

    def test_invalid_model_output_is_rejected(self):
        class BadClient:
            def infer(self, obs):
                return {'actions': np.zeros((9, 7))}
        broker = AdaptiveFutureBroker(BadClient(), ScriptedScheduler())
        try:
            with self.assertRaises(ValueError):
                broker.next_action(lambda: {})
        finally:
            broker.close()

    def test_maximum_trained_context_preserves_action_order(self):
        class WideScheduler(ScriptedScheduler):
            candidates = tuple(range(2, 10))

            def choose_for_next_chunk(self):
                return Decision(9, 1, False, True, .3, .05, .32, 0., 0.)
        client = ContinuationClient()
        broker = AdaptiveFutureBroker(client, WideScheduler())
        try:
            for step in range(25):
                self.assertEqual(broker.next_action(lambda: {'at_step': step})[0], step)
                broker.observe_action(.01)
        finally:
            broker.close()
        self.assertEqual(client.contexts, [0, 9, 9, 9])


if __name__ == '__main__':
    unittest.main(verbosity=2)

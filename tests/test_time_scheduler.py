"""Deterministic CPU checks; these are not LIBERO performance experiments."""
import unittest

from openpi_client.adaptive_k_scheduler import AdaptiveKScheduler


def ready_scheduler(**overrides):
    options = dict(candidates=(2, 3, 4), initial_k=2, margin_s=0., margin_fraction=0., min_requests=1,
                   min_actions=1, decrease_chunks=1)
    options.update(overrides)
    return AdaptiveKScheduler(**options)


class SchedulerTests(unittest.TestCase):
    def test_execution_length_cannot_be_changed(self):
        for length in (2, 5, 8, 9, 11, True, 10.0):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, 'E=H=10'):
                    AdaptiveKScheduler(executed_steps=length)

    def test_cold_start_is_explicit(self):
        d = AdaptiveKScheduler().choose_for_next_chunk()
        self.assertEqual((d.k, d.trigger_after, d.cold_start), (4, 6, True))
        self.assertIsNone(d.feasible)

    def test_smallest_feasible_and_overload(self):
        for latency, expected, feasible in [(.09, 2, True), (.14, 3, True),
                                             (.19, 4, True), (.24, 4, False)]:
            s = ready_scheduler()
            s.observe_request(latency)
            s.observe_action(.05)
            d = s.choose_for_next_chunk()
            self.assertEqual((d.k, d.feasible), (expected, feasible))
            if not feasible:
                self.assertAlmostEqual(d.estimated_wait_s, .04)

    def test_slow_execution_needs_fewer_context_actions(self):
        s = ready_scheduler()
        s.observe_request(.14)
        s.observe_action(.08)
        self.assertEqual(s.choose_for_next_chunk().k, 2)

    def test_increase_immediate_decrease_after_three_chunks(self):
        s = ready_scheduler(initial_k=2, latency_window=1, decrease_chunks=3)
        s.observe_action(.05)
        s.observe_request(.19)
        self.assertEqual(s.choose_for_next_chunk().k, 4)
        s.observe_request(.09)
        self.assertEqual([s.choose_for_next_chunk().k for _ in range(6)],
                         [4, 4, 3, 3, 3, 2])

    def test_margin_can_make_a_boundary_infeasible(self):
        s = ready_scheduler(margin_s=.01)
        s.observe_request(.20)
        s.observe_action(.05)
        d = s.choose_for_next_chunk()
        self.assertFalse(d.feasible)
        self.assertAlmostEqual(d.estimated_wait_s, 0.)
        self.assertAlmostEqual(d.margin_deficit_s, .01)

    def test_fast_action_tail_is_conservative(self):
        s = ready_scheduler()
        s.observe_request(.14)
        for duration in [.03, .03, .05, .05, .05]:
            s.observe_action(duration)
        self.assertFalse(s.choose_for_next_chunk().feasible)

    def test_training_and_execution_bounds(self):
        for options in [dict(candidates=(-1, 2)), dict(candidates=(10,)),
                        dict(executed_steps=4), dict(context_horizon=3)]:
            with self.assertRaises(ValueError):
                AdaptiveKScheduler(**options)

    def test_reject_bad_measurements(self):
        s = AdaptiveKScheduler()
        for value in [0., -1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                s.observe_action(value)
            with self.assertRaises(ValueError):
                s.observe_request(value)

    def test_dynamic_context_alignment_early_and_late_returns(self):
        # Virtual clock and perfect continuation policy. Each integer represents
        # one action ID. Every new prediction starts AFTER the fixed context.
        s = ready_scheduler(latency_window=1)
        s.observe_action(.05)
        s.observe_request(.09)
        current = list(range(10))
        executed = []
        observed_k = []
        waits = []
        for latency in [.09, .14, .19, .24, .09, .09, .09]:
            d = s.choose_for_next_chunk()
            observed_k.append(d.k)
            executed.extend(current[:d.trigger_after])
            context = current[d.trigger_after:]
            self.assertEqual(len(context), d.k)
            next_actions = list(range(context[-1]+1, context[-1]+11))
            # Early result stays buffered until EVERY context action executes.
            executed.extend(context)
            waits.append(max(0., latency-len(context)*.05))
            s.observe_request(latency)
            current = next_actions
        self.assertEqual(executed, list(range(70)))
        self.assertGreater(len(set(observed_k)), 1)
        self.assertTrue(any(wait > 0 for wait in waits))
        self.assertTrue(any(wait == 0 for wait in waits))


    def test_wide_candidates_cover_longer_measured_requests(self):
        s = ready_scheduler(candidates=tuple(range(2,10)), latency_window=1)
        s.observe_action(.05)
        s.observe_request(.28)
        self.assertEqual(s.choose_for_next_chunk().k, 6)
        s.observe_request(.44)
        self.assertEqual(s.choose_for_next_chunk().k, 9)
        s.observe_request(.60)
        self.assertFalse(s.choose_for_next_chunk().feasible)


if __name__ == '__main__':
    unittest.main(verbosity=2)

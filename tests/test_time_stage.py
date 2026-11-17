"""Stage caps, real-thread zero-context handovers, and physical motion fixtures."""
import json
from pathlib import Path
import unittest

import numpy as np

from openpi_client.adaptive_future_broker import AdaptiveFutureBroker
from openpi_client.adaptive_k_scheduler import AdaptiveKScheduler
from openpi_client.stage_k_cap import StageConfig, StageKCap


def config(**overrides):
    path = Path(__file__).resolve().parents[1]/'configs/stage_target20.json'
    options = dict(fine_max_k=0, coarse_max_k=4)
    options.update(overrides)
    return StageConfig.from_mapping(json.loads(path.read_text()), **options)


def moving_stage(**overrides):
    stage = StageKCap(config(**overrides))
    for i in range(11):
        t = i*.05
        stage.observe([.1*t, 0., 0.], t, -1.)
    return stage


def open_actions(steps=10):
    actions = np.zeros((steps, 7), dtype=np.float32)
    actions[:, 6] = -1.
    return actions


def scheduler(**overrides):
    options = dict(candidates=tuple(range(10)), initial_k=9, min_requests=1, min_actions=1,
                   margin_s=0., margin_fraction=0., decrease_chunks=3)
    options.update(overrides)
    result = AdaptiveKScheduler(**options)
    result.observe_request(.25)
    result.observe_action(.05)
    return result


class StageTests(unittest.TestCase):
    def test_stage_defaults_and_execution_context_limits(self):
        cfg = config(fine_max_k=None, coarse_max_k=None)
        self.assertEqual((cfg.requested_fine_max_k, cfg.requested_coarse_max_k), (3, 7))
        self.assertEqual((cfg.fine_max_k, cfg.coarse_max_k, cfg.hard_max_k), (3, 7, 9))
        for e in (2, 5, 7, 9, 11):
            with self.subTest(e=e):
                with self.assertRaisesRegex(ValueError, 'E=H=10'):
                    config(executed_steps=e)
        cfg = config(executed_steps=10, context_horizon=3, fine_max_k=None, coarse_max_k=None)
        self.assertEqual((cfg.fine_max_k, cfg.coarse_max_k), (3, 3))

    def test_requested_and_effective_caps_are_logged(self):
        stage = StageKCap(config(fine_max_k=None, coarse_max_k=None))
        decision = stage.choose_for_chunk(open_actions())
        self.assertEqual((decision['requested_max_k'], decision['max_k'], decision['hard_max_k']), (3, 3, 9))
        for i in range(11):
            t = i*.05
            stage.observe([.1*t, 0., 0.], t, -1.)
        decision = stage.choose_for_chunk(open_actions())
        self.assertEqual((decision['requested_max_k'], decision['max_k'], decision['hard_max_k']), (7, 7, 9))

    def test_history_missing_is_fine_and_steady_translation_is_coarse(self):
        self.assertEqual(StageKCap(config()).choose_for_chunk(open_actions())['max_k'], 0)
        decision = moving_stage().choose_for_chunk(open_actions())
        self.assertEqual((decision['phase'], decision['max_k']), ('coarse', 4))
        self.assertAlmostEqual(decision['features']['speed_m_s'], .1)
        self.assertAlmostEqual(decision['features']['change_m_s2'], 0.)

    def test_low_speed_and_large_motion_change_enter_fine(self):
        stopped = StageKCap(config())
        accelerating = StageKCap(config())
        for i in range(11):
            t = i*.05
            stopped.observe([0., 0., 0.], t, -1.)
            x = .1*t if t <= .25 else .025+.5*(t-.25)
            accelerating.observe([x, 0., 0.], t, -1.)
        self.assertIn('low_speed', stopped.choose_for_chunk(open_actions())['reasons'])
        self.assertIn('high_motion_change', accelerating.choose_for_chunk(open_actions())['reasons'])

    def test_upcoming_gripper_and_control_time_hold(self):
        stage = moving_stage(gripper_lookahead_steps=10)
        actions = open_actions()
        actions[7:, 6] = 1.
        self.assertIn('upcoming_gripper_event', stage.choose_for_chunk(actions)['reasons'])
        stage.observe([.055, 0., 0.], .55, 1.)
        closed = open_actions(); closed[:, 6] = 1.
        self.assertIn('gripper_hold', stage.choose_for_chunk(closed)['reasons'])
        for i in range(12, 16):
            t = i*.05
            stage.observe([.1*t, 0., 0.], t, 1.)
        self.assertEqual(stage.choose_for_chunk(closed)['phase'], 'coarse')

    def test_deadband_and_slow_recovery(self):
        stage = moving_stage(stable_chunks=2)
        actions = open_actions(); actions[:, 6] = .1
        self.assertEqual(stage.choose_for_chunk(actions)['phase'], 'fine')
        self.assertEqual(stage.choose_for_chunk(actions)['phase'], 'coarse')
        self.assertFalse(stage.choose_for_chunk(actions)['features']['upcoming_gripper_event'])

    def test_invalid_inputs_and_per_episode_reset(self):
        stage = moving_stage()
        self.assertEqual(stage.choose_for_chunk(open_actions())['phase'], 'coarse')
        with self.assertRaises(ValueError):
            stage.observe([0., 0., 0.], .5, -1.)
        with self.assertRaises(ValueError):
            stage.observe([float('nan'), 0., 0.], .6, -1.)
        with self.assertRaises(ValueError):
            config(fine_max_k=5, coarse_max_k=4)
        self.assertEqual(StageKCap(config()).choose_for_chunk(open_actions())['phase'], 'fine')


class CapSchedulerTests(unittest.TestCase):
    def test_cap_drop_overrides_cold_start_and_decrease_hysteresis(self):
        cold = AdaptiveKScheduler(candidates=tuple(range(10)), initial_k=9)
        self.assertEqual(cold.choose_for_next_chunk(max_k=0).k, 0)
        measured = scheduler()
        decision = measured.choose_for_next_chunk(max_k=3)
        self.assertEqual(decision.k, 3)
        self.assertFalse(decision.feasible)
        self.assertTrue(decision.cap_limited)
        self.assertEqual(decision.timing_min_k, 5)
        self.assertEqual(measured.choose_for_next_chunk(max_k=0).k, 0)
        self.assertEqual(measured.choose_for_next_chunk(max_k=4).k, 4)

    def test_cap_does_not_override_smallest_feasible_choice(self):
        result = scheduler(initial_k=0, latency_window=1)
        result.observe_request(.09)
        decision = result.choose_for_next_chunk(max_k=4)
        self.assertEqual(decision.k, 2)
        self.assertTrue(decision.feasible)

    def test_no_candidate_below_cap_is_an_error(self):
        with self.assertRaises(ValueError):
            AdaptiveKScheduler().choose_for_next_chunk(max_k=0)


class RecordingClient:
    def __init__(self):
        self.contexts = []

    def infer(self, request):
        start = request['at_step']
        context = request['action_context']
        np.testing.assert_array_equal(context[:, 0], np.arange(start, start+len(context)))
        self.contexts.append(len(context))
        actions = open_actions()
        actions[:, 0] = np.arange(start+len(context), start+len(context)+10)
        return {'actions': actions}


class BrokerStageTests(unittest.TestCase):
    def test_zero_k_waits_until_boundary_and_preserves_action_order(self):
        client = RecordingClient()
        broker = AdaptiveFutureBroker(client, scheduler(candidates=(0,), initial_k=0))
        try:
            for step in range(35):
                self.assertEqual(broker.next_action(lambda: {'at_step': step})[0], step)
                broker.observe_action(.01)
                if step == 9:
                    self.assertEqual(client.contexts, [0])
            self.assertEqual(client.contexts, [0, 0, 0, 0])
            self.assertTrue(all(h['cover_window_s'] == 0. for h in broker.handovers))
        finally:
            broker.close()

    def test_phase_caps_change_only_at_install_and_context_is_never_replaced(self):
        class ChangingStage:
            max_k = 0
            def choose_for_chunk(self, actions):
                return dict(phase='fine' if self.max_k == 0 else 'coarse', max_k=self.max_k)
        stage = ChangingStage()
        client = RecordingClient()
        broker = AdaptiveFutureBroker(client, scheduler(), stage_controller=stage)
        try:
            for step in range(45):
                if step == 4:
                    stage.max_k = 4
                if step == 17:
                    stage.max_k = 0
                self.assertEqual(broker.next_action(lambda: {'at_step': step})[0], step)
                broker.observe_action(.01)
            self.assertEqual([d['k'] for d in broker.decisions], [0, 4, 0, 0, 0])
            self.assertEqual(client.contexts, [0, 0, 4, 0, 0])
        finally:
            broker.close()

    def test_fixed_k_cannot_silently_ignore_stage(self):
        with self.assertRaises(ValueError):
            AdaptiveFutureBroker(RecordingClient(), scheduler(), fixed_k=4, stage_controller=moving_stage())


if __name__ == '__main__':
    unittest.main()

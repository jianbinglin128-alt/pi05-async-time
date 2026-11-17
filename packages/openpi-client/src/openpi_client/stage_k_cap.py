"""Motion-based coarse/fine classifier adapted from openpi_async_stage.

Motion features use simulation/control timestamps, never inference wall time.
Call observe after executed actions and choose_for_chunk once per installed chunk.
The labels are motion heuristics, not semantic task-stage ground truth.
"""
from collections import deque
from dataclasses import dataclass, field, fields
import math

import numpy as np


@dataclass(frozen=True)
class StageConfig:
    speed_enter_fine_m_s: float
    speed_exit_fine_m_s: float
    change_enter_fine_m_s2: float
    change_exit_fine_m_s2: float
    gripper_hold_s: float
    stable_chunks: int
    history_window_steps: int
    gripper_command_deadband: float
    executed_steps: int = 10
    context_horizon: int = 9
    fine_max_k: int | None = None
    coarse_max_k: int | None = None
    gripper_lookahead_steps: int = 5
    requested_fine_max_k: int = field(init=False)
    requested_coarse_max_k: int = field(init=False)
    hard_max_k: int = field(init=False)

    @classmethod
    def from_mapping(cls, data, **overrides):
        known = {f.name for f in fields(cls) if f.init}
        return cls(**dict({k: v for k, v in data.items() if k in known}, **overrides))

    def __post_init__(self):
        if type(self.executed_steps) is not int or not 2 <= self.executed_steps <= 10:
            raise ValueError('executed_steps must be an integer in 2..10')
        if type(self.context_horizon) is not int or not 0 <= self.context_horizon <= 9:
            raise ValueError('context_horizon must be an integer in 0..9 for the current checkpoint')
        fine = self.executed_steps//2 if self.fine_max_k is None else self.fine_max_k
        coarse = 2*self.executed_steps if self.coarse_max_k is None else self.coarse_max_k
        if any(type(k) is not int for k in (fine, coarse)) or not 0 <= fine <= coarse:
            raise ValueError('Require integer requested caps 0 <= fine_max_k <= coarse_max_k')
        hard = min(self.executed_steps-1, self.context_horizon)
        object.__setattr__(self, 'requested_fine_max_k', fine)
        object.__setattr__(self, 'requested_coarse_max_k', coarse)
        object.__setattr__(self, 'hard_max_k', hard)
        object.__setattr__(self, 'fine_max_k', min(fine, hard))
        object.__setattr__(self, 'coarse_max_k', min(coarse, hard))
        for name in ('speed_enter_fine_m_s', 'speed_exit_fine_m_s', 'change_enter_fine_m_s2',
                     'change_exit_fine_m_s2', 'gripper_hold_s', 'gripper_command_deadband'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('stable_chunks', 'history_window_steps', 'gripper_lookahead_steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.speed_exit_fine_m_s < self.speed_enter_fine_m_s or self.change_exit_fine_m_s2 > self.change_enter_fine_m_s2:
            raise ValueError('Invalid stage hysteresis thresholds')


class StageKCap:
    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=2*config.history_window_steps+1)
        self.fine = True
        self.stable = 0
        self.gripper_sign = -1
        self.last_gripper_event_time = -1e9

    def _sign(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Gripper command must be finite')
        deadband = self.config.gripper_command_deadband
        return 1 if value > deadband else -1 if value < -deadband else 0

    def observe(self, position, control_time_s, gripper_command):
        position = np.asarray(position, dtype=np.float64)
        control_time_s = float(control_time_s)
        if position.shape != (3,) or not np.isfinite(position).all() or not math.isfinite(control_time_s):
            raise ValueError('Require a finite xyz position and control timestamp')
        if self.history and control_time_s <= self.history[-1]['sim_time']:
            raise ValueError('Control timestamps must increase within an episode')
        new_sign = self._sign(gripper_command)
        if new_sign and self.gripper_sign and new_sign != self.gripper_sign:
            self.last_gripper_event_time = control_time_s
        if new_sign:
            self.gripper_sign = new_sign
        self.history.append(dict(position=position.copy(), sim_time=control_time_s))

    def choose_for_chunk(self, actions):
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError('Stage lookahead requires finite [steps, 7] actions')
        cfg = self.config
        n = min(cfg.history_window_steps, (len(self.history)-1)//2) if self.history else 0
        speed = change = 0.0
        if n > 0:
            now, prev, prev2 = self.history[-1], self.history[-1-n], self.history[-1-2*n]
            dt = now['sim_time']-prev['sim_time']
            before_dt = prev['sim_time']-prev2['sim_time']
            vector = (now['position']-prev['position'])/dt
            previous = (prev['position']-prev2['position'])/before_dt
            speed = float(np.linalg.norm(vector))
            change = float(np.linalg.norm(vector-previous)/dt)
        grip = self.gripper_sign
        upcoming = False
        for action in actions[:cfg.gripper_lookahead_steps]:
            new = self._sign(action[6])
            upcoming = upcoming or bool(new and grip and new != grip)
            if new:
                grip = new
        now_time = self.history[-1]['sim_time'] if self.history else None
        protection = now_time is not None and now_time-self.last_gripper_event_time < cfg.gripper_hold_s
        valid = n == cfg.history_window_steps
        reasons = []
        for condition, label in (
            (not valid, 'insufficient_history'), (upcoming, 'upcoming_gripper_event'),
            (protection, 'gripper_hold'), (speed <= cfg.speed_enter_fine_m_s, 'low_speed'),
            (change >= cfg.change_enter_fine_m_s2, 'high_motion_change'),
        ):
            if condition:
                reasons.append(label)
        stable = not reasons and speed >= cfg.speed_exit_fine_m_s and change <= cfg.change_exit_fine_m_s2
        if reasons:
            self.fine, self.stable = True, 0
        elif self.fine:
            self.stable = self.stable+1 if stable else 0
            if self.stable >= cfg.stable_chunks:
                self.fine = False
        phase = 'fine' if self.fine else 'coarse'
        return dict(phase=phase, max_k=cfg.fine_max_k if self.fine else cfg.coarse_max_k,
                    requested_max_k=cfg.requested_fine_max_k if self.fine else cfg.requested_coarse_max_k,
                    hard_max_k=cfg.hard_max_k,
                    reasons=reasons, stable_chunks=self.stable,
                    features=dict(speed_m_s=speed, change_m_s2=change, history_valid=valid,
                                  upcoming_gripper_event=upcoming, gripper_protection=protection,
                                  feature_control_time_s=now_time,
                                  last_gripper_event_time=self.last_gripper_event_time))

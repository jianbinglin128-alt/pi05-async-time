"""Timing-only ActSync scheduler prototype; no robot/model dependencies.

Call choose_for_next_chunk() exactly once when installing each new chunk.
Keep its k frozen until that chunk is consumed. All times are wall-clock seconds.
Feed latency from request preparation/submission through validated action readiness,
and action durations without inference waits, resets, logging, or terminal drains.
This module does not change the policy, execution length, or context alignment.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from openpi_client.time_protocol import EXECUTED_STEPS


def _quantile(values, probability):
    """Empirical nearest-rank quantile (no external dependency)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Duration must be positive and finite")
    return value


@dataclass(frozen=True)
class Decision:
    k: int
    trigger_after: int
    cold_start: bool
    feasible: bool | None
    latency_s: float | None
    action_s: float | None
    budget_s: float | None
    estimated_wait_s: float | None
    margin_deficit_s: float | None
    max_k: int | None = None
    cap_limited: bool = False
    timing_min_k: int | None = None


class AdaptiveKScheduler:
    """Smallest timing-feasible k, with immediate increases and slow decreases.

    Defaults are starting hypotheses for evaluation, not calibrated robot settings.
    Candidate lengths must be supported by the checkpoint AND execution protocol.
    Estimators belong to one device/backend/control mode; reset on mode changes.
    Use one instance across episodes on the same device, outside the rollout loop.
    """

    def __init__(self, *, executed_steps=EXECUTED_STEPS, candidates=(2, 3, 4, 5, 6, 7, 8, 9),
                 context_horizon=9, initial_k=4, latency_window=32,
                 action_window=128, latency_quantile=0.95,
                 action_quantile=0.20, margin_s=0.002, margin_fraction=0.05,
                 min_requests=3, min_actions=8, decrease_chunks=3):
        if type(executed_steps) is not int or executed_steps != EXECUTED_STEPS:
            raise ValueError("Future timing runtime requires E=H=10 action steps")
        candidates = tuple(sorted(set(candidates)))
        if not candidates or any(not isinstance(k, int) or
                                 not 0 <= k <= min(context_horizon, executed_steps-1)
                                 for k in candidates):
            raise ValueError("Candidates must satisfy 0 <= k <= min(context_horizon, E-1)")
        if initial_k not in candidates:
            raise ValueError("initial_k must be a candidate")
        if any(not isinstance(n, int) or n < 1 for n in
               (latency_window, action_window, min_requests, min_actions, decrease_chunks)):
            raise ValueError("Window sizes and sample counts must be positive integers")
        if min_requests > latency_window or min_actions > action_window:
            raise ValueError("Minimum sample counts must fit their windows")
        if not 0 < latency_quantile <= 1 or not 0 < action_quantile <= 1:
            raise ValueError("Quantiles must be in (0, 1]")
        if not all(math.isfinite(x) and x >= 0 for x in (margin_s, margin_fraction)):
            raise ValueError("Margins must be finite and nonnegative")
        self.executed_steps = executed_steps
        self.candidates = candidates
        self.k = initial_k
        self.latencies = deque(maxlen=latency_window)
        self.actions = deque(maxlen=action_window)
        self.latency_quantile = latency_quantile
        self.action_quantile = action_quantile
        self.margin_s = margin_s
        self.margin_fraction = margin_fraction
        self.min_requests = min_requests
        self.min_actions = min_actions
        self.decrease_chunks = decrease_chunks
        self._lower_streak = 0

    def observe_request(self, duration_s):
        # Exclude JIT/model-loading warmup; include worker queue and network time.
        self.latencies.append(_positive(duration_s))

    def observe_action(self, duration_s):
        # env.step or command slot duration, never chunk-boundary waiting.
        self.actions.append(_positive(duration_s))

    def choose_for_next_chunk(self, *, max_k=None):
        if max_k is not None and (not isinstance(max_k, int) or isinstance(max_k, bool) or max_k < 0):
            raise ValueError("max_k must be a nonnegative integer")
        allowed = tuple(k for k in self.candidates if max_k is None or k <= max_k)
        if not allowed:
            raise ValueError("No scheduler candidate satisfies max_k")
        previous_k = self.k
        self.k = min(self.k, allowed[-1])
        clamped = self.k != previous_k
        if clamped:
            self._lower_streak = 0
        if len(self.latencies) < self.min_requests or len(self.actions) < self.min_actions:
            self._lower_streak = 0
            return Decision(self.k, self.executed_steps-self.k, True,
                            None, None, None, None, None, None,
                            max_k=max_k, cap_limited=clamped)
        latency = _quantile(self.latencies, self.latency_quantile)
        action = _quantile(self.actions, self.action_quantile)
        budget = latency + max(self.margin_s, self.margin_fraction*latency)
        feasible = [k for k in allowed if k*action >= budget]
        globally_feasible = [k for k in self.candidates if k*action >= budget]
        timing_min_k = globally_feasible[0] if globally_feasible else None
        wanted = feasible[0] if feasible else allowed[-1]
        if wanted > self.k:
            self.k = wanted
            self._lower_streak = 0
        elif wanted < self.k:
            self._lower_streak += 1
            if self._lower_streak >= self.decrease_chunks:
                self.k = allowed[allowed.index(self.k)-1]
                self._lower_streak = 0
        else:
            self._lower_streak = 0
        return Decision(
            self.k, self.executed_steps-self.k, False, bool(feasible),
            latency, action, budget, max(0., latency-self.k*action),
            max(0., budget-self.k*action), max_k=max_k,
            cap_limited=clamped or (max_k is not None and not feasible and allowed[-1] < self.candidates[-1]),
            timing_min_k=timing_min_k)

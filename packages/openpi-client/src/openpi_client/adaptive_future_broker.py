"""After-context asynchronous broker with a per-chunk timing decision."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import time

import numpy as np

from openpi_client.time_protocol import EXECUTED_STEPS, PREDICTED_STEPS


class AdaptiveFutureBroker:
    """One request in flight; results never replace their committed context."""

    def __init__(self, client, scheduler, *, fixed_k=None, predicted_steps=PREDICTED_STEPS, stage_controller=None):
        self.client = client
        self.scheduler = scheduler
        self.e = scheduler.executed_steps
        if type(self.e) is not int or self.e != EXECUTED_STEPS or type(predicted_steps) is not int or predicted_steps != PREDICTED_STEPS:
            raise ValueError('Future timing runtime requires E=H=10 action steps')
        if fixed_k is not None and fixed_k not in scheduler.candidates:
            raise ValueError('Fixed k must be a scheduler candidate')
        self.fixed_k = fixed_k
        if fixed_k is not None and stage_controller is not None:
            raise ValueError('Fixed-k comparison must not use stage caps')
        self.stage_controller = stage_controller
        self.predicted_steps = predicted_steps
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='time-inference')
        self.current = None
        self.index = 0
        self.pending = None
        self.decision = None
        self.requests = []
        self.decisions = []
        self.handovers = []
        self.total_actions = 0
        self.chunk = -1
        self.last_action_end_ns = None

    def _request(self, request, started_ns, context_steps, at_step):
        worker_start = time.perf_counter_ns()
        result = self.client.infer(request)
        actions = np.asarray(result['actions'])
        if actions.shape != (self.predicted_steps, 7) or not np.isfinite(actions).all():
            raise ValueError(f'Expected finite ({self.predicted_steps},7) actions, got {actions.shape}')
        actions = actions.copy()
        ready_ns = time.perf_counter_ns()
        record = dict(start_ns=started_ns, worker_start_ns=worker_start, ready_ns=ready_ns,
                      request_s=(ready_ns-started_ns)/1e9, context_steps=context_steps,
                      at_step=at_step, output_skip_steps=0, terminal_prefetch=False)
        return actions, record

    def _install(self, actions):
        self.current = actions
        self.index = 0
        self.chunk += 1
        started = time.perf_counter_ns()
        stage = self.stage_controller.choose_for_chunk(actions) if self.stage_controller is not None else None
        choice = self.scheduler.choose_for_next_chunk(max_k=stage['max_k']) if stage is not None else self.scheduler.choose_for_next_chunk()
        self.decision = asdict(choice)
        self.decision['scheduler_ns'] = time.perf_counter_ns()-started
        if stage is not None:
            self.decision['stage'] = stage
        if self.fixed_k is not None:
            self.decision.update(k=self.fixed_k, trigger_after=self.e-self.fixed_k)
            if self.decision['action_s'] is not None:
                covered_s = self.fixed_k*self.decision['action_s']
                self.decision.update(
                    feasible=covered_s >= self.decision['budget_s'],
                    estimated_wait_s=max(0., self.decision['latency_s']-covered_s),
                    margin_deficit_s=max(0., self.decision['budget_s']-covered_s))

        self.decision.update(chunk=self.chunk, at_step=self.total_actions, mode='adaptive' if self.fixed_k is None else 'fixed')
        self.decisions.append(self.decision.copy())

    def next_action(self, observation_factory):
        if self.current is None:
            started = time.perf_counter_ns()
            request = observation_factory()
            request['action_context'] = np.zeros((0, 7), dtype=np.float32)
            actions, record = self._request(request, started, 0, self.total_actions)
            record['bootstrap'] = True
            self.requests.append(record)
            self.scheduler.observe_request(record['request_s'])
            self._install(actions)
        elif self.index == self.e:
            if self.pending is None:
                if self.decision['k'] != 0:
                    raise RuntimeError('No future request at chunk boundary')
                # k=0 takes a fresh observation only after the old chunk finishes.
                started = time.perf_counter_ns()
                request = observation_factory()
                request['action_context'] = np.zeros((0, 7), dtype=np.float32)
                self.pending = self.executor.submit(self._request, request, started, 0, self.total_actions)
            boundary_ns = self.last_action_end_ns or time.perf_counter_ns()
            actions, record = self.pending.result()
            self.pending = None
            self.requests.append(record)
            self.scheduler.observe_request(record['request_s'])
            self.handovers.append(dict(
                chunk=self.chunk, at_step=self.total_actions, k=record['context_steps'],
                stage=self.decision.get('stage', {}).get('phase'),
                max_k=self.decision.get('max_k'),
                boundary_ns=boundary_ns, ready_ns=record['ready_ns'],
                covered=record['ready_ns'] <= boundary_ns,
                wait_s=max(0., (record['ready_ns']-boundary_ns)/1e9),
                cover_window_s=max(0., (boundary_ns-record['start_ns'])/1e9),
                request_after_boundary_s=max(0., (record['start_ns']-boundary_ns)/1e9)))
            self._install(actions)
        if self.index == self.decision['trigger_after']:
            if self.pending is not None:
                raise RuntimeError('Duplicate in-flight request')
            context = self.current[self.index:].copy()
            if len(context) != self.decision['k']:
                raise RuntimeError('Context does not match frozen chunk decision')
            started = time.perf_counter_ns()
            request = observation_factory()
            request['action_context'] = context
            self.pending = self.executor.submit(self._request, request, started, len(context), self.total_actions)
        action = self.current[self.index].copy()
        self.index += 1
        self.total_actions += 1
        return action

    def observe_action(self, duration_s, *, completed_ns=None):
        self.scheduler.observe_action(duration_s)
        self.last_action_end_ns = completed_ns if completed_ns is not None else time.perf_counter_ns()

    def close(self):
        try:
            if self.pending is not None:
                _, record = self.pending.result()
                record['terminal_prefetch'] = True
                self.requests.append(record)
                self.pending = None
        finally:
            self.executor.shutdown(wait=True)

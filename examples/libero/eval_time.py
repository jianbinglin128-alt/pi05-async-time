"""Bounded LIBERO validation of adaptive k with unchanged future weights."""
import argparse
from collections import Counter
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import time

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools, websocket_client_policy
from openpi_client.adaptive_k_scheduler import AdaptiveKScheduler
from openpi_client.adaptive_future_broker import AdaptiveFutureBroker

MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)


def observation(obs, prompt):
    result = {'prompt': str(prompt)}
    for source, target in [('agentview_image', 'observation/image'), ('robot0_eye_in_hand_image', 'observation/wrist_image')]:
        image = np.ascontiguousarray(obs[source][::-1, ::-1])
        result[target] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    q = np.asarray(obs['robot0_eef_quat']).copy()
    q[3] = np.clip(q[3], -1., 1.)
    den = np.sqrt(1.-q[3]*q[3])
    axis = np.zeros(3) if np.isclose(den, 0.) else q[:3]*2.*np.arccos(q[3])/den
    result['observation/state'] = np.concatenate((obs['robot0_eef_pos'], axis, obs['robot0_gripper_qpos']))
    return result


def dump_rows(folder, filename, rows, tags):
    with (folder/filename).open('a') as f:
        for row in rows:
            f.write(json.dumps(dict(row, **tags))+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=19930)
    parser.add_argument('--suites', nargs='+', choices=list(MAX_STEPS), default=list(MAX_STEPS))
    parser.add_argument('--tasks', type=int, default=1)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--execute-steps', type=int, default=10)
    parser.add_argument('--candidates', nargs='+', type=int, default=[2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--initial-k', type=int, default=4)
    parser.add_argument('--fixed-k', type=int)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.tasks <= 10 or not 1 <= args.trials <= 50 or not 2 <= args.execute_steps <= 10:
        parser.error('Require tasks 1..10, trials 1..50, E 2..10')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    scheduler = AdaptiveKScheduler(executed_steps=args.execute_steps, candidates=tuple(args.candidates), initial_k=args.initial_k)
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    meta = client.get_server_metadata()
    if meta.get('action_target_alignment') != 'after_context' or meta.get('context_horizon') != 9:
        raise ValueError(f'Wrong model metadata: {meta}')
    (out/'run.json').write_text(json.dumps(dict(args=vars(args), metadata=meta,
        imports=dict(scheduler=inspect.getfile(AdaptiveKScheduler), broker=inspect.getfile(AdaptiveFutureBroker)),
        timing=dict(episode='after reset/10 settling steps to final env.step return, includes bootstrap',
                    request='before preparing trigger observation to validated output ready, includes worker queue',
                    action='env.step only; excludes waits, preprocessing and logging',
                    handover='max(0, ready_ns-boundary_ns), zero when result already ready',
                    excluded='JIT warmup, reset, settling, terminal drain, disk writes'),
        limitations='small functional validation under shared GPU load; no formal performance claim'), indent=2)+'\n')
    episodes = []
    warmed = False
    try:
        for suite_name in args.suites:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            for task_id in range(args.tasks):
                task = suite.get_task(task_id)
                states = suite.get_task_init_states(task_id)
                bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
                env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
                env.seed(args.seed)
                try:
                    for trial in range(args.trials):
                        env.reset()
                        obs = env.set_init_state(states[trial])
                        for _ in range(10):
                            obs, _, _, _ = env.step([0.]*6+[-1.])
                        client.reset()
                        if not warmed:
                            warm = []
                            for index in range(3):
                                request = observation(obs, task.language)
                                request['action_context'] = np.zeros((0, 7), dtype=np.float32)
                                before = time.perf_counter_ns()
                                result = client.infer(request)
                                warm.append(dict(index=index, duration_s=(time.perf_counter_ns()-before)/1e9))
                                assert np.asarray(result['actions']).shape == (10, 7)
                            (out/'warmup.json').write_text(json.dumps(warm, indent=2)+'\n')
                            warmed = True
                        # Reset policy noise per episode, independent of previous tests.
                        first = True
                        def make_observation():
                            nonlocal first
                            request = observation(obs, task.language)
                            if first:
                                request['_eval_seed'] = args.seed + trial
                                first = False
                            return request
                        broker = AdaptiveFutureBroker(client, scheduler, fixed_k=args.fixed_k)
                        actions = []
                        done = False
                        error = None
                        started = time.perf_counter_ns()
                        finished = None
                        try:
                            for step in range(MAX_STEPS[suite_name]):
                                action = broker.next_action(make_observation).tolist()
                                before = time.perf_counter_ns()
                                obs, _, done, _ = env.step(action)
                                after = time.perf_counter_ns()
                                broker.observe_action((after-before)/1e9, completed_ns=after)
                                actions.append(dict(step=step, start_ns=before, end_ns=after, duration_s=(after-before)/1e9))
                                if done:
                                    break
                            finished = after
                        except Exception as exc:
                            error = repr(exc)
                            finished = time.perf_counter_ns()
                            raise
                        finally:
                            finished = finished or time.perf_counter_ns()
                            broker.close()
                            tags = dict(suite=suite_name, task_id=task_id, trial=trial)
                            entry = dict(**tags, task=task.language, success=bool(done and error is None),
                                error=error, steps=len(actions), rollout_wall_s=(finished-started)/1e9,
                                handovers=len(broker.handovers), covered=sum(h['covered'] for h in broker.handovers),
                                wait_s=sum(h['wait_s'] for h in broker.handovers),
                                k_counts=dict(Counter(d['k'] for d in broker.decisions)),
                                infeasible_chunks=sum(d['feasible'] is False for d in broker.decisions),
                                cold_start_chunks=sum(d['cold_start'] for d in broker.decisions))
                            episodes.append(entry)
                            for filename, rows in [('episodes.jsonl', [entry]), ('actions.jsonl', actions),
                                                   ('requests.jsonl', broker.requests), ('decisions.jsonl', broker.decisions),
                                                   ('handovers.jsonl', broker.handovers)]:
                                dump_rows(out, filename, rows, tags)
                            print('EPISODE', json.dumps(entry), flush=True)
                finally:
                    env.close()
        waits = []
        for line in (out/'handovers.jsonl').read_text().splitlines():
            waits.append(json.loads(line)['wait_s'])
        decisions = [json.loads(line) for line in (out/'decisions.jsonl').read_text().splitlines()]
        summary = dict(episodes=len(episodes), successes=sum(e['success'] for e in episodes),
                       success_rate=sum(e['success'] for e in episodes)/len(episodes),
                       mean_success_wall_s=float(np.mean([e['rollout_wall_s'] for e in episodes if e['success']])) if any(e['success'] for e in episodes) else None,
                       coverage=sum(e['covered'] for e in episodes)/sum(e['handovers'] for e in episodes) if waits else None,
                       mean_wait_ms=1000*float(np.mean(waits)) if waits else None,
                       p95_wait_ms=1000*float(np.quantile(waits, .95)) if waits else None,
                       k_counts=dict(Counter(d['k'] for d in decisions)),
                       infeasible_chunks=sum(d['feasible'] is False for d in decisions),
                       mean_scheduler_us=float(np.mean([d['scheduler_ns'] for d in decisions]))/1000,
                       suites={s:dict(episodes=sum(e['suite']==s for e in episodes), successes=sum(e['success'] for e in episodes if e['suite']==s)) for s in args.suites})
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print('SUMMARY', json.dumps(summary), flush=True)
    finally:
        client._ws.close()


if __name__ == '__main__':
    main()

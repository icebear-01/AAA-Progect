from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Dict, Iterable, List, Sequence

from rl_env import SLPathEnv, StepResult
from sl_grid import GridSpec


def _worker_loop(
    conn: Connection,
    grid_spec_data: Dict[str, object],
    env_kwargs: Dict[str, object],
) -> None:
    env = SLPathEnv(GridSpec(**grid_spec_data), **env_kwargs)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                conn.send(env.reset())
            elif cmd == "reset_to_scenario_index":
                conn.send(env.reset_to_scenario_index(int(payload)))
            elif cmd == "step":
                conn.send(env.step(int(payload)))
            elif cmd == "close":
                conn.close()
                break
            else:
                raise RuntimeError(f"Unknown vector-env command: {cmd}")
    except EOFError:
        pass


class SyncVectorEnv:
    def __init__(
        self,
        primary_env: SLPathEnv,
        grid_spec: GridSpec,
        env_kwargs_list: Sequence[Dict[str, object]],
    ) -> None:
        self.grid_spec = grid_spec
        self.envs: List[SLPathEnv] = [primary_env]
        for kwargs in env_kwargs_list[1:]:
            self.envs.append(SLPathEnv(self.grid_spec, **kwargs))
        self.num_envs = len(self.envs)

    def reset(self) -> List[Dict[str, object]]:
        return [env.reset() for env in self.envs]

    def reset_indices(self, indices: Sequence[int]) -> List[Dict[str, object]]:
        return [self.envs[int(idx)].reset() for idx in indices]

    def reset_to_scenario_indices(
        self,
        scenario_indices: Sequence[int],
        *,
        env_indices: Sequence[int] | None = None,
    ) -> List[Dict[str, object]]:
        if env_indices is None:
            env_indices = list(range(len(scenario_indices)))
        if len(env_indices) != len(scenario_indices):
            raise ValueError("env_indices and scenario_indices must have the same length.")
        return [
            self.envs[int(env_idx)].reset_to_scenario_index(int(scenario_idx))
            for env_idx, scenario_idx in zip(env_indices, scenario_indices)
        ]

    def step(self, actions: Sequence[int]) -> List[StepResult]:
        return [env.step(int(action)) for env, action in zip(self.envs, actions)]

    def step_indices(self, indices: Sequence[int], actions: Sequence[int]) -> List[StepResult]:
        if len(indices) != len(actions):
            raise ValueError("indices and actions must have the same length.")
        return [
            self.envs[int(idx)].step(int(action))
            for idx, action in zip(indices, actions)
        ]

    def close(self) -> None:
        return None


class SubprocVectorEnv:
    def __init__(
        self,
        grid_spec: GridSpec,
        env_kwargs_list: Sequence[Dict[str, object]],
        *,
        start_method: str = "fork",
    ) -> None:
        ctx = mp.get_context(start_method)
        self.grid_spec = grid_spec
        self.num_envs = len(env_kwargs_list)
        self._parent_conns: List[Connection] = []
        self._processes: List[mp.Process] = []
        grid_spec_data = {
            "s_range": tuple(grid_spec.s_range),
            "l_range": tuple(grid_spec.l_range),
            "s_samples": int(grid_spec.s_samples),
            "l_samples": int(grid_spec.l_samples),
        }

        for env_kwargs in env_kwargs_list:
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_worker_loop,
                args=(child_conn, grid_spec_data, dict(env_kwargs)),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._processes.append(process)

    def reset(self) -> List[Dict[str, object]]:
        for conn in self._parent_conns:
            conn.send(("reset", None))
        return [conn.recv() for conn in self._parent_conns]

    def reset_indices(self, indices: Sequence[int]) -> List[Dict[str, object]]:
        reset_indices = [int(idx) for idx in indices]
        for idx in reset_indices:
            self._parent_conns[idx].send(("reset", None))
        return [self._parent_conns[idx].recv() for idx in reset_indices]

    def reset_to_scenario_indices(
        self,
        scenario_indices: Sequence[int],
        *,
        env_indices: Sequence[int] | None = None,
    ) -> List[Dict[str, object]]:
        if env_indices is None:
            env_indices = list(range(len(scenario_indices)))
        reset_env_indices = [int(idx) for idx in env_indices]
        reset_scenario_indices = [int(idx) for idx in scenario_indices]
        if len(reset_env_indices) != len(reset_scenario_indices):
            raise ValueError("env_indices and scenario_indices must have the same length.")
        for env_idx, scenario_idx in zip(reset_env_indices, reset_scenario_indices):
            self._parent_conns[env_idx].send(("reset_to_scenario_index", scenario_idx))
        return [self._parent_conns[idx].recv() for idx in reset_env_indices]

    def step(self, actions: Sequence[int]) -> List[StepResult]:
        for conn, action in zip(self._parent_conns, actions):
            conn.send(("step", int(action)))
        return [conn.recv() for conn in self._parent_conns]

    def step_indices(self, indices: Sequence[int], actions: Sequence[int]) -> List[StepResult]:
        step_indices = [int(idx) for idx in indices]
        step_actions = [int(action) for action in actions]
        if len(step_indices) != len(step_actions):
            raise ValueError("indices and actions must have the same length.")
        for idx, action in zip(step_indices, step_actions):
            self._parent_conns[idx].send(("step", action))
        return [self._parent_conns[idx].recv() for idx in step_indices]

    def close(self) -> None:
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                conn.close()
            except OSError:
                pass
        for process in self._processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

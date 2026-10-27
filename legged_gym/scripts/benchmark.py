"""Measure CPU collection throughput with real task rewards/history/resets.

python -m legged_gym.scripts.benchmark --task g1 --num_envs 4096 --steps 500
"""

import time

import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def main():
    # This measures CPU simulation only; it does not construct a learner.
    args = get_args(default_rl_device="cpu")
    env, _ = task_registry.make_env(args.task, args)
    actions = torch.zeros(env.num_envs, env.num_actions)
    try:
        for _ in range(10):
            env.step(actions)
        start = time.perf_counter()
        resets = 0
        for _ in range(args.steps):
            _, _, dones, _ = env.step(actions)
            resets += dones.sum().item()
        elapsed = time.perf_counter() - start
        transitions = args.steps * env.num_envs
        print(f"{args.task}: {env.num_envs} envs, {env.sim.batch.nthread} workers, {elapsed:.3f}s")
        print(
            f"{transitions / elapsed:,.0f} transitions/s; {transitions * env.cfg.control.decimation / elapsed:,.0f} physics steps/s"
        )
        print(f"{resets} resets; zero-action collection, excluding policy inference/PPO updates")
    finally:
        env.close()


if __name__ == "__main__":
    main()

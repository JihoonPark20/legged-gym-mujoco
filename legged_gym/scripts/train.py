# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Train Go2 or G1: python -m legged_gym.scripts.train --task g1 --headless."""

from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def train(args):
    env, _ = task_registry.make_env(name=args.task, args=args)
    try:
        runner, cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
        print(f"MuJoCo CPU: {env.num_envs} envs / {env.sim.batch.nthread} workers; PPO: {runner.device}")
        print(f"Logs: {runner.logger.log_dir}")
        resuming = args.resume or args.checkpoint_path is not None
        runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=not resuming)
    finally:
        env.close()


def main():
    train(get_args())


if __name__ == "__main__":
    main()

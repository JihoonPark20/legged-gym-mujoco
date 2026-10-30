# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Train Go2 or G1: python -m legged_gym.scripts.train --task g1 --headless."""

import shlex
import sys

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.agent_factory import agent_parameters, check_backend


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task, args)
    check_backend(train_cfg)  # Fail before allocating simulation if the backend is unavailable.
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    monitor = None
    try:
        runner, cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        algorithm = cfg.runner.algorithm.upper()
        print(f"MuJoCo CPU: {env.num_envs} envs / {env.sim.batch.nthread} workers; {algorithm}: {runner.device}")
        parameters = agent_parameters(cfg)
        fields = ["max_grad_norm", "desired_kl", "replay_buffer_size", "mini_batch_size"]
        fields += runner.agent_spec["learning_rate_fields"]
        settings = ", ".join(f"{key}={getattr(parameters, key)}" for key in fields if hasattr(parameters, key))
        print(f"{algorithm} settings: {settings}")
        print(f"Logs: {runner.logger.log_dir}")
        if cfg.viewer.enabled and not args.headless:
            try:
                from legged_gym.utils.training_monitor import TrainingMonitor

                monitor = TrainingMonitor(env, runner, cfg.viewer, task=args.task)
                print(f"Training monitor: {monitor.url}")
            except Exception as error:
                install = f"{shlex.quote(sys.executable)} -m pip install -e ."
                raise RuntimeError(
                    f"Training monitor failed to start with {sys.executable}: {error}. "
                    f"For missing dependencies, run `{install}` from the project directory. "
                    "Use --headless to train without the web monitor."
                ) from error
        resuming = args.resume or args.checkpoint_path is not None
        runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=not resuming)
    finally:
        if monitor is not None:
            monitor.close()
        env.close()


def main():
    train(get_args())


if __name__ == "__main__":
    main()

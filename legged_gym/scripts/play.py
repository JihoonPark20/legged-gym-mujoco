# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Checkpoint playback. On macOS use mjpython for the interactive viewer."""

import time

import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def play(args):
    args.resume = True
    args.play = True
    env_cfg, train_cfg = task_registry.get_cfgs(args.task, args)
    env_cfg.env.num_envs = args.num_envs or 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.curriculum = False
    env_cfg.domain_rand.randomize_encoder_bias = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.commands.ranges = env_cfg.commands.limit_ranges
    if args.velocity is not None:
        env_cfg.commands.heading_command = False
        env_cfg.commands.ranges = [[v, v] for v in args.velocity]
        env_cfg.commands.standing_fraction = 0.0
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    try:
        runner, _ = task_registry.make_alg_runner(env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
        print(f"Inference device: {runner.device}; MuJoCo physics: cpu")
        policy = runner.get_inference_policy(device=args.rl_device)
        env.headless = args.headless
        obs = env.get_observations()
        total_reward = 0.0
        done_count = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                start = time.perf_counter()
                actions = policy(obs.to(args.rl_device))
                obs, rewards, dones, _ = env.step(actions)
                total_reward += rewards.mean().item()
                done_count += dones.sum().item()
                if not args.headless:
                    if env._viewer is not None and not env._viewer.is_running():
                        break
                    time.sleep(max(0.0, env.dt - (time.perf_counter() - start)))
        print(f"Playback: accumulated mean reward={total_reward:.3f}, completed episodes={done_count}")
    finally:
        env.close()


def main():
    play(get_args())


if __name__ == "__main__":
    main()

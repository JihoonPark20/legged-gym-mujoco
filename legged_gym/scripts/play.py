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
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.curriculum = False
    # Generate the checkpoint's original atlas first; freezing progression must
    # not change how its terrain geometry is generated.
    if args.terrain_level is not None:
        if env_cfg.terrain.mesh_type == "plane" or args.terrain_level >= env_cfg.terrain.num_rows:
            raise ValueError("terrain_level must identify a row of a rough task's terrain atlas")
        env_cfg.terrain.max_init_terrain_level = args.terrain_level
    if args.terrain_column is not None:
        if env_cfg.terrain.mesh_type == "plane" or args.terrain_column >= env_cfg.terrain.num_cols:
            raise ValueError("terrain_column must identify a column of a rough task's terrain atlas")
    env_cfg.commands.ranges = env_cfg.commands.limit_ranges
    if args.velocity is not None:
        env_cfg.commands.ranges = [[v, v] for v in args.velocity]
        env_cfg.commands.standing_fraction = 0.0
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    if env.custom_origins:
        env.cfg.terrain.curriculum = False
        if args.terrain_level is not None or args.terrain_column is not None:
            if args.terrain_level is not None:
                env.terrain_levels[:] = args.terrain_level
            if args.terrain_column is not None:
                env.terrain_types[:] = args.terrain_column
            env._update_env_origins()
            env.reset()
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

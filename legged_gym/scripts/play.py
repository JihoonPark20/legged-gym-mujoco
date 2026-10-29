# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Checkpoint playback. On macOS use mjpython for the interactive viewer."""

import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.task_registry import policy_contract


def summarize_evaluation(trials):
    """Pool tracking samples; retain failed and non-settling trials in rate denominators."""
    steps = sum(row["tracked_steps"] for row in trials)
    settling = [row["settling_time_s"] for row in trials if row["settling_time_s"] is not None]
    summary = {
        "trials": len(trials),
        "tracked_steps": steps,
        "failure_rate": sum(row["failed"] for row in trials) / len(trials),
        "warmup_failure_rate": sum(row["failure_stage"] == "warmup" for row in trials) / len(trials),
        "tilt_failure_rate": sum(row["tilt_failure"] for row in trials) / len(trials),
        "contact_failure_rate": sum(row["contact_failure"] for row in trials) / len(trials),
        "mean_survival_s": float(np.mean([row["survival_s"] for row in trials])),
        "settled_fraction": len(settling) / len(trials),
        "mean_settling_time_s": float(np.mean(settling)) if settling else None,
    }
    for axis in ("x", "y", "yaw"):
        summary[f"mae_{axis}"] = (
            sum((row[f"mae_{axis}"] or 0) * row["tracked_steps"] for row in trials) / steps if steps else None
        )
        summary[f"rmse_{axis}"] = (
            math.sqrt(sum((row[f"rmse_{axis}"] or 0) ** 2 * row["tracked_steps"] for row in trials) / steps)
            if steps
            else None
        )
    return summary


@torch.inference_mode()
def evaluate(env, policy, args, cfg):
    seeds = args.eval_seeds if args.eval_seeds is not None else [args.seed] if args.seed is not None else cfg.seeds
    steps = args.steps or math.ceil(cfg.duration_s / env.dt)
    warmup_steps = math.ceil(cfg.warmup_s / env.dt)
    hold_steps = max(1, math.ceil(cfg.settling_hold_s / env.dt))
    tolerance = np.asarray(cfg.velocity_tolerance, dtype=float)
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be distinct and nonnegative")
    if tolerance.shape != (3,) or not np.isfinite(tolerance).all() or np.any(tolerance <= 0):
        raise ValueError("Evaluation needs three finite positive velocity tolerances")
    if not cfg.commands or cfg.warmup_s < 0 or cfg.duration_s <= 0 or cfg.settling_hold_s <= 0:
        raise ValueError("Provide evaluation commands, nonnegative warmup and positive duration/hold time")
    trials = []
    for name, (warmup_command, command) in cfg.commands.items():
        if np.asarray([warmup_command, command]).shape != (2, 3) or not np.isfinite([warmup_command, command]).all():
            raise ValueError(f"Invalid evaluation commands: {name}")
        for seed in seeds:
            # Each scenario starts from the same seeded reset distribution.
            set_seed(seed)
            env.rng = np.random.default_rng(seed)
            env.reset()
            active = np.ones(env.num_envs, dtype=bool)
            counts = np.zeros(env.num_envs, dtype=int)
            survival_steps = np.zeros(env.num_envs, dtype=int)
            absolute = np.zeros((env.num_envs, 3))
            squared = np.zeros_like(absolute)
            streak = np.zeros(env.num_envs, dtype=int)
            settling = np.full(env.num_envs, np.nan)
            tilt = np.zeros(env.num_envs, dtype=bool)
            contact = np.zeros(env.num_envs, dtype=bool)
            failure_stage = [None] * env.num_envs
            for stage, target, count in (("warmup", warmup_command, warmup_steps), ("tracking", command, steps)):
                env.commands[:] = target
                env.standing_envs[:] = False
                env._next_command[:] = np.inf
                env.compute_observations()
                obs = env.get_observations()
                for step in range(count):
                    if not active.any():
                        break
                    obs, _, done, _ = env.step(policy(obs.to(args.rl_device)))
                    if np.any(active & done.numpy() & ~env.terminated_buf):
                        raise RuntimeError("Evaluation timed out; its episode horizon must include warmup and tracking")
                    failed = active & env.terminated_buf
                    survival_steps[active] += 1
                    if stage == "tracking":
                        error = env.last_tracking_velocity - target
                        absolute[active] += np.abs(error[active])
                        squared[active] += np.square(error[active])
                        counts[active] += 1
                        inside = np.all(np.abs(error) <= tolerance, axis=1) & active & ~failed
                        streak[:] = np.where(inside, streak + 1, 0)
                        reached = (streak >= hold_steps) & np.isnan(settling)
                        # Start of the first continuously in-tolerance window.
                        settling[reached] = (step + 2 - hold_steps) * env.dt
                    tilt[failed] = env._tilt_failure[failed]
                    contact[failed] = env._contact_failure[failed]
                    for index in np.flatnonzero(failed):
                        failure_stage[index] = stage
                    active[failed] = False  # Never score an auto-reset as recovery.
            for index in range(env.num_envs):
                row = {
                    "scenario": name,
                    "seed": int(seed),
                    "env": index,
                    "tracked_steps": int(counts[index]),
                    "failed": bool(not active[index]),
                    "failure_stage": failure_stage[index],
                    "tilt_failure": bool(tilt[index]),
                    "contact_failure": bool(contact[index]),
                    "survival_s": float(survival_steps[index] * env.dt),
                    "settling_time_s": float(settling[index]) if np.isfinite(settling[index]) else None,
                }
                for column, axis in enumerate(("x", "y", "yaw")):
                    row[f"mae_{axis}"] = float(absolute[index, column] / counts[index]) if counts[index] else None
                    row[f"rmse_{axis}"] = (
                        float(np.sqrt(squared[index, column] / counts[index])) if counts[index] else None
                    )
                trials.append(row)

    summaries = []
    print("Scenario       Seed    Fail     Settled     MAE X / Y / yaw")
    for name in cfg.commands:
        for seed in [*seeds, "all"]:
            rows = [row for row in trials if row["scenario"] == name and (seed == "all" or row["seed"] == seed)]
            summary = {"scenario": name, "seed": seed, **summarize_evaluation(rows)}
            summaries.append(summary)
            if seed == "all":
                errors = " / ".join(
                    "—" if summary[f"mae_{axis}"] is None else f"{summary[f'mae_{axis}']:.3f}"
                    for axis in ("x", "y", "yaw")
                )
                print(
                    f"{name:<14} all   {summary['failure_rate']:6.1%}    {summary['settled_fraction']:6.1%}     {errors}"
                )
    checkpoint = Path(args.checkpoint_path).resolve()
    protocol = {
        "version": 1,
        "seeds": list(seeds),
        "num_envs": env.num_envs,
        "dt": env.dt,
        "warmup_steps": warmup_steps,
        "tracking_steps": steps,
        "settling_hold_steps": hold_steps,
        "velocity_tolerance": tolerance.tolist(),
        "commands": class_to_dict(cfg.commands),
        "velocity_frame": "root link body frame, matching tracking rewards",
        "observation_noise": False,
        "domain_randomization": False,
        "reset_distribution": "Task default root XY/yaw randomization, controlled by the evaluation seeds",
    }
    protocol_digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    report = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "rl_device": str(args.rl_device),
        "versions": {name: version(name) for name in ("mujoco", "torch", "rsl-rl-lib")},
        "protocol": protocol,
        "protocol_sha256": protocol_digest,
        "contract": policy_contract(env),
        "env_config": class_to_dict(env.cfg),
        "metric_definitions": {
            "tracking": "Sample-weighted MAE/RMSE after warmup, through first failure; X/Y in m/s, yaw in rad/s.",
            "failure": "First task termination (tilt or illegal contact), including warmup. Auto-reset samples excluded.",
            "survival": "Time observed before first failure or end of trial, including warmup; bounded by trial duration.",
            "settling": "First continuous all-axis tolerance window; time from command change. Null if not reached.",
            "mean_settling_time_s": "Mean among responders only; always interpret alongside settled_fraction and failure_rate.",
        },
        "summary": summaries,
        "trials": trials,
    }
    directory = Path(args.eval_output) if args.eval_output else checkpoint.parent / "evaluations"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = directory / f"{args.task}_{checkpoint.stem}_{stamp}"
    path.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as stream:
        identity = {"task": args.task, "checkpoint": str(checkpoint), "protocol_sha256": protocol_digest}
        writer = csv.DictWriter(stream, fieldnames=[*identity, *summaries[0]])
        writer.writeheader()
        writer.writerows({**identity, **row} for row in summaries)
    print(f"Evaluation: {path.with_suffix('.json')}\nSummary: {path.with_suffix('.csv')}")
    return report


def play(args):
    args.resume = True
    args.play = True
    # Evaluate every checkpoint with the current task's common protocol.
    evaluation_cfg = task_registry.get_cfgs(args.task)[1].evaluation
    env_cfg, train_cfg = task_registry.get_cfgs(args.task, args)
    evaluating = getattr(args, "evaluate", False)
    env_cfg.env.num_envs = args.num_envs or (evaluation_cfg.num_envs if evaluating else 1)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.curriculum = False
    env_cfg.domain_rand.randomize_encoder_bias = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.commands.ranges = env_cfg.commands.limit_ranges
    if evaluating:
        env_cfg.commands.heading_command = False
        env_cfg.commands.standing_fraction = 0.0
        env_cfg.commands.ranges = [[0, 0]] * 3
        dt = env_cfg.sim.dt * env_cfg.control.decimation
        count = (args.steps or math.ceil(evaluation_cfg.duration_s / dt)) + math.ceil(evaluation_cfg.warmup_s / dt)
        env_cfg.env.episode_length_s = max(env_cfg.env.episode_length_s, (count + 1) * dt)
    if args.velocity is not None:
        env_cfg.commands.heading_command = False
        env_cfg.commands.ranges = [[v, v] for v in args.velocity]
        env_cfg.commands.standing_fraction = 0.0
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    try:
        runner, _ = task_registry.make_alg_runner(env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
        print(f"Inference device: {runner.device}; MuJoCo physics: cpu")
        policy = runner.get_inference_policy(device=args.rl_device)
        if evaluating:
            return evaluate(env, policy, args, evaluation_cfg)
        env.headless = args.headless
        obs = env.get_observations()
        total_reward = 0.0
        done_count = 0
        with torch.inference_mode():
            for _ in range(args.steps or 1000):
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
    play(get_args(playback=True))


if __name__ == "__main__":
    main()

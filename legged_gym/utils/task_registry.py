# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym task_registry, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Task construction and the rsl-rl 5.5 runner configuration boundary."""

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR

from .mujoco_helpers import (
    class_to_dict,
    get_args,
    get_load_path,
    resolve_rl_device,
    set_seed,
    update_cfg_from_args,
    update_class_from_dict,
)


def policy_contract(env):
    """Fields that must match when a saved policy is applied to this task."""
    contract = {
        "version": 1,
        "robot": env.cfg.asset.file,
        "joint_names": env.dof_names,
        "policy_obs": env.num_obs,
        "critic_obs": env.num_privileged_obs,
        "history_length": env.cfg.env.history_length,
        "history_order": "term_oldest_first",
        "sim_dt": env.cfg.sim.dt,
        "decimation": env.cfg.control.decimation,
        "action_scale": env.cfg.control.action_scale,
        "clip_actions": env.cfg.control.clip_actions,
        "default_joint_pos": env.default_dof_pos.tolist(),
        "kp": env.p_gains.tolist(),
        "kd": env.d_gains.tolist(),
        "effort_limits": env.torque_limits.tolist(),
        "torque_speed_curve": env.cfg.control.torque_speed_curve,
        "armature": env.cfg.control.armature,
        "frictionloss": env.cfg.control.frictionloss,
    }
    if env.custom_origins or env.cfg.terrain.measure_heights:
        contract["terrain"] = {
            "mesh_type": env.cfg.terrain.mesh_type,
            "actor_height_scan": env.cfg.observations.actor_height_scan,
            "critic_height_scan": env.cfg.observations.critic_height_scan,
            "measured_points_x": list(env.cfg.terrain.measured_points_x),
            "measured_points_y": list(env.cfg.terrain.measured_points_y),
            "height_observation_offset": env.cfg.terrain.height_observation_offset,
            "height_observation_scale": env.cfg.terrain.height_observation_scale,
            "height_interpolation": "mujoco_triangles",
        }
    return contract


def runner_config(cfg):
    policy = class_to_dict(cfg.policy)
    return {
        "seed": cfg.seed,
        "num_steps_per_env": cfg.runner.num_steps_per_env,
        "save_interval": cfg.runner.save_interval,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": policy["actor_hidden_dims"],
            "activation": policy["activation"],
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "init_std": policy["init_noise_std"],
            },
        },
        "critic": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": policy["critic_hidden_dims"],
            "activation": policy["activation"],
            "obs_normalization": False,
        },
        "algorithm": {"class_name": "rsl_rl.algorithms:PPO", **class_to_dict(cfg.algorithm)},
        "logger": "tensorboard",
        "check_for_nan": True,
    }


class LeggedOnPolicyRunner(OnPolicyRunner):
    """Upstream PPO, with policy-contract and curriculum checkpoint metadata."""

    def save(self, path, infos=None):
        metadata = dict(infos or {})
        metadata["legged_gym"] = {
            "contract": policy_contract(self.env),
            "training_state": self.env.training_state_dict(),
            "next_iteration": self.current_learning_iteration + 1,
        }
        super().save(path, metadata)

    def load(self, path, load_cfg=None, strict=True, map_location=None, *, restore_env=True):
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        metadata = (checkpoint.get("infos") or {}).get("legged_gym")
        if metadata is None or metadata.get("contract") != policy_contract(self.env):
            raise ValueError("Checkpoint policy contract does not match this robot/configuration")
        del checkpoint
        # Load on CPU, then let model/optimizer state_dict loading place tensors
        # on the constructed learner device. This also keeps Adam step scalars portable.
        infos = super().load(path, load_cfg, strict, map_location or "cpu")
        if restore_env:
            self.current_learning_iteration = infos["legged_gym"].get(
                "next_iteration", self.current_learning_iteration + 1
            )
            self.env.load_training_state_dict(infos["legged_gym"]["training_state"])
        return infos


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name):
        if name not in self.task_classes:
            raise ValueError(f"Unknown task {name!r}; available: {', '.join(self.task_classes)}")
        return self.task_classes[name]

    def get_cfgs(self, name, args=None):
        self.get_task_class(name)
        env_cfg = deepcopy(self.env_cfgs[name])
        train_cfg = deepcopy(self.train_cfgs[name])
        if args is not None and (getattr(args, "resume", False) or getattr(args, "checkpoint_path", None)):
            if not getattr(args, "checkpoint_path", None):
                experiment = getattr(args, "experiment_name", None) or train_cfg.runner.experiment_name
                root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / experiment
                args.checkpoint_path = get_load_path(root, args.load_run, args.checkpoint)
            snapshot_path = Path(args.checkpoint_path).resolve().parent / "run_config.json"
            if snapshot_path.is_file():
                snapshot = json.loads(snapshot_path.read_text())
                if snapshot["env"]["asset"]["file"] != env_cfg.asset.file:
                    raise ValueError("Checkpoint snapshot belongs to a different robot")
                saved_terrain = snapshot["env"].get("terrain", {}).get("mesh_type", "plane")
                if saved_terrain != env_cfg.terrain.mesh_type:
                    raise ValueError("Checkpoint snapshot belongs to a different terrain task (flat/rough)")
                update_class_from_dict(env_cfg, snapshot["env"])
                update_class_from_dict(train_cfg, snapshot["train"])
        if args is not None:
            update_cfg_from_args(env_cfg, train_cfg, args)
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None):
        args = get_args() if args is None else args
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name, args)
        else:
            env_cfg = deepcopy(env_cfg)
            update_cfg_from_args(env_cfg, None, args)
        env_cfg.seed = set_seed(env_cfg.seed)
        env = self.get_task_class(name)(
            cfg=env_cfg,
            physics_engine=getattr(args, "physics_engine", "mujoco"),
            sim_device=getattr(args, "sim_device", "cpu"),
            headless=True,
        )
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"):
        args = get_args() if args is None else args
        if train_cfg is None:
            if name is None:
                raise ValueError("Provide a task name or train_cfg")
            _, train_cfg = self.get_cfgs(name, args)
        else:
            train_cfg = deepcopy(train_cfg)
            update_cfg_from_args(None, train_cfg, args)
        train_cfg.seed = env.cfg.seed
        device = resolve_rl_device(getattr(args, "rl_device", "auto"))
        if log_root == "default":
            log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
        log_dir = None
        if log_root is not None:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            if train_cfg.runner.run_name:
                run_name += "_" + train_cfg.runner.run_name
            directory = Path(log_root) / run_name
            directory.mkdir(parents=True, exist_ok=False)
            log_dir = str(directory)
            (directory / "run_config.json").write_text(
                json.dumps(
                    {
                        "task": name,
                        "env": class_to_dict(env.cfg),
                        "train": class_to_dict(train_cfg),
                        "rl_device": device,
                        "contract": policy_contract(env),
                    },
                    indent=2,
                )
                + "\n"
            )
        runner = LeggedOnPolicyRunner(env, runner_config(train_cfg), log_dir, device=device)
        if getattr(args, "resume", False) or getattr(args, "checkpoint_path", None):
            path = getattr(args, "checkpoint_path", None)
            if path is None:
                raise ValueError("Resolve the resume checkpoint through get_cfgs before constructing the runner")
            runner.load(path, restore_env=not getattr(args, "play", False))
        return runner, train_cfg


task_registry = TaskRegistry()

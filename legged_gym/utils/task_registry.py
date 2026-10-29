# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym task_registry, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Task construction and the rsl-rl runner configuration boundary."""

import json
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR

from .helpers import (
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
        "version": 2,
        "robot": env.cfg.asset.file,
        "asset_sha256": env.sim.asset_digest,
        "joint_names": env.dof_names,
        "policy_obs": env.num_obs,
        "critic_obs": env.num_privileged_obs,
        "sim_dt": env.cfg.sim.dt,
        "decimation": env.cfg.control.decimation,
        "action_scale": env.cfg.control.action_scale,
        "clip_actions": env.cfg.control.clip_actions,
        "default_joint_pos": env.default_dof_pos.tolist(),
        "kp": env.p_gains.tolist(),
        "kd": env.d_gains.tolist(),
        "effort_limits": env.torque_limits.tolist(),
        "armature": env.cfg.control.armature,
        "frictionloss": env.cfg.control.frictionloss,
        "foot_sites": env.cfg.asset.foot_sites,
        "solver_settings": {
            key: getattr(env.cfg.sim, key)
            for key in ("iterations", "cone", "ls_iterations", "tolerance", "ls_tolerance", "ccd_iterations")
        },
    }
    if hasattr(env.cfg.asset, "self_collisions"):
        contract["contact_settings"] = {
            key: getattr(env.cfg.asset, key)
            for key in ("self_collisions", "foot_contact_bodies", "foot_geoms", "foot_solimp", "reward_body")
        }
    return contract


def runner_config(cfg):
    policy = class_to_dict(cfg.policy)
    algorithm = class_to_dict(cfg.algorithm)
    return {
        "seed": cfg.seed,
        "num_steps_per_env": cfg.runner.num_steps_per_env,
        "save_interval": cfg.runner.save_interval,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": policy["actor_hidden_dims"],
            "activation": policy["activation"],
            "obs_normalization": policy["obs_normalization"],
            "distribution_cfg": {
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "std_type": "scalar",
                "init_std": policy["init_noise_std"],
            },
        },
        "critic": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": policy["critic_hidden_dims"],
            "activation": policy["activation"],
            "obs_normalization": policy["obs_normalization"],
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
            "rnd_cfg": None,
            "symmetry_cfg": None,
            **algorithm,
        },
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
        if metadata is None or any(
            metadata.get("contract", {}).get(key) != value for key, value in policy_contract(self.env).items()
        ):
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
                    raise ValueError("Checkpoint snapshot belongs to a different robot/configuration")
                update_class_from_dict(env_cfg, snapshot["env"], strict=False)
                update_class_from_dict(train_cfg, snapshot["train"], strict=False)
                train_cfg.runner.experiment_name = name
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
        if version("rsl-rl-lib") != "5.0.1":
            raise RuntimeError(
                "This project requires rsl-rl-lib 5.0.1. Install dependencies with: python -m pip install -e ."
            )
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
            # Loading Adam restores its saved LR. An explicit CLI override must
            # update both Adam and PPO's adaptive scheduler after that restore.
            learning_rate = getattr(args, "learning_rate", None)
            if learning_rate is not None:
                runner.alg.learning_rate = learning_rate
                for group in runner.alg.optimizer.param_groups:
                    group["lr"] = learning_rate
        return runner, train_cfg


task_registry = TaskRegistry()

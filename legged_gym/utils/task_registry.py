# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym task_registry, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Task construction, run snapshots, and delegation to the shared agent factory."""

import json
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from legged_gym import LEGGED_GYM_ROOT_DIR

from .agent_factory import create_agent, override_learning_rate, policy_contract
from .helpers import (
    class_to_dict,
    get_args,
    get_load_path,
    resolve_rl_device,
    set_seed,
    update_cfg_from_args,
    update_class_from_dict,
)


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
                        "rsl_rl_version": version("rsl-rl-lib"),
                        "contract": policy_contract(env),
                    },
                    indent=2,
                )
                + "\n"
            )
        runner = create_agent(env, train_cfg, log_dir, device=device)
        if getattr(args, "resume", False) or getattr(args, "checkpoint_path", None):
            path = getattr(args, "checkpoint_path", None)
            if path is None:
                raise ValueError("Resolve the resume checkpoint through get_cfgs before constructing the runner")
            runner.load(path, restore_env=not getattr(args, "play", False))
            # Loading Adam restores its saved LR. An explicit CLI override must
            # update both Adam and PPO's adaptive scheduler after that restore.
            learning_rate = getattr(args, "learning_rate", None)
            if learning_rate is not None:
                override_learning_rate(runner.alg, learning_rate)
        return runner, train_cfg


task_registry = TaskRegistry()

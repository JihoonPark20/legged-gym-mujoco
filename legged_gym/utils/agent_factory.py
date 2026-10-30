# SPDX-License-Identifier: BSD-3-Clause
"""Shared construction and checkpoint lifecycle for rsl_rl learning agents.

Agent definitions live in the training configuration. Upstream constructors are
used when available; the actor/critic/replay protocol handles backends whose
default constructor assumes a different simulator. Learning stays upstream.
"""

import importlib
import inspect
from copy import deepcopy

import torch

from .helpers import class_to_dict


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


def agent_spec(cfg):
    name = cfg.runner.algorithm
    if name not in cfg.agents:
        raise ValueError(f"Unknown algorithm {name!r}; configured agents: {', '.join(cfg.agents)}")
    spec = deepcopy(cfg.agents[name])
    if spec["kind"] not in ("on_policy", "off_policy"):
        raise ValueError("Agent kind must be on_policy or off_policy")
    if spec["construction"] not in ("upstream", "actor_critic_replay"):
        raise ValueError("Agent construction must be upstream or actor_critic_replay")
    if spec["construction"] == "actor_critic_replay" and spec["kind"] != "off_policy":
        raise ValueError("Replay construction requires an off_policy agent")
    if spec.get("action_transform") not in (None, "joint_limits"):
        raise ValueError("Unsupported action_transform; use joint_limits or the upstream constructor")
    return spec


def agent_parameters(cfg):
    return getattr(cfg, agent_spec(cfg)["parameter_group"])


def resolve_class(path):
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


def check_backend(cfg):
    """Resolve the configured classes before allocating simulation or replay."""
    spec = agent_spec(cfg)
    fields = ["runner_class", "algorithm_class", "actor_class", "critic_class"]
    if spec["construction"] == "actor_critic_replay":
        fields.append("storage_class")
    try:
        classes = {key: resolve_class(spec[key]) for key in fields}
    except (ImportError, AttributeError) as error:
        hint = spec.get("install_hint", "Install the package providing the configured classes.")
        raise RuntimeError(f"Backend for {cfg.runner.algorithm!r} is unavailable. {hint}") from error
    return classes


def model_api():
    # Retain the checkpoint compatibility marker used by existing project runs.
    model = resolve_class("rsl_rl.models:MLPModel")
    return "v5" if "distribution_cfg" in inspect.signature(model.__init__).parameters else "v4"


def runner_config(cfg):
    spec = agent_spec(cfg)
    policy = class_to_dict(cfg.policy)
    actor = {
        "class_name": spec["actor_class"],
        "hidden_dims": policy["actor_hidden_dims"],
        "activation": policy["activation"],
        "obs_normalization": policy["obs_normalization"],
    }
    critic = {
        "class_name": spec["critic_class"],
        "hidden_dims": policy["critic_hidden_dims"],
        "activation": policy["activation"],
        "obs_normalization": policy["obs_normalization"],
    }
    distribution = spec.get("distribution")
    if distribution == "gaussian":
        actor_class = resolve_class(spec["actor_class"])
        if "distribution_cfg" in inspect.signature(actor_class.__init__).parameters:
            actor["distribution_cfg"] = {
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "std_type": "scalar",
                "init_std": policy["init_noise_std"],
            }
        else:
            actor.update(stochastic=True, init_noise_std=policy["init_noise_std"], noise_std_type="scalar")
            critic["stochastic"] = False
    elif distribution == "squashed_gaussian":
        actor["init_noise_std"] = policy["init_noise_std"]
    elif distribution is not None:
        raise ValueError(f"Unsupported distribution preset: {distribution!r}; supply model options instead")
    actor.update(deepcopy(spec.get("actor_options", {})))
    critic.update(deepcopy(spec.get("critic_options", {})))
    config = {
        "seed": cfg.seed,
        "num_steps_per_env": cfg.runner.num_steps_per_env,
        "save_interval": cfg.runner.save_interval,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": actor,
        "critic": critic,
        "algorithm": {
            **class_to_dict(agent_parameters(cfg)),
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "class_name": spec["algorithm_class"],
        },
        "logger": "tensorboard",
        "check_for_nan": True,
        "agent": spec,
    }
    if spec["kind"] == "off_policy":
        config.update(class_to_dict(cfg.off_policy))
    if spec["construction"] == "actor_critic_replay":
        config["algorithm"]["class_name"] = "legged_gym.utils.agent_factory:ActorCriticReplay"
    return config


def action_bounds(env):
    """Policy-space bounds corresponding to MuJoCo's soft joint limits."""
    limits = torch.as_tensor(env.dof_pos_limits, dtype=torch.float32)
    default = torch.as_tensor(env.default_dof_pos, dtype=torch.float32)
    scale = torch.as_tensor(env._action_scale, dtype=torch.float32).expand(env.num_actions)
    if not torch.isfinite(scale).all() or (scale == 0).any():
        raise ValueError("Bounded policies require finite, nonzero action scales")
    endpoints = (limits - default[:, None]) / scale[:, None]
    lower, upper = endpoints.min(dim=1).values, endpoints.max(dim=1).values
    clip = env.cfg.control.clip_actions
    if clip is not None:
        lower, upper = lower.clamp(min=-clip), upper.clamp(max=clip)
    if not torch.isfinite(endpoints).all() or not (upper > lower).all():
        raise ValueError("Bounded policies require finite action bounds with positive range for every joint")
    return lower, upper


def validate_config(env, cfg):
    spec = cfg["agent"]
    if spec.get("action_transform") == "joint_limits":
        action_bounds(env)
    if spec["kind"] != "off_policy":
        return
    for key in ("num_steps_per_env", "log_interval"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"Off-policy {key} must be a positive integer")
    if not isinstance(cfg["start_training"], int) or cfg["start_training"] < 0:
        raise ValueError("Off-policy start_training must be a nonnegative integer")
    if spec["construction"] != "actor_critic_replay":
        return
    alg = cfg["algorithm"]
    for key in ("replay_buffer_size", "mini_batch_size", "num_learning_epochs", "num_mini_batches", "n_steps"):
        if not isinstance(alg[key], int) or alg[key] <= 0:
            raise ValueError(f"Replay {key} must be a positive integer")
    if "policy_frequency" in alg and (not isinstance(alg["policy_frequency"], int) or alg["policy_frequency"] <= 0):
        raise ValueError("policy_frequency must be a positive integer")
    if alg["replay_buffer_size"] // env.num_envs < alg["n_steps"]:
        raise ValueError("replay_buffer_size must be at least num_envs * n_steps")
    if (cfg["start_training"] + 1) * cfg["num_steps_per_env"] < alg["n_steps"]:
        raise ValueError("Warmup must collect at least n_steps per environment before updating")


class ActorCriticReplay:
    """Constructor protocol only; returns an unmodified upstream algorithm.

    Models accept observation groups; the critic additionally accepts num_actions.
    Storage accepts the rsl_rl replay constructor arguments below. A joint_limits
    transform requires actor action_bias/action_range/log_action_range buffers.
    Backends with different contracts should use their own upstream constructor.
    """

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        spec = cfg["agent"]
        validate_config(env, cfg)
        parameters = deepcopy(cfg["algorithm"])
        parameters.pop("class_name")
        models = []
        for name, output_dim in (("actor", env.num_actions), ("critic", 1)):
            model_cfg = deepcopy(cfg[name])
            model_class = resolve_class(model_cfg.pop("class_name"))
            if name == "critic":
                model_cfg["num_actions"] = env.num_actions
            models.append(model_class(obs, cfg["obs_groups"], name, output_dim, **model_cfg).to(device))
        actor, critic = models
        if spec.get("action_transform") == "joint_limits":
            lower, upper = (value.to(device) for value in action_bounds(env))
            actor.action_bias.copy_((upper + lower) / 2)
            actor.action_range.copy_((upper - lower) / 2)
            actor.log_action_range.copy_(actor.action_range.log().sum())
        storage = resolve_class(spec["storage_class"])(
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            buffer_size=parameters["replay_buffer_size"],
            n_steps=parameters["n_steps"],
            gamma=parameters["gamma"],
        )
        return resolve_class(spec["algorithm_class"])(
            actor, critic, storage, device=device, multi_gpu_cfg=cfg.get("multi_gpu"), **parameters
        )


class AgentEnvironment:
    """Keep simulation on its own device while transferring terminal observations."""

    def __init__(self, env, device):
        self.unwrapped = env
        self.learner_device = device

    def __getattr__(self, name):
        return getattr(self.unwrapped, name)

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    def get_observations(self):
        return self.unwrapped.get_observations()

    def step(self, actions):
        obs, rewards, dones, extras = self.unwrapped.step(actions)
        if "time_outs_obs" in extras:
            extras = {**extras, "time_outs_obs": extras["time_outs_obs"].to(self.learner_device)}
        return obs, rewards, dones, extras


def override_learning_rate(algorithm, value):
    """Update all exposed optimizers and their matching scheduler attributes."""
    for name, optimizer in vars(algorithm).items():
        if isinstance(optimizer, torch.optim.Optimizer) and name.endswith("optimizer"):
            rate_name = name.removesuffix("optimizer") + "learning_rate"
            if hasattr(algorithm, rate_name):
                setattr(algorithm, rate_name, value)
            for group in optimizer.param_groups:
                group["lr"] = value


class AgentCheckpointMixin:
    """One checkpoint lifecycle for both policy families, driven by the agent spec."""

    def save(self, path, infos=None):
        metadata = dict(infos or {})
        state = {
            "algorithm": self.algorithm_name,
            "model_api": model_api(),
            "agent_spec": self.agent_spec,
            "contract": policy_contract(self.env),
            "training_state": self.env.training_state_dict(),
            "next_iteration": self.current_learning_iteration + 1,
        }
        if self.agent_spec.get("action_transform") == "joint_limits":
            state["action_bounds"] = [value.tolist() for value in action_bounds(self.env)]
        for key in self.agent_spec.get("checkpoint_counters", []):
            state[key] = getattr(self.alg, key)
        metadata["legged_gym"] = state
        super().save(path, metadata)

    def load(self, path, load_cfg=None, strict=True, map_location=None, *, restore_env=True):
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        state = (checkpoint.get("infos") or {}).get("legged_gym")
        if state is None:
            raise ValueError("Checkpoint policy contract is missing")
        if state.get("algorithm", "ppo") != self.algorithm_name:
            raise ValueError("Checkpoint algorithm does not match the selected algorithm")
        if state.get("model_api", "v5") != model_api():
            raise ValueError("Checkpoint rsl-rl model API differs; use the backend that saved it")
        if "agent_spec" in state and state["agent_spec"] != self.agent_spec:
            raise ValueError("Checkpoint agent definition does not match the configured agent")
        if any(state.get("contract", {}).get(key) != value for key, value in policy_contract(self.env).items()):
            raise ValueError("Checkpoint policy contract does not match this robot/configuration")
        if self.agent_spec.get("action_transform") == "joint_limits":
            if state.get("action_bounds") != [value.tolist() for value in action_bounds(self.env)]:
                raise ValueError("Checkpoint action bounds do not match this robot/configuration")
        del checkpoint
        # State dict loading moves model/optimizer tensors onto the learner device.
        infos = super().load(path, load_cfg, strict, map_location or "cpu")
        if restore_env:
            self.current_learning_iteration = state.get("next_iteration", self.current_learning_iteration + 1)
            self.env.load_training_state_dict(state["training_state"])
            for key in self.agent_spec.get("checkpoint_counters", []):
                setattr(self.alg, key, state.get(key, 0))
            if self.agent_spec.get("refill_replay_on_resume", False):
                self.alg.clear_storage()
                self.start_training = self.current_learning_iteration + self.cfg["start_training"]
        return infos


def create_agent(env, cfg, log_dir=None, device="cpu"):
    """Construct either policy family through the same environment/runner boundary."""
    classes = check_backend(cfg)
    config = runner_config(cfg)
    validate_config(env, config)
    spec = config["agent"]
    env.capture_terminal_observations = spec["kind"] == "off_policy"
    runner_type = type(
        "LeggedAgentRunner",
        (AgentCheckpointMixin, classes["runner_class"]),
        {"algorithm_name": cfg.runner.algorithm, "agent_spec": deepcopy(spec)},
    )
    return runner_type(AgentEnvironment(env, device), config, log_dir, device=device)

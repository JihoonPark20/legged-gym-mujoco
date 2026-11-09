"""Execution extension for configured RSL-style off-policy runners.

The backend still owns construction, losses, replay, save/load and export.
Adapters isolate collection state so update() can run while physics is pending.
"""

import inspect
import math
import os
import time
from contextlib import ExitStack

import torch


class ReplayAdapter:
    """Collection contract for the pinned rsl_rl_sac SAC implementation."""

    def validate(self, algorithm):
        for method in ("act", "process_env_step", "update", "train_mode", "get_policy"):
            if not callable(getattr(algorithm, method, None)):
                raise TypeError(f"Off-policy backend must implement {method}()")
        if not hasattr(algorithm, "transition") or not hasattr(algorithm, "replay_buffer"):
            raise TypeError("This backend requires its own execution_adapter (see FlashSACAdapter)")

    def act(self, algorithm, obs):
        # Own the action before a compiled update can reuse forward output storage.
        return algorithm.act(obs).detach().clone()

    def ready(self, algorithm):
        return algorithm.replay_buffer.num_transitions >= algorithm.n_steps

    def learning_rate(self, algorithm):
        return algorithm.actor_learning_rate

    def observe(self, algorithm, obs, actions, next_obs, rewards, dones, extras):
        # Update owns the algorithm while physics runs; restore only the explicit
        # transition inputs afterwards instead of depending on mutable act caches.
        algorithm.transition.observations = obs
        algorithm.transition.actions = actions
        algorithm.process_env_step(next_obs, rewards, dones, extras)


class FlashSACAdapter(ReplayAdapter):
    """rsl_rl_flashsac API, including replay warmup and normalized random actions.

    Use the normalized_joint_limits agent transform for FlashSAC's [-1, 1]
    replay actions. The environment wrapper applies the affine map for physics.
    """

    schedule_fields = (
        "learning_rate_init",
        "learning_rate_peak",
        "learning_rate_end",
        "learning_rate_warmup_steps",
        "learning_rate_decay_steps",
    )

    @staticmethod
    def prepare_config(cfg, spec):
        parameters = getattr(cfg, spec["parameter_group"])
        if parameters.learning_rate_decay_steps is None:
            parameters.learning_rate_decay_steps = (
                cfg.runner.max_iterations * parameters.num_learning_epochs * parameters.num_mini_batches
            )

    @staticmethod
    def model_config(cfg, spec):
        from .helpers import class_to_dict

        policy = getattr(cfg, spec["parameter_group"] + "_policy")
        return class_to_dict(policy.actor), class_to_dict(policy.critic)

    @classmethod
    def validate_config(cls, env, cfg):
        alg = cfg["algorithm"]
        for key in (
            "replay_buffer_size",
            "buffer_min_length",
            "mini_batch_size",
            "num_learning_epochs",
            "num_mini_batches",
            "n_steps",
            "learning_rate_decay_steps",
        ):
            if type(alg[key]) is not int or alg[key] <= 0:
                raise ValueError(f"FlashSAC {key} must be a positive integer")
        if alg["buffer_min_length"] > alg["replay_buffer_size"]:
            raise ValueError("FlashSAC buffer_min_length must not exceed replay_buffer_size")
        if alg.get("buffer_optimize_memory_usage", True):
            if alg["replay_buffer_size"] <= alg["n_steps"] * env.num_envs:
                raise ValueError("FlashSAC memory-efficient replay requires replay_buffer_size > n_steps * num_envs")
        elif alg["replay_buffer_size"] < env.num_envs:
            raise ValueError("FlashSAC replay_buffer_size must be at least num_envs")
        warmup = alg["learning_rate_warmup_steps"]
        if type(warmup) is not int or not 0 <= warmup < alg["learning_rate_decay_steps"]:
            raise ValueError("FlashSAC LR warmup must be nonnegative and shorter than LR decay")
        for key in cls.schedule_fields[:3]:
            value = alg[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"FlashSAC {key} must be finite and positive")
        for name in ("actor", "critic"):
            if any(key in cfg[name] for key in ("hidden_dims", "activation", "obs_normalization")):
                raise ValueError("FlashSAC models use flashsac_policy settings, not the common MLP policy options")
            for key in ("hidden_dim", "num_blocks"):
                if type(cfg[name][key]) is not int or cfg[name][key] <= 0:
                    raise ValueError(f"FlashSAC {name}.{key} must be a positive integer")
        if cfg["actor"]["log_std_min"] >= cfg["actor"]["log_std_max"]:
            raise ValueError("FlashSAC actor log_std_min must be smaller than log_std_max")
        critic = cfg["critic"]
        if type(critic["num_bins"]) is not int or critic["num_bins"] < 2 or critic["min_v"] >= critic["max_v"]:
            raise ValueError("FlashSAC critic requires at least two bins and min_v < max_v")
        if type(critic["num_qs"]) is not int or critic["num_qs"] < 1:
            raise ValueError("FlashSAC critic.num_qs must be positive")

    @classmethod
    def checkpoint_state(cls, cfg):
        return {key: cfg["algorithm"][key] for key in cls.schedule_fields}

    @classmethod
    def legacy_checkpoint_state(cls, snapshot, cfg):
        train = snapshot["train"]
        parameters = train[cfg["agent"]["parameter_group"]]
        result = {key: parameters.get(key, cfg["algorithm"][key]) for key in cls.schedule_fields}
        result["learning_rate_decay_steps"] = parameters.get("learning_rate_decay_steps") or (
            train["runner"]["max_iterations"] * parameters["num_learning_epochs"] * parameters["num_mini_batches"]
        )
        return result

    @classmethod
    def restore_state(cls, algorithm, cfg, state):
        from .agent_factory import resolve_class

        schedule = resolve_class("rsl_rl_flashsac.utils.lr_scheduler:warmup_cosine_decay_scheduler")(
            init_value=state["learning_rate_init"],
            peak_value=state["learning_rate_peak"],
            end_value=state["learning_rate_end"],
            warmup_steps=state["learning_rate_warmup_steps"],
            decay_steps=state["learning_rate_decay_steps"],
        )
        # LambdaLR state_dict does not serialize these functions. Restore them
        # before the backend loads scheduler counters and optimizer state.
        for name in ("actor", "critic", "temperature"):
            scheduler = getattr(algorithm, name + "_scheduler")
            scheduler.lr_lambdas = [lambda step: schedule(step) / state["learning_rate_peak"]] * len(scheduler.base_lrs)
        cfg["algorithm"].update(state)

    def validate(self, algorithm):
        for name in (
            "act",
            "act_random",
            "can_start_training",
            "process_env_step",
            "update",
            "get_policy",
            "train_mode",
        ):
            if not callable(getattr(algorithm, name, None)):
                raise TypeError(f"FlashSAC execution contract requires {name}()")
        if not hasattr(algorithm, "_last_obs") or not hasattr(algorithm, "_last_actions"):
            raise TypeError("Unsupported FlashSAC collection state API")

    def act(self, algorithm, obs):
        function = algorithm.act if algorithm.can_start_training() else algorithm.act_random
        return function(obs).detach().clone()

    def ready(self, algorithm):
        return algorithm.can_start_training()

    def observe(self, algorithm, obs, actions, next_obs, rewards, dones, extras):
        algorithm._last_obs, algorithm._last_actions = obs, actions
        algorithm.process_env_step(next_obs, rewards, dones, extras)


class OffPolicyRunnerMixin:
    """Add process collection to an existing backend OffPolicyRunner.

    Exactly one update() call per eligible iteration, like the serial runner.
    The first eligible update runs after collection (warmup/n-step safety). Later
    updates can overlap the last CPU step of an iteration using completed replay.
    At log/save/return boundaries all collected transitions have been ingested.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Capture before the optional training monitor wraps logger.log(**kwargs).
        self._logger_parameters = inspect.signature(self.logger.log).parameters

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.cfg.get("collector_mode", "serial") == "serial":
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        if not isinstance(num_learning_iterations, int) or num_learning_iterations < 1:
            raise ValueError("num_learning_iterations must be positive")
        if self.is_distributed:
            raise NotImplementedError("Process collection currently supports a single learner rank")
        from .agent_factory import resolve_class
        from .off_policy_profile import OffPolicyProfiler
        from .replay_sampling import sampling_session

        self.profiler = OffPolicyProfiler(self.cfg.get("profile", "off"), self.device)
        profiler = self.profiler
        resources = ExitStack()

        adapter_class = self.cfg["agent"].get("execution_adapter", "legged_gym.utils.off_policy_runner:ReplayAdapter")
        adapter = resolve_class(adapter_class)()
        adapter.validate(self.alg)
        if not callable(getattr(self.env, "step_async", None)):
            raise ValueError("Process collection requires task_registry.make_env(..., train_cfg=train_cfg)")
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        self._learn_start_time = time.time()  # Also supports backend wall-time checkpoint hooks.
        log_parameters = self._logger_parameters
        window_collect = window_learn = 0.0
        window_iters = 0
        save_interval = self.cfg["save_interval"]
        if save_interval <= 0:
            save_interval = max(1, num_learning_iterations // 10)
        try:
            self.logger.init_logging_writer()
            resources.enter_context(sampling_session(self.alg, self.cfg.get("replay_sampling", "native"), profiler))
            for it in range(start_it, total_it):
                start = time.perf_counter()
                update_time = 0.0
                updated = False
                loss_dict = {}
                for step in range(self.cfg["num_steps_per_env"]):
                    with torch.no_grad():
                        with profiler.measure("inference"):
                            actions = adapter.act(self.alg, obs)
                        with profiler.measure("action_transfer"):
                            env_actions = actions.to(self.env.device)
                        with profiler.measure("dispatch"):
                            self.env.step_async(env_actions)
                    if (
                        step == self.cfg["num_steps_per_env"] - 1
                        and it > self.start_training
                        and adapter.ready(self.alg)
                    ):
                        before = time.perf_counter()
                        with profiler.measure("update"):
                            loss_dict = self.alg.update()
                        update_time += time.perf_counter() - before
                        updated = True
                    with torch.no_grad():
                        with profiler.measure("receive"):
                            next_obs, rewards, dones, extras = self.env.step_wait()
                        if profiler.enabled:
                            for key, value in self.env.last_step_metrics.items():
                                profiler.totals[key] += value
                        if self.cfg.get("check_for_nan", True):
                            if (
                                not all(torch.isfinite(v).all() for v in next_obs.values())
                                or not torch.isfinite(rewards).all()
                            ):
                                raise FloatingPointError("Nonfinite collector observation/reward")
                        with profiler.measure("observation_transfer"):
                            next_obs, rewards, dones = (
                                next_obs.to(self.device),
                                rewards.to(self.device),
                                dones.to(self.device),
                            )
                        with profiler.measure("replay_insert"):
                            adapter.observe(self.alg, obs, actions, next_obs, rewards, dones, extras)
                        self.logger.process_env_step(
                            rewards, dones, extras, getattr(self.alg, "intrinsic_rewards", None)
                        )
                        obs = next_obs
                if it >= self.start_training and not updated:
                    before = time.perf_counter()
                    with profiler.measure("update"):
                        loss_dict = self.alg.update()
                    update_time += time.perf_counter() - before
                elapsed = time.perf_counter() - start
                if profiler.enabled:
                    profiler.totals["iteration"] += elapsed
                # The sum remains elapsed wall time, not CPU time + overlapping GPU time.
                window_collect += max(0.0, elapsed - update_time)
                window_learn += update_time
                window_iters += 1
                self.current_learning_iteration = it
                if it % self.log_interval == 0 or it == total_it - 1:
                    self._log_overlap_window(
                        it,
                        start_it,
                        total_it,
                        window_collect,
                        window_learn,
                        window_iters,
                        loss_dict,
                        log_parameters,
                        adapter.learning_rate(self.alg),
                    )
                    profiler.record(it, window_iters, self.logger)
                    window_collect = window_learn = 0.0
                    window_iters = 0
                if self.logger.writer is not None and it % save_interval == 0 and it != 0:
                    self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            if self.logger.writer is not None:
                self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        except BaseException:
            self.env.close()
            abort = getattr(self.logger, "abort_logging_writer", self.logger.stop_logging_writer)
            abort()
            raise
        else:
            self.logger.stop_logging_writer()
        finally:
            resources.close()

    def _log_overlap_window(self, it, start_it, total_it, collect, learn, count, losses, parameters, learning_rate):
        rnd = getattr(self.alg, "rnd", None)
        kwargs = dict(
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=collect,
            learn_time=learn,
            loss_dict=losses,
            learning_rate=learning_rate,
            action_std=self.alg.get_policy().output_std,
            rnd_weight=rnd.weight if rnd is not None else None,
        )
        if "alpha" in parameters:
            kwargs["alpha"] = getattr(self.alg, "alpha", None)
        size = self.cfg["num_steps_per_env"] * self.env.num_envs * self.gpu_world_size
        if "collection_size_override" in parameters:
            kwargs["collection_size_override"] = size * count
        else:
            kwargs["collect_time"], kwargs["learn_time"] = collect / count, learn / count
            if self.logger.writer is not None:
                # Include skipped iterations before the logger prints/writes its totals.
                self.logger.tot_timesteps += size * (count - 1)
                self.logger.tot_time += (collect + learn) * (1 - 1 / count)
        self.logger.log(**kwargs)
        if self.logger.writer is not None and callable(getattr(self, "_elapsed_wall_time", None)):
            self.logger.writer.add_scalar("Policy/temperature", self.alg.alpha, it)
            self.logger.writer.add_scalar("Train/env_steps", self.logger.tot_timesteps, it)
            self.logger.writer.add_scalar("Train/wall_time", self._elapsed_wall_time(), it)

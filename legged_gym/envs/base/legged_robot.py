"""CPU MuJoCo locomotion environment following the legged_gym structure.

MDP definitions adapted from Unitree RL Mjlab (Apache-2.0).
"""

import math

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from legged_gym.sim.mujoco_batch import MujocoBatch, joint_parameters


def rotate_inverse(q, v):
    xyz = q[:, 1:]
    return v - 2 * q[:, :1] * np.cross(xyz, v) + 2 * np.cross(xyz, np.cross(xyz, v))


class LeggedRobot(VecEnv):
    def __init__(self, cfg, sim_params=None, physics_engine="mujoco", sim_device="cpu", headless=True):
        if physics_engine != "mujoco" or str(sim_device) != "cpu":
            raise ValueError("Physics uses CPU MuJoCo")
        if cfg.env.num_envs < 1 or cfg.sim.dt <= 0 or cfg.control.decimation < 1 or cfg.env.episode_length_s <= 0:
            raise ValueError("Environment count, timestep, decimation and episode length must be positive")
        self.cfg, self.headless = cfg, headless
        self.device = torch.device("cpu")
        self.num_envs, self.num_actions = cfg.env.num_envs, cfg.env.num_actions
        self.dt = cfg.sim.dt * cfg.control.decimation
        self.max_episode_length = math.ceil(cfg.env.episode_length_s / self.dt)
        self.rng = np.random.default_rng(cfg.seed)
        self.sim = MujocoBatch(cfg, self.rng)
        self.dof_names = list(cfg.asset.joint_names)
        if len(self.dof_names) != self.num_actions or len(set(self.dof_names)) != self.num_actions:
            raise ValueError("Provide one unique joint name per action")
        self.default_dof_pos = joint_parameters(self.dof_names, cfg.init_state.default_joint_angles, 0.0)
        self.p_gains = joint_parameters(self.dof_names, cfg.control.stiffness)
        self.d_gains = joint_parameters(self.dof_names, cfg.control.damping)
        self.torque_limits = joint_parameters(self.dof_names, cfg.control.effort_limits)
        limits = self.sim.model.jnt_range[self.sim.joint_ids]
        centers = limits.mean(axis=1)
        half_ranges = (limits[:, 1] - limits[:, 0]) * cfg.rewards.soft_dof_pos_limit / 2
        self.dof_pos_limits = np.stack((centers - half_ranges, centers + half_ranges), axis=1)
        self._action_scale = (
            joint_parameters(self.dof_names, cfg.control.action_scale)
            if isinstance(cfg.control.action_scale, dict)
            else cfg.control.action_scale
        )
        self._pose_std = [
            joint_parameters(self.dof_names, getattr(cfg.rewards, f"pose_std_{mode}"))
            for mode in ("standing", "walking", "running")
        ]
        self.heading_targets = np.zeros(self.num_envs)
        self.heading_envs = np.zeros(self.num_envs, dtype=bool)
        self.standing_envs = np.zeros(self.num_envs, dtype=bool)
        self._push_initialized = False
        self._foot_sensor_cache = None
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.common_step_counter = 0
        shape = (self.num_envs, self.num_actions)
        self.actions = np.zeros(shape)
        self.last_actions = np.zeros(shape)
        self.dof_acc = np.zeros(shape)
        self.commands = np.zeros((self.num_envs, 3))
        self.command_ranges = np.array(cfg.commands.ranges, dtype=float, copy=True)
        self._next_push = np.zeros(self.num_envs)
        self._next_command = np.zeros(self.num_envs)
        feet_shape = (self.num_envs, len(cfg.asset.foot_sites))
        self.feet_air_time = np.zeros(feet_shape)
        self.feet_contact_time = np.zeros(feet_shape)
        self._feet_contact = np.zeros(feet_shape, dtype=bool)
        self._collision_history = np.zeros(
            (cfg.rewards.contact_history_length, self.num_envs, self.sim.history_contact_count, 3)
        )
        self._contact_cursor = 0
        self._prepare_reward_function()
        self.episode_sums = {name: np.zeros(self.num_envs) for name in self.reward_scales}
        self._diagnostic_steps = np.zeros(self.num_envs, dtype=int)
        self._diagnostic_sums = {
            key: np.zeros(self.num_envs)
            for key in (
                [
                    f"{metric}_{axis}"
                    for axis in ("x", "y", "yaw")
                    for metric in ("command", "velocity", "tracking_mae", "tracking_rmse")
                ]
                + [f"contact_feet_{count}_fraction" for count in range(feet_shape[1] + 1)]
            )
        }
        self._liftoff_count = np.zeros(self.num_envs)
        self.obs_buf = None
        self._viewer = self._viewer_model = self._viewer_data = None
        self._closed = False
        self.reset()
        self.num_obs = self.obs_buf["policy"].shape[-1]
        self.num_privileged_obs = self.obs_buf["critic"].shape[-1]

    def _prepare_reward_function(self):
        self.reward_scales = {name: value for name, value in self.cfg.rewards.scales.items() if value != 0}
        self.reward_functions = {name: getattr(self, f"_reward_{name}") for name in self.reward_scales}

    def compute_reward(self):
        self.rew_buf = np.zeros(self.num_envs)
        for name, function in self.reward_functions.items():
            value = function() * self.reward_scales[name] * self.dt
            self.rew_buf += value
            self.episode_sums[name] += value
        if not np.isfinite(self.rew_buf).all():
            raise FloatingPointError("Nonfinite reward")

    def get_observations(self):
        return self.obs_buf

    def get_tracking_velocity(self):
        """Root-link body-frame X/Y velocity and yaw rate, as used by tracking rewards."""
        return np.column_stack((self.base_lin_vel[:, :2], self.base_ang_vel[:, 2]))

    def _update_diagnostics(self):
        # Retain the terminal sample for evaluation before step() auto-resets it.
        self.last_tracking_velocity = self.get_tracking_velocity()
        error = self.last_tracking_velocity - self.commands
        for index, axis in enumerate(("x", "y", "yaw")):
            self._diagnostic_sums[f"command_{axis}"] += self.commands[:, index]
            self._diagnostic_sums[f"velocity_{axis}"] += self.last_tracking_velocity[:, index]
            self._diagnostic_sums[f"tracking_mae_{axis}"] += np.abs(error[:, index])
            self._diagnostic_sums[f"tracking_rmse_{axis}"] += np.square(error[:, index])
        contacts = self._feet_contact.sum(axis=1)
        for count in range(self._feet_contact.shape[1] + 1):
            self._diagnostic_sums[f"contact_feet_{count}_fraction"] += contacts == count
        self._diagnostic_steps += 1

    def _episode_metrics(self, ids):
        steps = np.maximum(self._diagnostic_steps[ids], 1)
        return {
            **{
                f"Reward/{name}": float(values[ids].mean() / self.cfg.env.episode_length_s)
                for name, values in self.episode_sums.items()
            },
            "Episode/length_s": float(self.episode_length_buf.numpy()[ids].mean() * self.dt),
            "Episode/time_out_fraction": float(self.time_out_buf[ids].mean()),
            "Episode/failure_fraction": float(self.terminated_buf[ids].mean()),
            "Episode/tilt_failure_fraction": float(self._tilt_failure[ids].mean()),
            "Episode/contact_failure_fraction": float(self._contact_failure[ids].mean()),
            **{
                f"Diagnostics/{key}": float(
                    (np.sqrt(values[ids] / steps) if key.startswith("tracking_rmse_") else values[ids] / steps).mean()
                )
                for key, values in self._diagnostic_sums.items()
            },
            "Diagnostics/foot_liftoffs_per_second": float((self._liftoff_count[ids] / (steps * self.dt)).mean()),
            "Curriculum/max_command_x": float(self.command_ranges[0, 1]),
        }

    def training_state_dict(self):
        return {"command_ranges": self.command_ranges.tolist(), "common_step_counter": self.common_step_counter}

    def load_training_state_dict(self, state):
        ranges = np.asarray(state["command_ranges"], dtype=float)
        if ranges.shape != (3, 2) or not np.isfinite(ranges).all() or np.any(ranges[:, 0] > ranges[:, 1]):
            raise ValueError("Invalid command curriculum state")
        self.command_ranges[:] = ranges
        self.common_step_counter = int(state["common_step_counter"])
        self._resample_commands(np.arange(self.num_envs))
        self._update_commands()
        self.compute_observations()

    def render(self):
        if self._viewer is None:
            import mujoco.viewer

            self._viewer_model = self.sim.batch.get_model(0)
            self._viewer_data = self.sim.batch.get_data(0)
            self._viewer = mujoco.viewer.launch_passive(self._viewer_model, self._viewer_data)
            self._viewer.cam.distance = 3.0
            self._viewer.cam.elevation = -20
        if self._viewer.is_running():
            self.sim.batch.get_data(0, self._viewer_data)
            with self._viewer.lock():
                self._viewer.cam.lookat[:] = self.base_pos[0]
            self._viewer.sync()

    def close(self):
        if not self._closed:
            if self._viewer is not None:
                self._viewer.close()
            self.sim.close()
            self._closed = True

    def step(self, actions):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        actions = np.asarray(actions)
        if actions.shape != self.actions.shape or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions of shape {self.actions.shape}")
        self.last_actions[:] = self.actions
        self.actions[:] = actions
        if self.cfg.control.clip_actions is not None:
            np.clip(self.actions, -self.cfg.control.clip_actions, self.cfg.control.clip_actions, out=self.actions)
        targets = self.default_dof_pos + self._action_scale * self.actions - self.sim.encoder_bias
        for _ in range(self.cfg.control.decimation):
            self.sim.step(targets)
            self._update_contacts()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._refresh_state()
        # ContactSensor.data concatenates its fields on the first reward read.
        # A reset of any environment invalidates this cache for the entire batch;
        # otherwise the critic sees the pre-forward contact sample from rewards.
        self._foot_sensor_cache = self.sim.foot_contacts
        self.check_termination()
        self.compute_reward()
        self._update_diagnostics()
        rewards = torch.from_numpy(self.rew_buf.astype(np.float32))
        dones = torch.from_numpy(self.reset_buf.copy())
        extras = {"time_outs": torch.from_numpy(self.time_out_buf.copy())}
        ids = np.flatnonzero(self.reset_buf)
        if ids.size:
            extras["log"] = self._episode_metrics(ids)
            self.reset_idx(ids)
        # mjlab calls full forward once here, then commands and interval events.
        self.sim.forward()
        self._refresh_state()
        self._next_command -= self.dt
        self._resample_commands(np.flatnonzero(self._next_command <= 0))
        self._update_commands()
        self._push_robots()
        # Push writes qvel but does not forward again in the reference: builtin
        # velocity sensors retain the pre-push values until the next physics tick.
        self.compute_observations()
        if not self.headless:
            self.render()
        return self.obs_buf, rewards, dones, extras

    def _refresh_state(self):
        self.base_pos = self.sim.xpos[:, self.sim.root_id].copy()
        self.base_quat = self.sim.xquat[:, self.sim.root_id].copy()
        self.base_lin_vel_w = self.sim.sensor("root_link_velocity").copy()
        self.base_lin_vel = rotate_inverse(self.base_quat, self.base_lin_vel_w)
        self.base_ang_vel_w = self.sim.sensor("root_link_angular_velocity").copy()
        self.base_ang_vel = rotate_inverse(self.base_quat, self.base_ang_vel_w)
        self.projected_gravity = rotate_inverse(self.base_quat, np.broadcast_to([0, 0, -1], (self.num_envs, 3)))
        self.dof_pos = self.sim.qpos[:, self.sim.qpos_ids]
        self.dof_vel = self.sim.qvel[:, self.sim.qvel_ids]
        self.dof_acc[:] = self.sim.qacc[:, self.sim.qvel_ids]
        self.feet_pos = self.sim.site_xpos[:, self.sim.site_ids].copy()
        self.feet_vel = self.sim.sensordata[:, self.sim.site_velocity_columns].copy()
        self.base_height = self.base_pos[:, 2]

    def _update_contacts(self):
        contacts = self.sim.foot_contacts[:, :, 0] > 0
        liftoff = (self.feet_contact_time > 0) & ~contacts
        self._liftoff_count += liftoff.sum(axis=1)
        self.feet_air_time[:] = np.where(contacts, 0, self.feet_air_time + self.cfg.sim.dt)
        self.feet_contact_time[:] = np.where(contacts, self.feet_contact_time + self.cfg.sim.dt, 0)
        self._feet_contact[:] = contacts
        self._collision_history[self._contact_cursor] = self.sim.history_contacts[:, :, 1:]
        self._contact_cursor = (self._contact_cursor + 1) % self.cfg.rewards.contact_history_length

    def reset_idx(self, ids):
        ids = np.asarray(ids, dtype=int)
        if not ids.size:
            return
        self._foot_sensor_cache = None
        self._update_command_curriculum(ids)
        self.sim.reset(ids)
        self.sim.qpos[ids, :3] = self.cfg.init_state.pos
        self.sim.qpos[ids, :2] += self.rng.uniform(-0.5, 0.5, (len(ids), 2))
        yaw = self.rng.uniform(-3.14, 3.14, len(ids))
        self.sim.qpos[ids, 3:7] = np.stack(
            (np.cos(yaw / 2), np.zeros(len(ids)), np.zeros(len(ids)), np.sin(yaw / 2)), axis=1
        )
        self.sim.qpos[np.ix_(ids, self.sim.qpos_ids)] = np.clip(
            self.default_dof_pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1]
        )
        self.sim.qvel[ids] = 0
        for values in (
            self.actions,
            self.last_actions,
            self.feet_air_time,
            self.feet_contact_time,
            self._feet_contact,
        ):
            values[ids] = 0
        self._collision_history[:, ids] = 0
        self._resample_commands(ids)
        self._diagnostic_steps[ids] = 0
        self._liftoff_count[ids] = 0
        for values in (*self.episode_sums.values(), *self._diagnostic_sums.values()):
            values[ids] = 0
        self.episode_length_buf[ids] = 0
        if not self._push_initialized:
            # mjlab 1.2 does not reset timers of function-based interval events.
            self._next_push[:] = self.rng.uniform(*self.cfg.domain_rand.push_interval_s, self.num_envs)
            self._push_initialized = True

    def _resample_commands(self, ids):
        if not len(ids):
            return
        self._next_command[ids] = self.rng.uniform(*self.cfg.commands.resampling_time_range, len(ids))
        self.commands[ids] = self.rng.uniform(self.command_ranges[:, 0], self.command_ranges[:, 1], (len(ids), 3))
        self.commands[ids] *= (np.linalg.norm(self.commands[ids], axis=1) > 0.1)[:, None]
        self.heading_targets[ids] = self.rng.uniform(*self.cfg.commands.heading_range, len(ids))
        self.heading_envs[ids] = self.rng.random(len(ids)) <= self.cfg.commands.heading_fraction
        self.standing_envs[ids] = self.rng.random(len(ids)) <= self.cfg.commands.standing_fraction

    def _update_commands(self):
        if self.cfg.commands.heading_command:
            w, x, y, z = self.base_quat.T
            heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            error = (self.heading_targets - heading + np.pi) % (2 * np.pi) - np.pi
            ids = self.heading_envs
            self.commands[ids, 2] = np.clip(self.cfg.commands.heading_stiffness * error[ids], *self.command_ranges[2])
        self.commands[self.standing_envs] = 0

    def _update_command_curriculum(self, ids):
        if self.cfg.commands.curriculum and self.common_step_counter > 0:
            self.command_ranges[:] = (
                self.cfg.commands.limit_ranges
                if self.common_step_counter > self.cfg.commands.expansion_step
                else self.cfg.commands.initial_stage_ranges
            )
        return {}

    def _push_robots(self):
        if not self.cfg.domain_rand.push_robots:
            return
        self._next_push -= self.dt
        ids = np.flatnonzero(self._next_push < 1e-6)
        if not ids.size:
            return
        limits = np.asarray(self.cfg.domain_rand.push_velocity_ranges)
        delta = self.rng.uniform(limits[:, 0], limits[:, 1], (len(ids), 6))
        self.sim.qvel[ids, :3] = self.base_lin_vel_w[ids] + delta[:, :3]
        angular_w = self.base_ang_vel_w[ids] + delta[:, 3:]
        self.sim.qvel[ids, 3:6] = rotate_inverse(self.base_quat[ids], angular_w)
        self._next_push[ids] = self.rng.uniform(*self.cfg.domain_rand.push_interval_s, len(ids))

    def reset(self):
        self.reset_idx(np.arange(self.num_envs))
        self.sim.forward()
        self._refresh_state()
        # CommandTerm.reset samples but does not apply heading/standing until compute.
        self.compute_observations()
        return self.obs_buf

    def check_termination(self):
        self._tilt_failure = np.arccos(np.clip(-self.projected_gravity[:, 2], -1, 1)) > self.cfg.termination.max_tilt
        threshold = self.cfg.termination.illegal_contact_threshold
        self._contact_failure = (
            np.linalg.norm(self._collision_history, axis=-1).max(axis=(0, 2)) > threshold
            if threshold is not None
            else np.zeros(self.num_envs, dtype=bool)
        )
        self.terminated_buf = self._tilt_failure | self._contact_failure
        self.time_out_buf = self.episode_length_buf.numpy() >= self.max_episode_length
        self.reset_buf = self.terminated_buf | self.time_out_buf

    def compute_observations(self, ids=None):
        angle = (
            (self.episode_length_buf.numpy() * self.dt)
            % self.cfg.rewards.gait_period
            / self.cfg.rewards.gait_period
            * 2
            * np.pi
        )
        phase = np.stack((np.sin(angle), np.cos(angle)), axis=1)
        phase[np.linalg.norm(self.commands, axis=1) < 0.1] = 0
        clean = np.concatenate(
            (
                self.sim.sensor("imu_ang_vel"),
                self.projected_gravity,
                self.commands,
                phase,
                self.dof_pos - self.default_dof_pos,
                self.dof_vel,
                self.actions,
            ),
            axis=1,
        )
        actor = clean.copy()
        if self.cfg.noise.add_noise:
            cfg = self.cfg.noise
            scale = np.r_[
                np.full(3, cfg.ang_vel),
                np.full(3, cfg.gravity),
                np.zeros(5),
                np.full(self.num_actions, cfg.joint_pos),
                np.full(self.num_actions, cfg.joint_vel),
                np.zeros(self.num_actions),
            ]
            actor += self.rng.uniform(-1, 1, actor.shape) * scale
        if self._foot_sensor_cache is None:
            self._foot_sensor_cache = self.sim.foot_contacts
        contacts = self._foot_sensor_cache
        forces = contacts[:, :, 1:].reshape(self.num_envs, -1)
        critic = np.concatenate(
            (
                clean,
                self.sim.sensor("imu_lin_vel"),
                self.feet_pos[:, :, 2],
                self.feet_air_time,
                (contacts[:, :, 0] > 0).astype(float),
                np.sign(forces) * np.log1p(np.abs(forces)),
            ),
            axis=1,
        )
        if not np.isfinite(actor).all() or not np.isfinite(critic).all():
            raise FloatingPointError("Nonfinite mjlab observation")
        self.obs_buf = TensorDict(
            {
                "policy": torch.from_numpy(actor.astype(np.float32)),
                "critic": torch.from_numpy(critic.astype(np.float32)),
            },
            batch_size=[self.num_envs],
        )

    def _moving(self):
        return (
            np.linalg.norm(self.commands[:, :2], axis=1) + np.abs(self.commands[:, 2])
            > self.cfg.rewards.command_threshold
        )

    def _reward_tracking_lin_vel(self):
        error = (
            np.square(self.commands[:, :2] - self.base_lin_vel[:, :2]).sum(axis=1) + 2 * self.base_lin_vel[:, 2] ** 2
        )
        return np.exp(-error / 0.25)

    def _reward_tracking_ang_vel(self):
        error = np.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) + 0.05 * np.square(
            self.base_ang_vel[:, :2]
        ).sum(axis=1)
        return np.exp(-error / 0.5)

    def _reward_pose(self):
        speed = np.linalg.norm(self.commands[:, :2], axis=1) + np.abs(self.commands[:, 2])
        std = np.where(
            (speed < 0.1)[:, None],
            self._pose_std[0],
            np.where((speed < 1.5)[:, None], self._pose_std[1], self._pose_std[2]),
        )
        return np.exp(-np.square((self.dof_pos - self.default_dof_pos) / std).mean(axis=1))

    def _reward_orientation(self):
        quat = self.sim.xquat[:, self.sim.reward_body_id]
        gravity = rotate_inverse(quat, np.broadcast_to([0, 0, -1], (self.num_envs, 3)))
        return np.square(gravity[:, :2]).sum(axis=1)

    def _reward_body_ang_vel(self):
        velocity = (
            self.sim.sensor("reward_body_angular_velocity")
            if self.sim.reward_body_id != self.sim.root_id
            else self.base_ang_vel_w
        )
        return np.square(velocity[:, :2]).sum(axis=1)

    def _reward_self_collisions(self):
        # Source sensor stores its first matching self-contact, not a count of all
        # simultaneous contact pairs. Count physics samples above the threshold.
        hit = np.linalg.norm(self._collision_history, axis=-1) > self.cfg.rewards.self_collision_threshold
        return hit.any(axis=2).sum(axis=0).astype(float)

    def _reward_angular_momentum(self):
        return np.square(self.sim.sensor("root_angmom")).sum(axis=1)

    def _reward_is_terminated(self):
        return self.terminated_buf.astype(float)

    def _reward_dof_acc(self):
        return np.square(self.dof_acc).sum(axis=1)

    def _reward_foot_gait(self):
        phase = (self.episode_length_buf.numpy() * self.dt / self.cfg.rewards.gait_period)[:, None]
        stance = (phase + self.cfg.rewards.gait_offsets) % 1 < self.cfg.rewards.gait_stance_fraction
        return (stance == (self.feet_contact_time > 0)).mean(axis=1) * self._moving()

    def _reward_foot_clearance(self):
        return (
            np.abs(self.feet_pos[:, :, 2] - self.cfg.rewards.feet_clearance_target)
            * np.linalg.norm(self.feet_vel[:, :, :2], axis=-1)
        ).sum(axis=1) * self._moving()

    def _reward_foot_slip(self):
        return (np.square(self.feet_vel[:, :, :2]).sum(axis=-1) * (self.sim.foot_contacts[:, :, 0] > 0)).sum(
            axis=1
        ) * self._moving()

    def _reward_soft_landing(self):
        first = (self.feet_contact_time > 0) & (self.feet_contact_time < self.dt + 1e-8)
        return (np.linalg.norm(self.sim.foot_contacts[:, :, 1:], axis=-1) * first).sum(axis=1) * self._moving()

    def _reward_stand_still(self):
        return np.square(self.dof_pos - self.default_dof_pos).sum(axis=1) * ~self._moving()

    def _reward_action_rate(self):
        return np.square(self.actions - self.last_actions).sum(axis=1)

    def _reward_dof_pos_limits(self):
        return (
            np.maximum(self.dof_pos_limits[:, 0] - self.dof_pos, 0)
            + np.maximum(self.dof_pos - self.dof_pos_limits[:, 1], 0)
        ).sum(axis=1)

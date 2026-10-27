"""Class-based legged_gym task on MuJoCo CPU batches.

Simulation, rewards and history stay on CPU. rsl-rl transfers the compact
TensorDict to the learner device (MPS, CUDA or CPU) once per control step.
"""

import math
import re

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from legged_gym.sim.mujoco_batch import MujocoBatch, joint_parameters


def rotate_inverse(q, v):
    """Rotate world vectors into a body's frame; MuJoCo quaternions are wxyz."""
    xyz = q[:, 1:4]
    uv = np.cross(xyz, v)
    return v - 2 * q[:, :1] * uv + 2 * np.cross(xyz, uv)


class LeggedRobot(VecEnv):
    def __init__(self, cfg, sim_params=None, physics_engine="mujoco", sim_device="cpu", headless=True):
        if physics_engine != "mujoco" or str(sim_device) != "cpu":
            raise ValueError("MuJoCo batch physics requires physics_engine='mujoco', sim_device='cpu'")
        if cfg.env.num_envs <= 0 or cfg.sim.dt <= 0 or cfg.control.decimation < 1:
            raise ValueError("num_envs, sim.dt and decimation must be positive")
        if cfg.env.history_length < 1 or cfg.rewards.contact_history_length < 1 or cfg.env.episode_length_s <= 0:
            raise ValueError("Observation/contact history lengths and episode length must be positive")
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.headless = headless
        self.num_envs = cfg.env.num_envs
        self.num_actions = cfg.env.num_actions
        self.dt = cfg.sim.dt * cfg.control.decimation
        self.max_episode_length = math.ceil(cfg.env.episode_length_s / self.dt)
        self.rng = np.random.default_rng(cfg.seed)
        self.sim = MujocoBatch(cfg, self.rng)
        self.terrain = self.sim.terrain
        self.custom_origins = self.terrain.type != "plane"
        self.terrain_levels = np.zeros(self.num_envs, dtype=int)
        self.terrain_types = np.zeros(self.num_envs, dtype=int)
        self.env_origins = np.zeros((self.num_envs, 3))
        if self.custom_origins:
            highest = cfg.terrain.max_init_terrain_level if cfg.terrain.curriculum else cfg.terrain.num_rows - 1
            if highest < 0:
                raise ValueError("max_init_terrain_level must be nonnegative")
            self.terrain_levels[:] = self.rng.integers(0, min(highest, cfg.terrain.num_rows - 1) + 1, self.num_envs)
            self.terrain_types[:] = np.minimum(
                np.arange(self.num_envs) * cfg.terrain.num_cols // self.num_envs, cfg.terrain.num_cols - 1
            )
            self._update_env_origins()
        x, y = np.meshgrid(cfg.terrain.measured_points_x, cfg.terrain.measured_points_y, indexing="ij")
        self.height_points = np.stack((x.ravel(), y.ravel()), axis=-1)
        if cfg.terrain.measure_heights and (not self.height_points.size or not np.isfinite(self.height_points).all()):
            raise ValueError("Height scan requires finite, nonempty measured_points_x/y")
        if (
            cfg.observations.actor_height_scan or cfg.observations.critic_height_scan
        ) and not cfg.terrain.measure_heights:
            raise ValueError("Height scan observations require terrain.measure_heights")
        self.dof_names = list(cfg.asset.joint_names)
        if len(self.dof_names) != self.num_actions or len(set(self.dof_names)) != self.num_actions:
            raise ValueError("asset.joint_names must contain one unique name per action")
        self.default_dof_pos = joint_parameters(self.dof_names, cfg.init_state.default_joint_angles, 0.0)
        self.p_gains = joint_parameters(self.dof_names, cfg.control.stiffness)
        self.d_gains = joint_parameters(self.dof_names, cfg.control.damping)
        self.torque_limits = joint_parameters(self.dof_names, cfg.control.effort_limits)
        limits = self.sim.model.jnt_range[self.sim.joint_ids]
        centers = limits.mean(axis=1)
        half_ranges = (limits[:, 1] - limits[:, 0]) * cfg.rewards.soft_dof_pos_limit / 2
        self.dof_pos_limits = np.stack((centers - half_ranges, centers + half_ranges), axis=1)
        self.feet_indices = self.sim.feet_ids - 1  # Contact buffers omit the world body.
        self._root_contact_index = self.sim.model.body(cfg.asset.root_body).id - 1
        self._penalty_indices = np.array(
            [
                i
                for i, name in enumerate(self.sim.body_names)
                if any(re.fullmatch(p, name) for p in cfg.asset.penalize_contacts_on)
            ],
            dtype=int,
        )
        self._arm_indices = self._joint_indices(r".*_(shoulder_.*|elbow|wrist_.*)_joint")
        self._waist_indices = self._joint_indices(r"waist_.*")
        self._leg_indices = self._joint_indices(r".*_hip_(roll|yaw)_joint")
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.common_step_counter = 0
        shape = (self.num_envs, self.num_actions)
        self.actions = np.zeros(shape)
        self.last_actions = np.zeros(shape)
        self.last_dof_vel = np.zeros(shape)
        self.torques = np.zeros(shape)
        self.commands = np.zeros((self.num_envs, 3))
        self.command_ranges = np.array(cfg.commands.ranges, dtype=float, copy=True)
        self._next_push = np.zeros(self.num_envs)
        feet_shape = (self.num_envs, len(self.feet_indices))
        self.feet_air_time = np.zeros(feet_shape)
        self.feet_contact_time = np.zeros(feet_shape)
        self.last_air_time = np.zeros(feet_shape)
        self.last_contact_time = np.zeros(feet_shape)
        self._feet_contact = np.zeros(feet_shape, dtype=bool)
        self._first_contact = np.zeros(feet_shape, dtype=bool)
        self._touchdown_air_time = np.zeros(feet_shape)
        self._contact_history = np.zeros(
            (cfg.rewards.contact_history_length, self.num_envs, self.sim.model.nbody - 1, 3)
        )
        self._contact_cursor = 0
        self._prepare_reward_function()
        self.episode_sums = {name: np.zeros(self.num_envs) for name in self.reward_scales}
        self._history = {}
        self.obs_buf = None
        self._viewer = None
        self._viewer_model = None
        self._viewer_data = None
        self._closed = False
        self.reset()
        self.num_obs = self.obs_buf["policy"].shape[-1]
        self.num_privileged_obs = self.obs_buf["critic"].shape[-1]

    def _joint_indices(self, pattern):
        return np.array([i for i, name in enumerate(self.dof_names) if re.fullmatch(pattern, name)], dtype=int)

    def _prepare_reward_function(self):
        self.reward_scales = {name: value for name, value in self.cfg.rewards.scales.items() if value != 0}
        self.reward_functions = {name: getattr(self, f"_reward_{name}") for name in self.reward_scales}

    def _compute_torques(self, actions):
        target = self.default_dof_pos + self.cfg.control.action_scale * actions
        pos = self.sim.qpos[:, self.sim.qpos_ids]
        vel = self.sim.qvel[:, self.sim.qvel_ids]
        torque = self.p_gains * (target - pos) - self.d_gains * vel
        limits = self.torque_limits
        if self.cfg.control.torque_speed_curve:
            # Unitree Go2HV torque-speed envelope, recalculated every physics tick.
            limits = np.where(vel * torque > 0, 20.2, 23.4)
            limits *= np.clip((30.0 - np.abs(vel)) / (30.0 - 13.5), 0.0, 1.0)
            limits = np.minimum(limits, self.torque_limits)
        return np.clip(torque, -limits, limits)

    def step(self, actions):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        actions = np.asarray(actions)
        if actions.shape != self.actions.shape or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions of shape {self.actions.shape}")
        self.actions[:] = np.clip(actions, -self.cfg.control.clip_actions, self.cfg.control.clip_actions)
        self._push_robots()
        self._first_contact.fill(False)
        self._touchdown_air_time.fill(0.0)
        for substep in range(self.cfg.control.decimation):
            self.torques[:] = self._compute_torques(self.actions)
            self.sim.step(self.torques, observe=substep == self.cfg.control.decimation - 1)
            self._update_contacts()
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._refresh_state()
        self.check_termination()
        self.compute_reward()
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        # Reward uses the command that produced the transition. Resample only
        # afterward so the next observation and next action see the same command.
        resample_steps = max(1, round(self.cfg.commands.resampling_time / self.dt))
        ids = np.flatnonzero((self.episode_length_buf.numpy() % resample_steps == 0) & ~self.reset_buf)
        self._resample_commands(ids)
        self.compute_observations()
        dones = torch.from_numpy(self.reset_buf.copy())
        rewards = torch.from_numpy(self.rew_buf.astype(np.float32))
        extras = {"time_outs": torch.from_numpy(self.time_out_buf.copy())}
        reset_ids = np.flatnonzero(self.reset_buf)
        if reset_ids.size:
            extras["terminal_observations"] = self.obs_buf[torch.from_numpy(reset_ids)].clone()
            extras["terminal_env_ids"] = torch.from_numpy(reset_ids)
            extras["log"] = self._episode_metrics(reset_ids)
            self._update_terrain_curriculum(reset_ids)
            self._update_command_curriculum(reset_ids)
            self.reset_idx(reset_ids)
            self._refresh_state()
            self.compute_observations(reset_ids)
        if not self.headless:
            self.render()
        return self.obs_buf, rewards, dones, extras

    def _update_contacts(self):
        forces = self.sim.contact_forces
        self._contact_history[self._contact_cursor] = forces
        self._contact_cursor = (self._contact_cursor + 1) % self.cfg.rewards.contact_history_length
        contacts = np.linalg.norm(forces[:, self.feet_indices], axis=-1) > self.cfg.rewards.contact_threshold
        landing = contacts & ~self._feet_contact
        liftoff = ~contacts & self._feet_contact
        self.last_air_time[landing] = self.feet_air_time[landing]
        self.last_contact_time[liftoff] = self.feet_contact_time[liftoff]
        self._touchdown_air_time[landing] = self.feet_air_time[landing]
        self._first_contact |= landing
        self.feet_air_time[:] = np.where(contacts, 0.0, self.feet_air_time + self.cfg.sim.dt)
        self.feet_contact_time[:] = np.where(contacts, self.feet_contact_time + self.cfg.sim.dt, 0.0)
        self._feet_contact[:] = contacts

    def _refresh_state(self):
        self.base_pos = self.sim.qpos[:, :3].copy()
        self.base_quat = self.sim.qpos[:, 3:7].copy()
        self.base_lin_vel = rotate_inverse(self.base_quat, self.sim.qvel[:, :3])
        self.base_ang_vel = self.sim.qvel[:, 3:6].copy()
        self.projected_gravity = rotate_inverse(self.base_quat, np.broadcast_to([0.0, 0.0, -1.0], (self.num_envs, 3)))
        self.dof_pos = self.sim.qpos[:, self.sim.qpos_ids]
        self.dof_vel = self.sim.qvel[:, self.sim.qvel_ids]
        self.feet_pos = self.sim.xpos[:, self.sim.feet_ids]
        self.feet_vel = self.sim.feet_velocities
        self.contact_forces = self.sim.contact_forces
        self._contact_peak = np.linalg.norm(self._contact_history, axis=-1).max(axis=0)
        self.base_ground_height = self.terrain.heights_at(self.base_pos[:, :2])
        self.base_height = self.base_pos[:, 2] - self.base_ground_height
        self.feet_ground_height = self.terrain.heights_at(self.feet_pos[:, :, :2])
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

    def _get_heights(self):
        # Rotate the scan by root yaw only, as in legged_gym, rather than roll/pitch.
        w, x, y, z = self.base_quat.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        px, py = self.height_points.T
        points = np.stack(
            (
                np.cos(yaw)[:, None] * px - np.sin(yaw)[:, None] * py,
                np.sin(yaw)[:, None] * px + np.cos(yaw)[:, None] * py,
            ),
            axis=-1,
        )
        points += self.base_pos[:, None, :2]
        return self.terrain.heights_at(points)

    def check_termination(self):
        tilt = np.arccos(np.clip(-self.projected_gravity[:, 2], -1.0, 1.0))
        self.terminated_buf = (tilt > self.cfg.termination.max_tilt) | (
            self.base_height < self.cfg.termination.min_height
        )
        self.terminated_buf |= ~self.terrain.contains(self.base_pos[:, :2])
        if self.cfg.termination.base_contact:
            self.terminated_buf |= self._contact_peak[:, self._root_contact_index] > self.cfg.rewards.contact_threshold
        finite = np.isfinite(self.sim.qpos).all(axis=1) & np.isfinite(self.sim.qvel).all(axis=1)
        if not finite.all():
            raise FloatingPointError(f"Nonfinite MuJoCo state in environments {np.flatnonzero(~finite).tolist()}")
        # Failure and time limit occurring together are true terminations.
        self.time_out_buf = (self.episode_length_buf.numpy() >= self.max_episode_length) & ~self.terminated_buf
        self.reset_buf = self.terminated_buf | self.time_out_buf

    def compute_reward(self):
        self.rew_buf = np.zeros(self.num_envs)
        for name, function in self.reward_functions.items():
            value = function() * self.reward_scales[name] * self.dt
            self.rew_buf += value
            self.episode_sums[name] += value
        if not np.isfinite(self.rew_buf).all():
            raise FloatingPointError("Nonfinite reward")

    def _observation_terms(self, critic):
        terms = {}
        if critic:
            terms["lin_vel"] = self.base_lin_vel
        terms.update(
            ang_vel=self.base_ang_vel * 0.2,
            gravity=self.projected_gravity,
            commands=self.commands,
            joint_pos=self.dof_pos - self.default_dof_pos,
            joint_vel=self.dof_vel * 0.05,
        )
        if critic and self.cfg.observations.critic_joint_effort:
            terms["effort"] = self.torques * 0.01
        terms["actions"] = self.actions
        use_heights = self.cfg.observations.critic_height_scan if critic else self.cfg.observations.actor_height_scan
        if use_heights:
            heights = self.base_pos[:, 2:3] - self.measured_heights - self.cfg.terrain.height_observation_offset
            terms["height_scan"] = np.clip(heights, -1.0, 1.0) * self.cfg.terrain.height_observation_scale
        return terms

    def compute_observations(self, ids=None):
        if ids is None:
            ids = np.arange(self.num_envs)
        data = {}
        for group in ("policy", "critic"):
            terms = self._observation_terms(critic=group == "critic")
            flattened = []
            for name, value in terms.items():
                value = value[ids].copy()
                if group == "policy" and self.cfg.noise.add_noise:
                    noise = {
                        "ang_vel": self.cfg.noise.ang_vel * 0.2,
                        "gravity": self.cfg.noise.gravity,
                        "joint_pos": self.cfg.noise.joint_pos,
                        "joint_vel": self.cfg.noise.joint_vel * 0.05,
                        "height_scan": self.cfg.noise.height_measurements * self.cfg.terrain.height_observation_scale,
                    }.get(name, 0.0)
                    if noise:
                        value += self.rng.uniform(-noise, noise, value.shape)
                key = (group, name)
                if key not in self._history:
                    self._history[key] = np.zeros(
                        (self.num_envs, self.cfg.env.history_length, value.shape[-1]), dtype=np.float32
                    )
                history = self._history[key]
                history[ids, :-1] = history[ids, 1:]
                history[ids, -1] = np.clip(value, -100.0, 100.0)
                # Isaac Lab flattens history within each term, oldest first.
                flattened.append(history.reshape(self.num_envs, -1))
            data[group] = torch.from_numpy(np.concatenate(flattened, axis=-1))
        self.obs_buf = TensorDict(data, batch_size=[self.num_envs])

    def get_observations(self):
        return self.obs_buf

    def reset(self):
        self.reset_idx(np.arange(self.num_envs))
        self._refresh_state()
        self.compute_observations()
        return self.get_observations()

    def reset_idx(self, ids):
        ids = np.asarray(ids, dtype=int)
        if not ids.size:
            return
        self.sim.reset(ids)
        self.sim.qpos[ids, :3] = self.env_origins[ids] + self.cfg.init_state.pos
        self.sim.qpos[ids, :2] += self.rng.uniform(-0.5, 0.5, (len(ids), 2))
        yaw = self.rng.uniform(-np.pi, np.pi, len(ids))
        self.sim.qpos[ids, 3:7] = np.stack(
            (np.cos(yaw / 2), np.zeros(len(ids)), np.zeros(len(ids)), np.sin(yaw / 2)), axis=-1
        )
        self.sim.qpos[np.ix_(ids, self.sim.qpos_ids)] = self.default_dof_pos
        self.sim.qvel[np.ix_(ids, self.sim.qvel_ids)] = self.rng.uniform(
            *self.cfg.init_state.joint_velocity_range, (len(ids), self.num_actions)
        )
        self.actions[ids] = 0
        self.last_actions[ids] = 0
        self.torques[ids] = 0
        self.last_dof_vel[ids] = self.sim.qvel[np.ix_(ids, self.sim.qvel_ids)]
        self.episode_length_buf[ids] = 0
        self._resample_commands(ids)
        self._next_push[ids] = self.rng.uniform(*self.cfg.domain_rand.push_interval_s, len(ids))
        self._contact_history[:, ids] = 0
        for buf in (
            self.feet_air_time,
            self.feet_contact_time,
            self.last_air_time,
            self.last_contact_time,
            self._feet_contact,
            self._first_contact,
            self._touchdown_air_time,
        ):
            buf[ids] = 0
        for history in self._history.values():
            history[ids] = 0
        for values in self.episode_sums.values():
            values[ids] = 0
        self.sim.forward(ids)

    def _resample_commands(self, ids):
        if len(ids):
            self.commands[ids] = self.rng.uniform(self.command_ranges[:, 0], self.command_ranges[:, 1], (len(ids), 3))
            standing = self.rng.random(len(ids)) < self.cfg.commands.standing_fraction
            self.commands[ids[standing]] = 0.0

    def _push_robots(self):
        if not self.cfg.domain_rand.push_robots:
            return
        self._next_push -= self.dt
        ids = np.flatnonzero(self._next_push <= 0)
        if ids.size:
            limit = self.cfg.domain_rand.max_push_vel_xy
            self.sim.qvel[ids, :2] = self.rng.uniform(-limit, limit, (len(ids), 2))
            self._next_push[ids] = self.rng.uniform(*self.cfg.domain_rand.push_interval_s, len(ids))

    def _episode_metrics(self, ids):
        return {
            **{
                f"Reward/{name}": float(values[ids].mean() / self.cfg.env.episode_length_s)
                for name, values in self.episode_sums.items()
            },
            "Episode/length_s": float(self.episode_length_buf.numpy()[ids].mean() * self.dt),
            "Episode/time_out_fraction": float(self.time_out_buf[ids].mean()),
            "Curriculum/max_command_x": float(self.command_ranges[0, 1]),
            **({"Curriculum/terrain_level": float(self.terrain_levels[ids].mean())} if self.custom_origins else {}),
        }

    def _update_env_origins(self):
        self.env_origins[:] = self.terrain.env_origins[self.terrain_levels, self.terrain_types]

    def _update_terrain_curriculum(self, ids):
        if not self.custom_origins or not self.cfg.terrain.curriculum:
            return
        distance = np.linalg.norm(self.base_pos[ids, :2] - self.env_origins[ids, :2], axis=1)
        move_up = distance > self.cfg.terrain.terrain_length / 2
        expected = np.linalg.norm(self.commands[ids, :2], axis=1) * self.cfg.env.episode_length_s * 0.5
        move_down = (distance < expected) & ~move_up
        levels = self.terrain_levels[ids] + move_up.astype(int) - move_down.astype(int)
        mastered = levels >= self.cfg.terrain.num_rows
        levels[mastered] = self.rng.integers(0, self.cfg.terrain.num_rows, mastered.sum())
        self.terrain_levels[ids] = np.maximum(levels, 0)
        self._update_env_origins()

    def _update_command_curriculum(self, ids):
        if self.cfg.commands.curriculum and self.common_step_counter % self.max_episode_length == 0:
            score = self.episode_sums["tracking_lin_vel"][ids].mean() / self.cfg.env.episode_length_s
            if score > self.reward_scales["tracking_lin_vel"] * self.cfg.commands.curriculum_threshold:
                limits = np.asarray(self.cfg.commands.limit_ranges)
                delta = np.array([-1.0, 1.0]) * self.cfg.commands.curriculum_step
                self.command_ranges[:2] = np.clip(self.command_ranges[:2] + delta, limits[:2, :1], limits[:2, 1:])

    def training_state_dict(self):
        state = {"command_ranges": self.command_ranges.tolist(), "common_step_counter": self.common_step_counter}
        if self.custom_origins:
            state["terrain"] = {"levels": self.terrain_levels.tolist(), "types": self.terrain_types.tolist()}
        return state

    def load_training_state_dict(self, state):
        ranges = np.asarray(state["command_ranges"], dtype=float)
        if ranges.shape != (3, 2) or not np.isfinite(ranges).all() or np.any(ranges[:, 0] > ranges[:, 1]):
            raise ValueError("Invalid command curriculum state")
        self.command_ranges[:] = ranges
        self.common_step_counter = int(state["common_step_counter"])
        if self.custom_origins:
            terrain = state.get("terrain")
            if terrain is None:
                raise ValueError("Rough checkpoint is missing terrain curriculum state")
            levels = np.asarray(terrain["levels"], dtype=int)
            types = np.asarray(terrain["types"], dtype=int)
            if (
                levels.ndim != 1
                or not levels.size
                or types.shape != levels.shape
                or np.any(levels < 0)
                or np.any(levels >= self.cfg.terrain.num_rows)
                or np.any(types < 0)
                or np.any(types >= self.cfg.terrain.num_cols)
            ):
                raise ValueError("Invalid terrain curriculum state")
            if len(levels) == self.num_envs:
                self.terrain_levels[:] = levels
                self.terrain_types[:] = types
            else:
                # Preserve the trained level distribution when collector count changes.
                self.terrain_levels[:] = levels[np.arange(self.num_envs) % len(levels)]
            self._update_env_origins()
            self.reset()
            return
        self._resample_commands(np.arange(self.num_envs))
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

    # Reward functions return unweighted per-environment values; dt is applied once.
    def _reward_tracking_lin_vel(self):
        velocity = self.base_lin_vel[:, :2]
        if self.cfg.rewards.tracking_frame == "yaw":
            w, x, y, z = self.base_quat.T
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            vx, vy = self.sim.qvel[:, :2].T
            velocity = np.stack((np.cos(yaw) * vx + np.sin(yaw) * vy, -np.sin(yaw) * vx + np.cos(yaw) * vy), axis=-1)
        return np.exp(-np.square(self.commands[:, :2] - velocity).sum(axis=1) / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        return np.exp(-np.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) / self.cfg.rewards.tracking_sigma)

    def _reward_alive(self):
        return (~self.terminated_buf).astype(float)

    def _reward_lin_vel_z(self):
        return np.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return np.square(self.base_ang_vel[:, :2]).sum(axis=1)

    def _reward_dof_vel(self):
        return np.square(self.dof_vel).sum(axis=1)

    def _reward_dof_acc(self):
        return np.square((self.dof_vel - self.last_dof_vel) / self.dt).sum(axis=1)

    def _reward_torques(self):
        return np.square(self.torques).sum(axis=1)

    def _reward_action_rate(self):
        return np.square(self.actions - self.last_actions).sum(axis=1)

    def _reward_dof_pos_limits(self):
        return (
            np.maximum(self.dof_pos_limits[:, 0] - self.dof_pos, 0)
            + np.maximum(self.dof_pos - self.dof_pos_limits[:, 1], 0)
        ).sum(axis=1)

    def _reward_energy(self):
        return np.abs(self.dof_vel * self.torques).sum(axis=1)

    def _reward_orientation(self):
        return np.square(self.projected_gravity[:, :2]).sum(axis=1)

    def _reward_base_height(self):
        return np.square(self.base_height - self.cfg.rewards.base_height_target)

    def _reward_joint_position(self):
        error = np.linalg.norm(self.dof_pos - self.default_dof_pos, axis=1)
        moving = (np.linalg.norm(self.commands, axis=1) > 0) | (
            np.linalg.norm(self.base_lin_vel[:, :2], axis=1) > self.cfg.rewards.velocity_threshold
        )
        return error * np.where(moving, 1.0, self.cfg.rewards.stand_still_scale)

    def _deviation(self, ids):
        return np.abs(self.dof_pos[:, ids] - self.default_dof_pos[ids]).sum(axis=1)

    def _reward_joint_deviation_arms(self):
        return self._deviation(self._arm_indices)

    def _reward_joint_deviation_waists(self):
        return self._deviation(self._waist_indices)

    def _reward_joint_deviation_legs(self):
        return self._deviation(self._leg_indices)

    def _reward_feet_air_time(self):
        return ((self._touchdown_air_time - self.cfg.rewards.feet_air_time_threshold) * self._first_contact).sum(
            axis=1
        ) * (np.linalg.norm(self.commands[:, :2], axis=1) > 0.1)

    def _reward_air_time_variance(self):
        return np.clip(self.last_air_time, 0, 0.5).var(axis=1, ddof=1) + np.clip(self.last_contact_time, 0, 0.5).var(
            axis=1, ddof=1
        )

    def _reward_feet_slide(self):
        contact = self._contact_peak[:, self.feet_indices] > self.cfg.rewards.contact_threshold
        return (np.linalg.norm(self.feet_vel[:, :, :2], axis=-1) * contact).sum(axis=1)

    def _reward_feet_clearance(self):
        error = np.square(self.feet_pos[:, :, 2] - self.feet_ground_height - self.cfg.rewards.feet_clearance_target)
        speed = np.tanh(self.cfg.rewards.feet_clearance_tanh_mult * np.linalg.norm(self.feet_vel[:, :, :2], axis=-1))
        return np.exp(-(error * speed).sum(axis=1) / self.cfg.rewards.feet_clearance_std)

    def _reward_gait(self):
        phase = (
            self.episode_length_buf.numpy()[:, None] * self.dt / self.cfg.rewards.gait_period
            + np.asarray(self.cfg.rewards.gait_offsets)
        ) % 1.0
        return ((phase < self.cfg.rewards.gait_stance_fraction) == self._feet_contact).sum(axis=1) * (
            np.linalg.norm(self.commands, axis=1) > 0.1
        )

    def _reward_collision(self):
        return (self._contact_peak[:, self._penalty_indices] > self.cfg.rewards.contact_threshold).sum(axis=1)

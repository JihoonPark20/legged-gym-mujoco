"""Regression tests for real MuJoCo batches and task transition semantics."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from legged_gym.envs import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot, rotate_inverse
from legged_gym.utils.mujoco_helpers import get_args


@pytest.fixture(params=["go2", "g1"])
def env(request):
    cfg, _ = task_registry.get_cfgs(request.param)
    cfg.env.num_envs = 4
    cfg.sim.nthread = 2
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    task = LeggedRobot(cfg)
    yield task
    task.close()


def test_registry_configs_are_independent():
    cfg, train = task_registry.get_cfgs("g1")
    cfg.commands.ranges[0][1] = 99
    cfg.rewards.scales["alive"] = 99
    train.policy.actor_hidden_dims[0] = 1
    fresh, fresh_train = task_registry.get_cfgs("g1")
    assert fresh.commands.ranges[0][1] == 0.1
    assert fresh.rewards.scales["alive"] == 0.15
    assert fresh_train.policy.actor_hidden_dims == [512, 256, 128]


def test_joint_order_and_real_batch_transition(env):
    assert env.sim.model.nu == env.num_actions
    for i, name in enumerate(env.dof_names):
        assert env.sim.model.actuator(i).name == name
        assert env.sim.model.actuator_trnid[i, 0] == env.sim.model.joint(name).id
    expected = {12: (45, 60), 29: (480, 495)}[env.num_actions]
    initial = env.get_observations().clone()
    obs, rew, done, extras = env.step(torch.zeros(env.num_envs, env.num_actions))
    assert obs["policy"].shape == (env.num_envs, expected[0])
    assert obs["critic"].shape == (env.num_envs, expected[1])
    assert obs["policy"].dtype == torch.float32
    assert rew.dtype == torch.float32 and done.dtype == torch.bool
    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all()
    assert not torch.equal(initial["critic"], obs["critic"])
    assert not extras["time_outs"].any()


def test_subset_reset_preserves_other_simulations_and_history(env):
    env.step(torch.zeros(env.num_envs, env.num_actions))
    before = env.sim.batch.state.copy()
    histories = {key: h.copy() for key, h in env._history.items()}
    env.reset_idx([1])
    env._refresh_state()
    env.compute_observations(np.array([1]))
    np.testing.assert_array_equal(before[[0, 2, 3]], env.sim.batch.state[[0, 2, 3]])
    for key, h in env._history.items():
        np.testing.assert_array_equal(h[[0, 2, 3]], histories[key][[0, 2, 3]])
        np.testing.assert_array_equal(h[1, :-1], 0)
    assert env.episode_length_buf.tolist() == [1, 0, 1, 1]
    assert not env._contact_history[:, 1].any()


def test_timeout_autoreset_and_terminal_observation(env):
    env.max_episode_length = 1
    obs, rew, done, extras = env.step(torch.zeros(env.num_envs, env.num_actions))
    assert done.all() and extras["time_outs"].all()
    assert env.episode_length_buf.sum() == 0
    assert extras["terminal_observations"]["policy"].shape == obs["policy"].shape
    assert not torch.equal(extras["terminal_observations"]["critic"], obs["critic"])
    assert torch.isfinite(rew).all()


def test_failure_is_not_bootstrapped_as_timeout(env):
    env.max_episode_length = 1
    env.sim.qpos[0, 3:7] = [np.cos(0.6), np.sin(0.6), 0, 0]  # 1.2 rad tilt.
    env.sim.forward([0])
    _, _, done, extras = env.step(torch.zeros(env.num_envs, env.num_actions))
    assert done[0] and not extras["time_outs"][0]
    assert extras["time_outs"][1:].all()


def test_contact_sensors_and_go2_foot_assignment(env):
    # Check native contact outputs against static robot weight after settling.
    if env.num_actions != 12:
        return
    for _ in range(100):
        env.step(torch.zeros(env.num_envs, env.num_actions))
    feet_force = env.contact_forces[:, env.feet_indices, 2].sum(axis=1)
    weight = env.sim.model.body_mass.sum() * 9.81
    np.testing.assert_allclose(feet_force, weight, rtol=0.1)
    assert env._reward_collision().sum() == 0
    for foot in env.cfg.asset.feet:
        leg = foot[:-5]
        geom = env.sim.model.geom(leg)
        assert geom.bodyid[0] == env.sim.model.body(foot).id


def test_pd_gains_limits_and_go2_speed_curve(env):
    assert not env.sim.model.jnt_actfrclimited.any()
    actions = np.full_like(env.actions, 100.0)
    torques = env._compute_torques(actions)
    assert np.all(np.abs(torques) <= env.torque_limits + 1e-10)
    if env.num_actions == 12:
        env.sim.qvel[:, env.sim.qvel_ids] = 30.0
        np.testing.assert_array_equal(env._compute_torques(actions), 0)
        env.sim.qvel[:, env.sim.qvel_ids] = 0.0
        env.torque_limits[:] = 5.0
        np.testing.assert_array_equal(env._compute_torques(actions), 5.0)
    else:
        for name in ("left_wrist_pitch_joint", "right_wrist_yaw_joint"):
            idx = env.dof_names.index(name)
            assert env.torque_limits[idx] == 5.0
            assert torques[0, idx] > 0


def test_native_actuation_uses_configured_limits_and_joint_mapping(env):
    actual = env.sim.batch.bind("qfrc_actuator")
    env.sim.ctrl[:] = env.torque_limits * 2
    env.sim.forward()
    expected = np.broadcast_to(env.torque_limits, (env.num_envs, env.num_actions))
    np.testing.assert_allclose(actual[:, env.sim.qvel_ids], expected)


def test_reward_scales_include_control_dt_once(env):
    env.reward_functions = {"tracking_lin_vel": lambda: np.ones(env.num_envs)}
    env.reward_scales = {"tracking_lin_vel": 1.5}
    env.compute_reward()
    np.testing.assert_allclose(env.rew_buf, 1.5 * env.dt)


def test_noise_is_actor_only_and_get_observations_is_read_only(env):
    env.cfg.noise.add_noise = True
    env.cfg.env.history_length = 1
    env._history.clear()
    env.compute_observations()
    first = env.get_observations().clone()
    assert torch.equal(first["policy"], env.get_observations()["policy"])
    env.compute_observations()
    assert torch.equal(first["critic"], env.get_observations()["critic"])
    assert not torch.equal(first["policy"], env.get_observations()["policy"])


def test_curriculum_limits_and_state_roundtrip(env):
    env.common_step_counter = env.max_episode_length
    env.episode_sums["tracking_lin_vel"][:] = env.reward_scales["tracking_lin_vel"] * env.cfg.env.episode_length_s
    for _ in range(30):
        env._update_command_curriculum(np.arange(env.num_envs))
    np.testing.assert_allclose(env.command_ranges[:2], env.cfg.commands.limit_ranges[:2])
    state = env.training_state_dict()
    env.command_ranges[:] = 0
    env.load_training_state_dict(state)
    np.testing.assert_allclose(env.command_ranges, state["command_ranges"])


def test_batch_is_identical_across_worker_counts():
    cfg, _ = task_registry.get_cfgs("go2")
    cfg.env.num_envs = 4
    cfg.domain_rand.push_robots = False
    one_cfg = deepcopy(cfg)
    one_cfg.sim.nthread = 1
    cfg.sim.nthread = 3
    one, many = LeggedRobot(one_cfg), LeggedRobot(cfg)
    try:
        actions = torch.full((4, 12), 0.1)
        for _ in range(10):
            obs1, rew1, done1, _ = one.step(actions)
            obs2, rew2, done2, _ = many.step(actions)
            np.testing.assert_array_equal(one.sim.batch.state, many.sim.batch.state)
            assert torch.equal(obs1["policy"], obs2["policy"])
            assert torch.equal(rew1, rew2) and torch.equal(done1, done2)
    finally:
        one.close()
        many.close()


def test_quaternion_world_to_body_convention():
    angle = np.pi / 2
    quat = np.array([[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]])
    np.testing.assert_allclose(rotate_inverse(quat, np.array([[1.0, 0.0, 0.0]])), [[0.0, -1.0, 0.0]], atol=1e-15)


def test_invalid_actions_do_not_advance_physics(env):
    before = env.sim.batch.state.copy()
    with pytest.raises(ValueError, match="finite actions"):
        env.step(torch.full_like(torch.zeros(env.num_envs, env.num_actions), float("nan")))
    np.testing.assert_array_equal(before, env.sim.batch.state)


def test_cli_keeps_physics_and_learner_devices_separate():
    args = get_args(["--task", "g1", "--rl_device", "cpu", "--num_envs", "8"])
    assert args.sim_device == "cpu" and args.physics_engine == "mujoco"
    with pytest.raises(SystemExit):
        get_args(["--sim_device", "mps"])

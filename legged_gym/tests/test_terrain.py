"""Procedural terrain must match MuJoCo collision geometry and reset semantics."""

from copy import deepcopy

import mujoco
import numpy as np
import pytest
import torch

from legged_gym.envs import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.sim.mujoco_batch import load_model
from legged_gym.utils.task_registry import policy_contract
from legged_gym.utils.terrain import Terrain


def small_cfg(task="go2_rough"):
    cfg, _ = task_registry.get_cfgs(task)
    cfg.env.num_envs = 4
    cfg.sim.nthread = 2
    cfg.terrain.num_rows = 3
    cfg.terrain.num_cols = 6
    cfg.terrain.terrain_length = 4.0
    cfg.terrain.terrain_width = 4.0
    cfg.terrain.border_size = 1.0
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.push_robots = False
    return cfg


@pytest.fixture(params=["go2_rough", "g1_rough"])
def rough_env(request):
    env = LeggedRobot(small_cfg(request.param))
    yield env
    env.close()


@pytest.mark.parametrize(
    "kind", ["flat", "slope", "rough", "rough_slope", "stairs_up", "stairs_down", "obstacles", "stepping_stones"]
)
def test_generated_surface_matches_native_mujoco_rays(kind):
    cfg = small_cfg()
    cfg.terrain.terrain_width = 6.0  # Non-square maps catch transposed hfield data.
    cfg.terrain.selected = True
    cfg.terrain.terrain_kwargs = {"type": kind}
    terrain = Terrain(cfg.terrain, 4, cfg.seed)
    model = load_model(cfg, terrain)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(123)
    low = -cfg.terrain.border_size + 0.01
    high = (
        np.array([cfg.terrain.num_rows * cfg.terrain.terrain_length, cfg.terrain.num_cols * cfg.terrain.terrain_width])
        + cfg.terrain.border_size
        - 0.01
    )
    points = rng.uniform(low, high, (80, 2))
    expected = terrain.heights_at(points)
    floor = model.geom("floor").id
    measured = np.array(
        [
            10 - mujoco.mj_rayHfield(model, data, floor, np.r_[point, 10.0], np.array([0.0, 0.0, -1.0]))
            for point in points
        ]
    )
    np.testing.assert_allclose(measured, expected, atol=2e-6)
    assert model.nhfield == 1
    if kind != "flat":
        assert np.ptp(terrain.height_field_raw) > 0


def test_terrain_is_seeded_and_selected_overrides_are_not_mutated():
    cfg = small_cfg().terrain
    cfg.selected = True
    cfg.terrain_kwargs = {"type": "rough", "min_height": -0.03, "max_height": 0.03}
    before = deepcopy(cfg.terrain_kwargs)
    first, second, other = Terrain(cfg, 4, 1), Terrain(cfg, 4, 1), Terrain(cfg, 4, 2)
    np.testing.assert_array_equal(first.height_field_raw, second.height_field_raw)
    assert not np.array_equal(first.height_field_raw, other.height_field_raw)
    assert cfg.terrain_kwargs == before


def test_slope_difficulty_and_flat_spawn_platform():
    cfg = small_cfg().terrain
    cfg.selected = True
    cfg.terrain_kwargs = {"type": "slope"}
    terrain = Terrain(cfg, 4)
    assert np.all(np.diff(np.abs(terrain.env_origins[:, 0, 2])) > 0)
    for origin in terrain.env_origins.reshape(-1, 3):
        corners = origin[:2] + np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_allclose(terrain.heights_at(corners), origin[2], atol=1e-12)


def test_rough_task_observations_and_subset_reset(rough_env):
    env = rough_env
    expected = {12: (232, 247), 29: (1415, 1430)}[env.num_actions]
    assert (env.num_obs, env.num_privileged_obs) == expected
    np.testing.assert_allclose(env.base_height, env.cfg.init_state.pos[2])
    assert env.measured_heights.shape == (4, 187)
    np.testing.assert_array_equal(env.sim.model.hfield_data, env.sim.batch.get_model(0).hfield_data)
    for _ in range(8):
        obs, rewards, _, _ = env.step(torch.zeros(4, env.num_actions))
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
    saved = {k: v.copy() for k, v in env._history.items()}
    state = env.sim.batch.state.copy()
    env.reset_idx([1])
    env._refresh_state()
    env.compute_observations(np.array([1]))
    np.testing.assert_array_equal(state[[0, 2, 3]], env.sim.batch.state[[0, 2, 3]])
    for key in (("policy", "height_scan"), ("critic", "height_scan")):
        np.testing.assert_array_equal(env._history[key][[0, 2, 3]], saved[key][[0, 2, 3]])
        np.testing.assert_array_equal(env._history[key][1, :-1], 0)


def test_curriculum_progress_demotion_bounds_and_subset(rough_env):
    env = rough_env
    env.terrain_levels[:] = 1
    env._update_env_origins()
    env.base_pos[:] = env.env_origins
    env.base_pos[0, 0] += env.cfg.terrain.terrain_length * 0.6
    env.commands[:] = [0.5, 0, 0]
    env._update_terrain_curriculum(np.array([0, 1]))
    assert env.terrain_levels.tolist() == [2, 0, 1, 1]
    env.base_pos[0, :2] = env.env_origins[0, :2] + [3, 0]
    env.base_pos[1, :2] = env.env_origins[1, :2]
    env._update_terrain_curriculum(np.array([0, 1]))
    assert 0 <= env.terrain_levels[0] < env.cfg.terrain.num_rows
    assert env.terrain_levels[1] == 0
    np.testing.assert_array_equal(env.env_origins, env.terrain.env_origins[env.terrain_levels, env.terrain_types])


def test_ground_relative_termination_and_height_reward():
    cfg = small_cfg("g1_rough")
    cfg.terrain.selected = True
    cfg.terrain.terrain_kwargs = {"type": "stairs_down", "step_height": -0.15}
    cfg.terrain.max_init_terrain_level = 2
    cfg.termination.min_height = 0.6
    env = LeggedRobot(cfg)
    try:
        assert np.all(env.base_ground_height < 0)
        assert np.all(env.base_pos[:, 2] < cfg.termination.min_height)
        env.check_termination()
        assert not env.terminated_buf.any()
        np.testing.assert_allclose(
            env._reward_base_height(), (cfg.init_state.pos[2] - cfg.rewards.base_height_target) ** 2
        )
        env.sim.qpos[0, 2] = env.base_ground_height[0] + 0.1
        env.sim.forward([0])
        env._refresh_state()
        env.check_termination()
        assert env.terminated_buf[0]
    finally:
        env.close()


def test_native_batch_supports_robot_weight_on_elevated_platform():
    cfg = small_cfg()
    cfg.terrain.selected = True
    cfg.terrain.terrain_kwargs = {"type": "stairs_up", "step_height": 0.12}
    env = LeggedRobot(cfg)
    try:
        assert np.all(env.base_ground_height > 0)
        resets = 0
        for _ in range(100):
            _, _, done, _ = env.step(torch.zeros(4, 12))
            resets += done.sum().item()
        assert resets == 0
        assert np.all(env.base_height > 0.2)
        supported_weight = env.contact_forces[:, env.feet_indices, 2].sum(axis=1)
        np.testing.assert_allclose(supported_weight, env.sim.model.body_mass.sum() * 9.81, rtol=0.1)
    finally:
        env.close()


def test_height_scan_rotates_with_yaw(rough_env):
    env = rough_env
    # Known x-gradient independently checks yaw rotation and scan ordering.
    terrain = env.terrain
    terrain.height_field_raw[:] = np.arange(terrain.height_field_raw.shape[0])[:, None]
    env.height_points = np.array([[0.4, 0.0], [0.0, 0.3]])
    env.base_quat[:] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    measured = env._get_heights()
    base_x = env.base_pos[:, 0]
    expected = np.stack((base_x, base_x - 0.3), axis=1)
    expected = (
        (expected + env.cfg.terrain.border_size) / env.cfg.terrain.horizontal_scale * env.cfg.terrain.vertical_scale
    )
    np.testing.assert_allclose(measured, expected, atol=1e-12)


def test_scan_geometry_is_part_of_checkpoint_contract(rough_env):
    before = policy_contract(rough_env)
    rough_env.cfg.terrain.measured_points_x = list(reversed(rough_env.cfg.terrain.measured_points_x))
    assert policy_contract(rough_env) != before


def test_resized_collector_preserves_terrain_level_distribution():
    cfg = small_cfg()
    old = LeggedRobot(cfg)
    try:
        old.terrain_levels[:] = [0, 2, 1, 2]
        saved = old.training_state_dict()
    finally:
        old.close()
    cfg.env.num_envs = 6
    new = LeggedRobot(cfg)
    try:
        new.load_training_state_dict(saved)
        assert new.terrain_levels.tolist() == [0, 2, 1, 2, 0, 2]
        np.testing.assert_allclose(new.base_height, cfg.init_state.pos[2])
        assert new.get_observations()["policy"].shape[0] == 6
    finally:
        new.close()


def test_map_boundary_is_a_failure_not_a_timeout(rough_env):
    env = rough_env
    env.sim.qpos[0, 0] = -env.cfg.terrain.border_size - 0.5
    env.sim.forward([0])
    env._refresh_state()
    env.episode_length_buf[0] = env.max_episode_length
    env.check_termination()
    assert env.reset_buf[0] and not env.time_out_buf[0]


@pytest.mark.parametrize(
    "field,value", [("horizontal_scale", 0), ("terrain_proportions", [1, -1]), ("platform_size", 5)]
)
def test_invalid_terrain_settings_fail_early(field, value):
    cfg = small_cfg().terrain
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        Terrain(cfg, 4)

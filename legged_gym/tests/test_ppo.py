"""Real CPU/MPS/CUDA PPO updates and portable checkpoint integration tests."""

import json

import pytest
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.mujoco_helpers import class_to_dict
from legged_gym.utils.task_registry import LeggedOnPolicyRunner, runner_config


def require_device(device):
    kind = torch.device(device).type
    if kind == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple Metal is unavailable")
    if kind == "cuda" and not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA is unavailable")


@pytest.mark.parametrize("task", ["go2", "g1", "go2_rough", "g1_rough"])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("mps", marks=pytest.mark.mps), pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_ppo_update_checkpoint_and_curriculum(task, device, tmp_path):
    require_device(device)
    args = get_args(["--task", task, "--rl_device", device, "--num_envs", "4", "--num_threads", "2"])
    cfg, train_cfg = task_registry.get_cfgs(task)
    if "rough" in task:
        cfg.terrain.num_rows = 3
        cfg.terrain.num_cols = 4
        cfg.terrain.terrain_length = 4.0
        cfg.terrain.terrain_width = 4.0
        cfg.terrain.border_size = 1.0
    cfg.env.episode_length_s = 0.04  # Exercise timeouts and resets inside every rollout.
    train_cfg.policy.actor_hidden_dims = [32, 16]
    train_cfg.policy.critic_hidden_dims = [32, 16]
    train_cfg.runner.num_steps_per_env = 4
    train_cfg.algorithm.num_learning_epochs = 1
    train_cfg.algorithm.num_mini_batches = 2
    env, _ = task_registry.make_env(task, args, env_cfg=cfg)
    try:
        runner = LeggedOnPolicyRunner(env, runner_config(train_cfg), log_dir=None, device=device)
        assert env.get_observations()["policy"].device.type == "cpu"
        for model in (runner.alg.actor, runner.alg.critic):
            assert all(
                p.dtype == torch.float32 and p.device.type == torch.device(device).type for p in model.parameters()
            )
        original = [p.detach().clone() for p in runner.alg.actor.parameters()]
        original_critic = [p.detach().clone() for p in runner.alg.critic.parameters()]
        runner.learn(2)
        assert any(not torch.equal(p, q) for p, q in zip(original, runner.alg.actor.parameters()))
        assert any(not torch.equal(p, q) for p, q in zip(original_critic, runner.alg.critic.parameters()))
        assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
        assert all(torch.isfinite(p).all() for p in runner.alg.critic.parameters())
        env.command_ranges[:2] = [[-0.2, 0.2], [-0.2, 0.2]]
        if env.custom_origins:
            env.terrain_levels[:] = 1
            env._update_env_origins()
        checkpoint = tmp_path / "model.pt"
        runner.save(str(checkpoint))
        if env.custom_origins:
            (tmp_path / "run_config.json").write_text(
                json.dumps({"env": class_to_dict(env.cfg), "train": class_to_dict(train_cfg)})
            )
            wrong_args = get_args(
                ["--task", task.removesuffix("_rough"), "--rl_device", device, "--checkpoint_path", str(checkpoint)]
            )
            with pytest.raises(ValueError, match="different terrain task"):
                task_registry.get_cfgs(wrong_args.task, wrong_args)
        expected_state = env.training_state_dict()
        obs = env.get_observations().to(device)
        policy = runner.get_inference_policy(device=device)
        with torch.inference_mode():
            expected_actions = policy(obs).detach().cpu()
        replacement = LeggedOnPolicyRunner(env, runner_config(train_cfg), log_dir=None, device=device)
        env.command_ranges[:] = 0
        replacement.load(str(checkpoint))
        assert env.training_state_dict() == expected_state
        assert replacement.current_learning_iteration == 2
        with torch.inference_mode():
            actual_actions = replacement.get_inference_policy(device=device)(obs).cpu()
        torch.testing.assert_close(actual_actions, expected_actions)
        # A policy with identical dimensions still must not load with different PD semantics.
        env.p_gains *= 2
        with pytest.raises(ValueError, match="policy contract"):
            replacement.load(str(checkpoint))
    finally:
        env.close()


@pytest.mark.parametrize("task", ["go2", "g1_rough"])
@pytest.mark.parametrize(
    "source,target",
    [
        pytest.param("cpu", "mps", marks=pytest.mark.mps),
        pytest.param("mps", "cpu", marks=pytest.mark.mps),
        pytest.param("cpu", "cuda:0", marks=pytest.mark.cuda),
        pytest.param("cuda:0", "cpu", marks=pytest.mark.cuda),
    ],
)
def test_checkpoint_can_resume_on_another_learner_device(task, source, target, tmp_path):
    require_device(source)
    require_device(target)
    args = get_args(["--task", task, "--rl_device", source, "--num_envs", "4", "--num_threads", "2"])
    cfg, train = task_registry.get_cfgs(task)
    cfg.terrain.num_rows = 2
    cfg.terrain.num_cols = 2
    cfg.terrain.terrain_length = cfg.terrain.terrain_width = 4.0
    cfg.terrain.border_size = 1.0
    cfg.env.episode_length_s = 0.04
    train.policy.actor_hidden_dims = [32, 16]
    train.policy.critic_hidden_dims = [32, 16]
    train.runner.num_steps_per_env = 4
    train.algorithm.num_learning_epochs = 1
    train.algorithm.num_mini_batches = 2
    env, _ = task_registry.make_env(task, args, env_cfg=cfg)
    try:
        first = LeggedOnPolicyRunner(env, runner_config(train), log_dir=None, device=source)
        first.learn(1)
        checkpoint = tmp_path / "model.pt"
        first.save(str(checkpoint))
        state = env.training_state_dict()
        obs = env.get_observations().clone()
        with torch.inference_mode():
            expected = first.get_inference_policy(device=source)(obs.to(source)).cpu()
        second = LeggedOnPolicyRunner(env, runner_config(train), log_dir=None, device=target)
        second.load(str(checkpoint))
        assert env.training_state_dict() == state
        assert second.current_learning_iteration == 1
        for parameter, values in second.alg.optimizer.state.items():
            for name in ("exp_avg", "exp_avg_sq"):
                assert values[name].device == parameter.device
        with torch.inference_mode():
            actual = second.get_inference_policy(device=target)(obs.to(target)).cpu()
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        # Exercise the restored optimizer on the destination, not just inference.
        second.learn(1)
        assert all(
            p.device.type == torch.device(target).type and torch.isfinite(p).all()
            for p in second.alg.actor.parameters()
        )
    finally:
        env.close()


def test_config_snapshot_can_recreate_custom_training_run(tmp_path):
    args = get_args(["--task", "g1", "--rl_device", "cpu", "--num_envs", "2", "--num_threads", "1"])
    cfg, train_cfg = task_registry.get_cfgs("g1")
    cfg.control.action_scale = 0.15
    train_cfg.policy.actor_hidden_dims = [32]
    env, _ = task_registry.make_env("g1", args, env_cfg=cfg)
    try:
        runner, _ = task_registry.make_alg_runner(env, "g1", args, train_cfg, log_root=tmp_path)
        directory = runner.logger.log_dir
        checkpoint = str(tmp_path / directory / "model_0.pt")
        runner.save(checkpoint)
        snapshot = json.loads((tmp_path / directory / "run_config.json").read_text())
        assert snapshot["env"]["control"]["action_scale"] == 0.15
        load_args = get_args(["--task", "g1", "--rl_device", "cpu", "--checkpoint_path", checkpoint])
        restored_cfg, restored_train = task_registry.get_cfgs("g1", load_args)
        assert restored_cfg.control.action_scale == 0.15
        assert restored_train.policy.actor_hidden_dims == [32]
        assert restored_cfg.env.num_envs == 2
        restored_cfg.commands.ranges[0][0] = -99
        fresh, _ = task_registry.get_cfgs("g1")
        assert fresh.commands.ranges[0][0] == -0.1
        wrong_args = get_args(["--task", "go2", "--checkpoint_path", checkpoint, "--rl_device", "cpu"])
        with pytest.raises(ValueError, match="different robot"):
            task_registry.get_cfgs("go2", wrong_args)
    finally:
        env.close()

# Legged Gym: MuJoCo + rsl_rl

**English** | [한국어](README.ko.md)

Train **Unitree Go2** and **G1 (29 DoF)** to walk on flat ground. Physics runs on the CPU with **MuJoCo 3.17.0**; the learner device is selected automatically in this order: **CUDA → MPS → CPU**. PPO, SAC, and FlashSAC are supported.

Tasks are based on [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab), following the [legged_gym](https://github.com/leggedrobotics/legged_gym) project structure.

## Installation

Use an activated **Python 3.10–3.13** environment. For NVIDIA GPUs, install a CUDA-enabled [PyTorch](https://pytorch.org/get-started/locally/) build first.

The default PPO installation uses **rsl-rl 5.5.2**.

```bash
python -m pip install -e '.[ppo]'
```

Robot models and meshes are included in [resources/robots](resources/robots).

### Using SAC or FlashSAC

SAC uses the [rsl_rl_sac fork](https://github.com/leggedrobotics/rsl_rl_sac). Installing it replaces upstream rsl-rl in the current environment, so **use a separate virtual environment if you want to preserve your PPO setup**. The fork supports both PPO and SAC, but its PPO differs from upstream 5.5.2. Do not install `.[ppo,sac]` together.

```bash
python -m pip install -e '.[sac]'
```

For FlashSAC, install with `python -m pip install -e '.[flashsac]'`. It uses upstream rsl-rl 5.5.2; keep it in a separate environment from the SAC fork.

## Training

```bash
python -m legged_gym.scripts.train --task go2
python -m legged_gym.scripts.train --task g1

# Run in an environment with SAC installed
python -m legged_gym.scripts.train --task go2 --algorithm sac --num_envs 64 --headless

# Run in an environment with FlashSAC installed
python -m legged_gym.scripts.train --task g1 --algorithm flashsac --num_envs 64 --headless
```

Defaults are **go2, PPO, 4096 environments, 10 CPU workers, and 10001 iterations**. Adjust them with `--num_envs`, `--num_threads`, and `--max_iterations`. Select a learner explicitly with `--rl_device mps`, `cuda`, `cuda:N`, or `cpu`; physics always runs on the CPU.

The Viser monitor shows the robot and tracking velocities during training. Its default address is **http://127.0.0.1:8080**; the actual URL is printed at startup. Use `--headless` to disable the monitor and **Ctrl-C** to stop training.

New SAC/FlashSAC runs on MPS/CUDA automatically overlap CPU collection with learning and reuse replay sample buffers where supported; CPU runs default to serial execution. No extra flags are needed. Use `--collector_mode serial` for serial collection.

For SAC/FlashSAC, adjust replay capacity, batch size, and initial collection iterations with `--replay_buffer_size`, `--batch_size`, and `--start_training`. PPO and off-policy algorithms collect different amounts of data per iteration, so iteration counts alone are not comparable. Run `python -m legged_gym.scripts.train --help` for all options.

## Checkpoints and resuming

Checkpoints, settings, and TensorBoard logs are saved under `logs/go2/` or `logs/g1/` by default. Separate experiments with names such as `--experiment_name go2_sac`.

```bash
python -m legged_gym.scripts.train --task g1 --resume --max_iterations 1000
tensorboard --logdir logs
```

`--resume` loads the latest checkpoint in the selected experiment; `--max_iterations` specifies **additional training iterations**. Use `--checkpoint_path PATH` to select a file. Keep `run_config.json` with the checkpoint and use the same task and backend as during training. The algorithm is restored from the saved settings.

SAC/FlashSAC refill their replay buffers when resuming, so this is not an exact continuation of the interrupted training state. Older FlashSAC checkpoints require the original `run_config.json` to restore the learning-rate schedule.

Saved execution settings take precedence over new defaults. To reselect collection and replay sampling automatically when resuming SAC/FlashSAC, add `--collector_mode auto --replay_sampling auto`.

## Playback and evaluation

On macOS, launch the interactive viewer with `mjpython`. On Linux, use `python`.

```bash
mjpython -m legged_gym.scripts.play --task g1 --velocity 0.5 0 0
```

Playback loads the latest checkpoint by default. Select a file with `--checkpoint_path PATH`; `--velocity VX VY WZ` sets forward/lateral velocity (m/s) and yaw rate (rad/s). To run without a viewer, use `python` with `--headless --steps 500`.

Evaluate fixed commands across multiple seeds:

```bash
python -m legged_gym.scripts.play --task go2 --evaluate --checkpoint_path PATH
```

Tracking errors, failure rates, and settling times are saved as JSON/CSV beside the checkpoint under `evaluations/`. Defaults are 16 environments and seeds 0/1/2. Adjust with `--eval_seeds`, `--num_envs`, `--steps`, and `--eval_output`; use identical evaluation settings when comparing checkpoints.

## Configuration and licenses

Shared settings: [legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py). Robot settings: [go2_config.py](legged_gym/envs/go2/go2_config.py), [g1_config.py](legged_gym/envs/g1/g1_config.py).

Licenses: [project](LICENSE) · [third-party](licenses).

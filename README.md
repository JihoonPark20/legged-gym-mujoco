# Legged Gym: MuJoCo + PPO

**English** | [한국어](README.ko.md)

Train **Unitree Go2** and **G1 (29 DoF)** to walk on flat ground with PPO. The tasks are based on [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab), following the [legged_gym](https://github.com/leggedrobotics/legged_gym) project structure.

Physics runs on the CPU using MuJoCo **3.17.0**'s official `mujoco.batch.Batch` API. No separate `mjbatch` package or MJX/JAX/Warp is required. PPO uses **rsl-rl-lib 5.0.1**, with automatic device selection: **CUDA → MPS → CPU**.

## Installation

Use **Python 3.10–3.13** in an activated environment. For NVIDIA GPUs on Linux, install a CUDA-enabled [PyTorch](https://pytorch.org/get-started/locally/) build first.

```bash
python -m pip install -e .
```

Robot models and meshes are included in [`resources/robots`](resources/robots).

## Training

```bash
python -m legged_gym.scripts.train --task go2
python -m legged_gym.scripts.train --task g1
```

The default task is `go2`. Both tasks use **4096 environments, 10 CPU workers and 10001 iterations**. Adjust these with `--num_envs`, `--num_threads` and `--max_iterations`. Use `--rl_device mps`, `cuda`, `cuda:N` or `cpu` to select a learner explicitly; physics always runs on the CPU.

Open **http://127.0.0.1:8080** for the Viser training monitor. It shows the robot, commanded/actual forward and yaw velocities, and training progress. The actual URL is printed at startup. Use `--headless` to disable the monitor and **Ctrl-C** to stop training.

## Checkpoints and resuming

Runs are saved under `logs/go2/` or `logs/g1/`, including checkpoints, configuration and TensorBoard logs.

```bash
python -m legged_gym.scripts.train --task g1 --resume --max_iterations 1000
tensorboard --logdir logs
```

`--resume` selects the latest run; `--max_iterations` is the number of **additional** iterations. Use `--checkpoint_path PATH` to select a specific checkpoint. Keep the same task as the checkpoint; checkpoints from different tasks are not interchangeable.

## Playback

On macOS, use `mjpython` for the interactive viewer:

```bash
mjpython -m legged_gym.scripts.play --task g1 --velocity 0.5 0 0
```

On Linux, replace `mjpython` with `python`. Use `--task go2` for Go2. Playback loads the latest checkpoint by default; `--checkpoint_path PATH` selects another. `--velocity VX VY WZ` sets forward/lateral velocity (m/s) and yaw rate (rad/s). For playback without a viewer, use `python` with `--headless --steps 500`.

## Configuration and references

Shared settings: [`legged_robot_config.py`](legged_gym/envs/base/legged_robot_config.py). Robot settings: [`go2_config.py`](legged_gym/envs/go2/go2_config.py), [`g1_config.py`](legged_gym/envs/g1/g1_config.py).

Licenses: [project](LICENSE) · [third-party](licenses).

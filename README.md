# Legged Gym: MuJoCo + PPO

**English** | [한국어](README.ko.md)

Train velocity-tracking locomotion for **Unitree Go2** and **G1 (29 DoF)** on flat or procedurally generated terrain. The project follows the [legged_gym](https://github.com/leggedrobotics/legged_gym) structure, with environment and PPO settings based on [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab).

Physics runs on the CPU using MuJoCo **3.17.0**'s official `mujoco.batch.Batch` API. No separate `mjbatch` package or MJX/JAX/Warp is required. PPO uses **rsl-rl-lib 5.5.1** with automatic device selection: **CUDA → MPS → CPU**.

## Installation

Use an activated **Python 3.10+** environment. For Ubuntu/Linux with NVIDIA GPUs, first install a CUDA-enabled PyTorch build using the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

```bash
python -m pip install -e .
```

Robot models and meshes are included in [`resources/robots`](resources/robots); no separate download is needed.

## Training

```bash
python -m legged_gym.scripts.train --task go2 --num_envs 4096 --num_threads 10
python -m legged_gym.scripts.train --task g1 --num_envs 4096 --num_threads 10
```

| Task | Robot | Terrain |
| --- | --- | --- |
| `go2` (alias: `go2_flat`) | Go2, 12 DoF | Flat |
| `g1` (alias: `g1_29dof`) | G1, 29 DoF | Flat |
| `go2_rough` | Go2, 12 DoF | Generated terrain |
| `g1_rough` | G1, 29 DoF | Generated terrain |

Training runs without a viewer. Use `play` to view a trained policy.

Useful options:

| Option | Purpose |
| --- | --- |
| `--rl_device auto` | Default: select CUDA, then MPS, then CPU |
| `--rl_device mps / cuda / cuda:N / cpu` | Explicitly select one learning device |
| `--num_envs 4096` | Number of parallel environments (default: 4096) |
| `--num_threads 10` | MuJoCo worker threads (default: 10); `0` selects the CPU count |
| `--max_iterations 1000` | PPO iterations to run in this invocation |
| `--seed 1` | Random seed |

Adjust environment and thread counts to fit your CPU and memory. Physics always runs on the CPU, regardless of the learning device. An explicitly selected unavailable device raises an error. The selected device is printed at startup.

For a short trial run:

```bash
python -m legged_gym.scripts.train --task go2 --num_envs 8 --num_threads 4 --max_iterations 2
```

### Logs and resuming

Runs are saved under `logs/<experiment>/<run>/`, including TensorBoard logs, `run_config.json`, and `model_*.pt` checkpoints.

```bash
tensorboard --logdir logs

# Resume the latest run for this task
python -m legged_gym.scripts.train --task g1 --resume --max_iterations 1000

# Resume a specific checkpoint (replace RUN with the run directory)
python -m legged_gym.scripts.train --task g1 --checkpoint_path logs/g1_29dof_flat/RUN/model_100.pt --max_iterations 1000
```

`--max_iterations` specifies **additional** iterations when resuming. Saved training settings, model, optimizer, and curriculum are restored; simulation starts with new episodes. You can resume on another learning device with `--rl_device`.

## Playback

On macOS, use `mjpython` for the interactive viewer:

```bash
mjpython -m legged_gym.scripts.play --task g1 --velocity 0.5 0 0
```

On Linux with a graphical display, use `python`:

```bash
python -m legged_gym.scripts.play --task go2 --velocity 0.5 0 0
```

For playback without a viewer:

```bash
python -m legged_gym.scripts.play --task g1 --headless --steps 500
```

Playback loads the task's latest checkpoint by default. Use `--checkpoint_path PATH` or `--load_run RUN --checkpoint 100` to select one. `--velocity VX VY WZ` sets forward/lateral velocity (m/s) and yaw rate (rad/s). Use the same task as the checkpoint; flat and rough policies are not interchangeable.

## Generated terrain

Rough tasks include slopes, uneven ground, stairs, obstacles, and stepping stones, with a terrain difficulty curriculum.

```bash
python -m legged_gym.scripts.train --task go2_rough --num_envs 4096 --num_threads 10
python -m legged_gym.scripts.train --task g1_rough --num_envs 4096 --num_threads 10
mjpython -m legged_gym.scripts.play --task go2_rough --terrain_level 2 --terrain_column 3 --velocity 0.5 0 0
```

For the viewer command above, use `python` on Linux. `--terrain_level` selects difficulty (`0` is easiest); `--terrain_column` selects terrain type. With the default configuration, columns are: `0` slope, `1–2` uneven ground, `3–4` ascending stairs, `5–7` descending stairs, `8` obstacles, `9` stepping stones.

Terrain is generated as a MuJoCo heightfield. Adjust its proportions, dimensions, resolution, and difficulty in the terrain configuration.

## Configuration

| File | What to customize |
| --- | --- |
| [`mujoco_config.py`](legged_gym/envs/base/mujoco_config.py) | Shared environment, terrain, and PPO settings |
| [`go2_flat_config.py`](legged_gym/envs/go2/go2_flat_config.py) | Go2 control, observations, and rewards |
| [`g1_config.py`](legged_gym/envs/g1/g1_config.py) | G1 control, observations, and rewards |
| [`go2_rough_config.py`](legged_gym/envs/go2/go2_rough_config.py), [`g1_rough_config.py`](legged_gym/envs/g1/g1_rough_config.py) | Robot-specific rough terrain settings |
| [`legged_robot.py`](legged_gym/envs/base/legged_robot.py) | Environment steps, resets, and reward functions |

## Tests and benchmark

```bash
python -m pip install -e '.[test]'
python -B -m pytest -q -p no:cacheprovider
python -B -m legged_gym.scripts.benchmark --task go2 --num_envs 4096 --num_threads 10 --steps 500
```

The benchmark measures CPU collection throughput with zero actions, excluding policy inference and PPO updates. Tests skip unavailable accelerators. Hardware validation has been performed on macOS MPS/CPU; CUDA execution and long-run locomotion convergence remain to be validated.

## Credits and licenses

- Project structure: [legged_gym](https://github.com/leggedrobotics/legged_gym), [BSD-3-Clause](LICENSE).
- Environment and PPO settings: [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab), [Apache-2.0](licenses/unitree_rl_lab_LICENSE).
- Robot assets: [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), [BSD-3-Clause](resources/robots/UNITREE_LICENSE). Source revisions and checksums: [`provenance.json`](resources/robots/provenance.json).
- Simulation: [MuJoCo batch API](https://mujoco.readthedocs.io/en/stable/python.html#batch) and [changelog](https://mujoco.readthedocs.io/en/stable/changelog.html).

# Legged Gym: MuJoCo + rsl_rl

[English](README.md) | **한국어**

**Unitree Go2**와 **G1 (29 DoF)**의 평지 보행을 학습합니다. 물리는 **MuJoCo 3.17.0**으로 CPU에서 실행하며, 학습 장치는 **CUDA → MPS → CPU** 순으로 자동 선택합니다. 현재 PPO와 SAC를 지원합니다.

태스크는 [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)을 기반으로 구성했으며, [legged_gym](https://github.com/leggedrobotics/legged_gym)의 프로젝트 구조를 따릅니다.

## 설치

활성화한 **Python 3.10–3.13** 환경에서 설치합니다. NVIDIA GPU를 사용한다면 CUDA 지원 [PyTorch](https://pytorch.org/get-started/locally/)를 먼저 설치하세요.

기본 PPO 설치는 **rsl-rl 5.5.2**를 사용합니다.

```bash
python -m pip install -e '.[ppo]'
```

로봇 모델과 메시는 [resources/robots](resources/robots)에 포함되어 있습니다.

### SAC 사용 시

SAC는 [rsl_rl_sac 포크](https://github.com/leggedrobotics/rsl_rl_sac)를 설치합니다. 기존 환경에서 실행하면 upstream rsl-rl을 교체하므로, **기존 PPO 환경을 유지하려면 별도 가상환경을 권장합니다.** 이 포크에서도 PPO와 SAC를 모두 사용할 수 있지만, PPO는 upstream 5.5.2 구현과 다릅니다. `.[ppo,sac]`를 함께 설치하지 마세요.

```bash
python -m pip install -e '.[sac]'
```

## 학습

```bash
python -m legged_gym.scripts.train --task go2
python -m legged_gym.scripts.train --task g1

# SAC 설치 환경에서 실행
python -m legged_gym.scripts.train --task go2 --algorithm sac --num_envs 64 --headless
```

기본값은 **go2, PPO, 환경 4096개, CPU 워커 10개, 10001 iteration**입니다. `--num_envs`, `--num_threads`, `--max_iterations`로 조절하세요. 학습 장치는 `--rl_device mps`, `cuda`, `cuda:N` 또는 `cpu`로 지정할 수 있습니다. 물리는 항상 CPU에서 실행합니다.

학습 중 Viser 모니터에서 로봇과 추종 속도를 확인할 수 있습니다. 기본 주소는 **http://127.0.0.1:8080**이며, 실제 주소는 시작 시 출력됩니다. `--headless`로 모니터를 끄고, **Ctrl-C**로 학습을 중단합니다.

SAC의 replay 용량, batch 크기, 초기 수집 횟수는 `--replay_buffer_size`, `--batch_size`, `--start_training`으로 조절합니다. PPO와 SAC는 iteration당 수집량이 다르므로 iteration 수만으로 학습량을 비교하지 마세요. 전체 옵션은 `python -m legged_gym.scripts.train --help`로 확인할 수 있습니다.

## 체크포인트와 재개

체크포인트, 설정, TensorBoard 로그는 기본적으로 `logs/go2/` 또는 `logs/g1/`에 저장됩니다. 실험을 구분하려면 `--experiment_name go2_sac`처럼 지정하세요.

```bash
python -m legged_gym.scripts.train --task g1 --resume --max_iterations 1000
tensorboard --logdir logs
```

`--resume`은 선택한 실험의 최신 체크포인트를 불러오며, `--max_iterations`는 **추가 학습 횟수**입니다. 특정 파일은 `--checkpoint_path PATH`로 선택합니다. 체크포인트와 함께 `run_config.json`도 보관하고, 학습할 때와 같은 태스크·백엔드를 사용하세요. 알고리즘은 저장된 설정에서 자동 복구합니다.

SAC는 재개 시 replay buffer를 다시 채운 뒤 학습하므로 중단 직후의 학습 상태를 완전히 재현하지는 않습니다.

## 재생과 평가

macOS의 대화형 뷰어는 `mjpython`으로 실행합니다. Linux에서는 `python`을 사용하세요.

```bash
mjpython -m legged_gym.scripts.play --task g1 --velocity 0.5 0 0
```

기본적으로 최신 체크포인트를 불러옵니다. `--checkpoint_path PATH`로 파일을 지정하고, `--velocity VX VY WZ`로 전후·좌우 속도(m/s)와 yaw 각속도(rad/s)를 설정합니다. 뷰어 없이 실행하려면 `python`과 `--headless --steps 500`을 사용하세요.

고정 명령으로 여러 시드에서 평가하려면:

```bash
python -m legged_gym.scripts.play --task go2 --evaluate --checkpoint_path PATH
```

추종 오차, 실패율, 응답 안정화 시간을 체크포인트 옆 `evaluations/`에 JSON/CSV로 저장합니다. 기본값은 환경 16개와 시드 0/1/2입니다. `--eval_seeds`, `--num_envs`, `--steps`, `--eval_output`으로 조절하며, 체크포인트 비교 시 같은 평가 조건을 사용하세요.

## 설정과 라이선스

공통 설정: [legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py). 로봇별 설정: [go2_config.py](legged_gym/envs/go2/go2_config.py), [g1_config.py](legged_gym/envs/g1/g1_config.py).

라이선스: [프로젝트](LICENSE) · [외부 구성요소](licenses).

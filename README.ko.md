# Legged Gym: MuJoCo + PPO

[English](README.md) | **한국어**

**Unitree Go2**와 **G1 (29 DoF)**의 평지 보행을 PPO로 학습합니다. [legged_gym](https://github.com/leggedrobotics/legged_gym)의 프로젝트 구조를 따르며, 태스크는 [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)을 기반으로 구성했습니다.

물리는 MuJoCo **3.17.0**의 공식 `mujoco.batch.Batch` API로 CPU에서 실행합니다. 별도 `mjbatch` 패키지나 MJX/JAX/Warp가 필요하지 않습니다. PPO는 **rsl-rl-lib 5.0.1**을 사용하며, 학습 장치는 **CUDA → MPS → CPU** 순으로 자동 선택합니다.

## 설치

활성화한 **Python 3.10–3.13** 환경을 사용하세요. Linux의 NVIDIA GPU를 사용한다면 CUDA 지원 [PyTorch](https://pytorch.org/get-started/locally/)를 먼저 설치하세요.

```bash
python -m pip install -e .
```

로봇 모델과 메시는 [`resources/robots`](resources/robots)에 포함되어 있습니다.

## 학습

```bash
python -m legged_gym.scripts.train --task go2
python -m legged_gym.scripts.train --task g1
```

기본 태스크는 `go2`입니다. 두 태스크 모두 **환경 4096개, CPU 워커 10개, 10001 iteration**을 기본으로 사용합니다. `--num_envs`, `--num_threads`, `--max_iterations`로 조절하세요. 학습 장치를 직접 선택하려면 `--rl_device mps`, `cuda`, `cuda:N` 또는 `cpu`를 지정합니다. 물리는 항상 CPU에서 실행합니다.

**http://127.0.0.1:8080**에서 Viser 학습 모니터를 엽니다. 로봇, 명령·실제 X/Y 속도와 yaw 각속도, 학습 진행 상황을 확인할 수 있습니다. 실제 접속 주소는 시작 시 출력됩니다. `--headless`로 모니터를 끄고, **Ctrl-C**로 학습을 중단할 수 있습니다.

## 체크포인트와 재개

체크포인트, 설정, TensorBoard 로그는 `logs/go2/` 또는 `logs/g1/`에 저장됩니다.

```bash
python -m legged_gym.scripts.train --task g1 --resume --max_iterations 1000
tensorboard --logdir logs
```

`--resume`은 최신 실행을 선택하며, `--max_iterations`는 **추가로** 학습할 반복 횟수입니다. 특정 체크포인트를 선택하려면 `--checkpoint_path PATH`를 사용하세요. 체크포인트와 같은 태스크를 지정하세요. 서로 다른 태스크의 체크포인트는 호환되지 않습니다.

## 재생

macOS에서는 대화형 뷰어를 `mjpython`으로 실행합니다.

```bash
mjpython -m legged_gym.scripts.play --task g1 --velocity 0.5 0 0
```

Linux에서는 `mjpython` 대신 `python`을 사용합니다. Go2는 `--task go2`로 선택하세요. 기본적으로 최신 체크포인트를 불러오며, `--checkpoint_path PATH`로 직접 지정할 수도 있습니다. `--velocity VX VY WZ`는 전후·좌우 속도(m/s)와 yaw 각속도(rad/s)입니다. 뷰어 없이 재생하려면 `python`과 `--headless --steps 500`을 사용하세요.

고정된 명령을 여러 시드에서 뷰어 없이 평가합니다.

```bash
python -m legged_gym.scripts.play --task go2 --evaluate --checkpoint_path PATH
```

체크포인트 옆 `evaluations/`에 X/Y/yaw MAE·RMSE, 실패율, 응답 안정화 시간을 JSON/CSV로 저장합니다. 기본값은 환경 16개, 시드 0/1/2, 명령별 준비 2초·측정 5초입니다. 체크포인트를 비교할 때는 같은 설정을 사용하세요. `--eval_seeds 0 1 2`, `--num_envs`, `--steps`(명령별 측정 스텝), `--eval_output DIR`로 조절하며, 평가 조건은 `LeggedRobotCfgPPO.evaluation`에 있습니다. 학습 로그에도 축별 추종 오차와 종료 원인을 기록합니다.

## 설정과 참고 자료

공통 설정: [`legged_robot_config.py`](legged_gym/envs/base/legged_robot_config.py). 로봇별 설정: [`go2_config.py`](legged_gym/envs/go2/go2_config.py), [`g1_config.py`](legged_gym/envs/g1/g1_config.py).

라이선스: [프로젝트](LICENSE) · [외부 구성요소](licenses).

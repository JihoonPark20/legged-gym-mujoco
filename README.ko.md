# Legged Gym: MuJoCo + PPO

[English](README.md) | **한국어**

**Unitree Go2**와 **G1 (29 DoF)**의 속도 추종 보행을 평지와 생성 지형에서 학습합니다. [legged_gym](https://github.com/leggedrobotics/legged_gym)의 구조를 따르며, 환경과 PPO 설정은 [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)을 참고했습니다.

물리는 MuJoCo **3.17.0**의 공식 `mujoco.batch.Batch` API로 CPU에서 실행합니다. 별도 `mjbatch` 패키지나 MJX/JAX/Warp가 필요하지 않습니다. PPO는 **rsl-rl-lib 5.5.1**을 사용하며, 학습 장치는 **CUDA → MPS → CPU** 순으로 자동 선택합니다.

## 설치

활성화한 **Python 3.10 이상** 환경에서 실행하세요. Ubuntu/Linux의 NVIDIA GPU를 사용한다면 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에 따라 CUDA 지원 PyTorch를 먼저 설치하세요.

```bash
python -m pip install -e .
```

로봇 모델과 메시는 [`resources/robots`](resources/robots)에 포함되어 있어 별도 다운로드가 필요하지 않습니다.

## 학습

```bash
python -m legged_gym.scripts.train --task go2 --num_envs 4096 --num_threads 10
python -m legged_gym.scripts.train --task g1 --num_envs 4096 --num_threads 10
```

| 태스크 | 로봇 | 지형 |
| --- | --- | --- |
| `go2` (별칭: `go2_flat`) | Go2, 12 DoF | 평지 |
| `g1` (별칭: `g1_29dof`) | G1, 29 DoF | 평지 |
| `go2_rough` | Go2, 12 DoF | 생성 지형 |
| `g1_rough` | G1, 29 DoF | 생성 지형 |

학습은 뷰어 없이 실행합니다. 학습한 정책은 `play`로 확인하세요.

주요 옵션:

| 옵션 | 용도 |
| --- | --- |
| `--rl_device auto` | 기본값. CUDA → MPS → CPU 순으로 선택 |
| `--rl_device mps / cuda / cuda:N / cpu` | 학습 장치 중 하나를 직접 지정 |
| `--num_envs 4096` | 병렬 환경 수 (기본값: 4096) |
| `--num_threads 10` | MuJoCo 워커 스레드 수 (기본값: 10). `0`이면 CPU 개수 사용 |
| `--max_iterations 1000` | 이번 실행에서 학습할 PPO 반복 횟수 |
| `--seed 1` | 난수 시드 |

환경 수와 스레드 수는 CPU와 메모리에 맞춰 조절하세요. 물리는 학습 장치와 관계없이 항상 CPU에서 실행합니다. 직접 지정한 장치를 사용할 수 없으면 오류가 발생합니다. 선택된 장치는 시작 시 출력됩니다.

짧은 시험 실행:

```bash
python -m legged_gym.scripts.train --task go2 --num_envs 8 --num_threads 4 --max_iterations 2
```

### 로그와 학습 재개

실행 결과는 `logs/<experiment>/<run>/`에 저장됩니다. TensorBoard 로그, `run_config.json`, `model_*.pt` 체크포인트가 포함됩니다.

```bash
tensorboard --logdir logs

# 해당 태스크의 최신 실행에서 재개
python -m legged_gym.scripts.train --task g1 --resume --max_iterations 1000

# 특정 체크포인트에서 재개 (RUN을 실제 실행 디렉토리로 변경)
python -m legged_gym.scripts.train --task g1 --checkpoint_path logs/g1_29dof_flat/RUN/model_100.pt --max_iterations 1000
```

재개할 때 `--max_iterations`는 **추가로** 학습할 반복 횟수입니다. 저장된 학습 설정, 모델, 옵티마이저와 커리큘럼을 복원하고, 시뮬레이션은 새 에피소드에서 시작합니다. `--rl_device`로 다른 학습 장치를 선택해 재개할 수 있습니다.

## 재생

macOS에서는 대화형 뷰어를 `mjpython`으로 실행합니다.

```bash
mjpython -m legged_gym.scripts.play --task g1 --velocity 0.5 0 0
```

그래픽 디스플레이가 있는 Linux에서는 `python`을 사용합니다.

```bash
python -m legged_gym.scripts.play --task go2 --velocity 0.5 0 0
```

뷰어 없이 실행하려면:

```bash
python -m legged_gym.scripts.play --task g1 --headless --steps 500
```

기본적으로 해당 태스크의 최신 체크포인트를 불러옵니다. 직접 선택하려면 `--checkpoint_path PATH` 또는 `--load_run RUN --checkpoint 100`을 사용하세요. `--velocity VX VY WZ`는 전후·좌우 속도(m/s)와 yaw 각속도(rad/s)를 지정합니다. 체크포인트와 같은 태스크를 사용하세요. 평지와 rough 정책은 서로 호환되지 않습니다.

## 지형 생성

Rough 태스크는 경사, 거친 지면, 계단, 장애물, 디딤돌을 포함하며 지형 난이도 커리큘럼을 사용합니다.

```bash
python -m legged_gym.scripts.train --task go2_rough --num_envs 4096 --num_threads 10
python -m legged_gym.scripts.train --task g1_rough --num_envs 4096 --num_threads 10
mjpython -m legged_gym.scripts.play --task go2_rough --terrain_level 2 --terrain_column 3 --velocity 0.5 0 0
```

위 뷰어 명령은 Linux에서 `python`으로 실행하세요. `--terrain_level`은 난이도(`0`이 가장 쉬움), `--terrain_column`은 지형 종류를 선택합니다. 기본 설정에서 열은 `0` 경사, `1–2` 거친 지면, `3–4` 올라가는 계단, `5–7` 내려가는 계단, `8` 장애물, `9` 디딤돌입니다.

지형은 MuJoCo heightfield로 생성합니다. 지형 설정에서 종류별 비율, 크기, 해상도와 난이도를 조절할 수 있습니다.

## 설정 변경

| 파일 | 수정할 내용 |
| --- | --- |
| [`mujoco_config.py`](legged_gym/envs/base/mujoco_config.py) | 공통 환경·지형·PPO 설정 |
| [`go2_flat_config.py`](legged_gym/envs/go2/go2_flat_config.py) | Go2 제어·관측·보상 |
| [`g1_config.py`](legged_gym/envs/g1/g1_config.py) | G1 제어·관측·보상 |
| [`go2_rough_config.py`](legged_gym/envs/go2/go2_rough_config.py), [`g1_rough_config.py`](legged_gym/envs/g1/g1_rough_config.py) | 로봇별 rough 지형 설정 |
| [`legged_robot.py`](legged_gym/envs/base/legged_robot.py) | 환경 스텝·리셋·보상 함수 |

## 테스트와 벤치마크

```bash
python -m pip install -e '.[test]'
python -B -m pytest -q -p no:cacheprovider
python -B -m legged_gym.scripts.benchmark --task go2 --num_envs 4096 --num_threads 10 --steps 500
```

벤치마크는 action을 0으로 설정한 CPU 데이터 수집 처리량이며, 정책 추론과 PPO 업데이트는 포함하지 않습니다. 테스트는 사용할 수 없는 가속기를 건너뜁니다. 실제 하드웨어 검증은 macOS MPS/CPU에서 수행했으며, CUDA 실행과 장기 보행 학습 수렴은 추가 검증이 필요합니다.

## 출처와 라이선스

- 프로젝트 구조: [legged_gym](https://github.com/leggedrobotics/legged_gym), [BSD-3-Clause](LICENSE).
- 환경·PPO 설정: [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab), [Apache-2.0](licenses/unitree_rl_lab_LICENSE).
- 로봇 자산: [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), [BSD-3-Clause](resources/robots/UNITREE_LICENSE). 원본 버전과 체크섬: [`provenance.json`](resources/robots/provenance.json).
- 시뮬레이션: [MuJoCo batch API](https://mujoco.readthedocs.io/en/stable/python.html#batch), [변경 내역](https://mujoco.readthedocs.io/en/stable/changelog.html).

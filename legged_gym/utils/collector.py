"""Spawned CPU environment with one bounded in-flight step.

Only CPU snapshots cross the pipe. Models, optimizers and replay stay in the
learner process; receiving a snapshot never aliases the next collector step.
"""

import multiprocessing as mp
import random
import signal
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict


def _encode(value):
    if isinstance(value, TensorDict):
        return ("tensordict", list(value.batch_size), {k: _encode(v) for k, v in value.items()})
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Collector transport accepts CPU tensors only")
        return ("tensor", value.detach().numpy())
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    return value


def _decode(value):
    if isinstance(value, tuple) and value[0] == "tensor":
        return torch.from_numpy(value[1])
    if isinstance(value, tuple) and value[0] == "tensordict":
        return TensorDict({k: _decode(v) for k, v in value[2].items()}, batch_size=value[1])
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


@dataclass
class EnvironmentFactory:
    task_class: type
    cfg: object

    def __call__(self):
        return self.task_class(cfg=self.cfg, physics_engine="mujoco", sim_device="cpu", headless=True)


def _worker(connection, factory):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    random.seed(factory.cfg.seed)
    np.random.seed(factory.cfg.seed)
    # Do not initialize an accelerator in the collector.
    torch.random.default_generator.manual_seed(factory.cfg.seed)
    env = None
    try:
        env = factory()
        env.capture_terminal_observations = True
        names = (
            "num_envs",
            "num_actions",
            "num_obs",
            "num_privileged_obs",
            "dt",
            "max_episode_length",
            "dof_names",
            "dof_pos_limits",
            "default_dof_pos",
            "_action_scale",
            "p_gains",
            "d_gains",
            "torque_limits",
        )
        metadata = {name: getattr(env, name) for name in names}
        metadata["sim"] = {"asset_digest": env.sim.asset_digest, "nthread": env.sim.batch.nthread}

        def snapshot():
            return {
                "obs": env.get_observations(),
                "lengths": env.episode_length_buf,
                "state": env.training_state_dict(),
            }

        connection.send(("ok", _encode((metadata, snapshot()))))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "step":
                result = env.step(_decode(payload))
                response = (result, {"lengths": env.episode_length_buf, "state": env.training_state_dict()})
            elif command == "lengths":
                env.episode_length_buf = _decode(payload)
                response = snapshot()
            elif command == "load_state":
                env.load_training_state_dict(payload)
                response = snapshot()
            elif command == "frame":
                response = env.get_monitor_frame(payload)
            else:
                raise ValueError(f"Unknown collector command: {command}")
            connection.send(("ok", _encode(response)))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class ProcessEnvironment:
    """CPU VecEnv proxy. Call close() in the caller's finally block."""

    def __init__(self, factory, timeout=120.0):
        self.cfg = factory.cfg
        self.device = torch.device("cpu")
        self.headless = True
        self.timeout = timeout
        self._closed = False
        self._pending = False
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, factory), name="legged-gym-collector", daemon=True)
        try:
            self.process.start()
            child.close()
            metadata, snapshot = self._receive()
            sim = metadata.pop("sim")
            self.__dict__.update(metadata)
            self.sim = SimpleNamespace(asset_digest=sim["asset_digest"], batch=SimpleNamespace(nthread=sim["nthread"]))
            self._accept(snapshot)
        except BaseException:
            child.close()
            self.close()
            raise

    def _accept(self, snapshot):
        self.obs_buf = snapshot["obs"]
        self._episode_length_buf = snapshot["lengths"]
        self._training_state = snapshot["state"]
        self.common_step_counter = self._training_state["common_step_counter"]

    def _receive(self):
        deadline = time.monotonic() + self.timeout
        while not self._connection.poll(0.1):
            if not self.process.is_alive():
                raise RuntimeError(f"Collector exited with code {self.process.exitcode}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Collector did not respond within {self.timeout}s")
        try:
            status, payload = self._connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError("Collector connection closed unexpectedly") from error
        if status == "error":
            raise RuntimeError(f"Collector failed:\n{payload}")
        return _decode(payload)

    def _send(self, command, payload=None):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._pending:
            raise RuntimeError("Receive the pending collector step before issuing another command")
        try:
            self._connection.send((command, _encode(payload)))
        except (BrokenPipeError, EOFError, OSError) as error:
            raise RuntimeError("Collector connection closed unexpectedly") from error

    def get_observations(self):
        return self.obs_buf

    @property
    def episode_length_buf(self):
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self._send("lengths", value.detach().to("cpu").clone())
        self._accept(self._receive())

    def step_async(self, actions):
        self._send("step", actions.detach().to(device="cpu", dtype=torch.float32))
        self._pending = True

    def step_wait(self):
        if not self._pending:
            raise RuntimeError("No collector step is pending")
        try:
            result, snapshot = self._receive()
            snapshot["obs"] = result[0]
            self._accept(snapshot)
            return tuple(result)
        finally:
            self._pending = False

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def training_state_dict(self):
        if self._pending:
            raise RuntimeError("Checkpoint requires a completed collector step")
        return dict(self._training_state)

    def load_training_state_dict(self, state):
        self._send("load_state", state)
        self._accept(self._receive())

    def get_monitor_frame(self, index):
        self._send("frame", index)
        return self._receive()

    def get_monitor_model(self):
        from legged_gym.sim.mujoco_batch import load_model

        return load_model(self.cfg)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.is_alive() and not self._pending:
                self._connection.send(("close", None))
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._connection.close()
        if self.process.pid is not None:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)

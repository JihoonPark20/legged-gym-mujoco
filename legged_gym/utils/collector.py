"""Spawned CPU environment with one bounded in-flight step.

CPU arrays use reusable shared memory; the socket carries control messages. Models, optimizers and replay stay in the
learner process; receiving a snapshot never aliases the next collector step.
"""

import multiprocessing as mp
import pickle
import random
import select
import signal
import socket
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
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


class _SharedArrays:
    """One bounded slot, reused only after the peer has copied the prior payload."""

    def __init__(self, *, size=0, name=None):
        self.owner = name is None
        options = {"track": False} if name is not None and sys.version_info >= (3, 13) else {}
        self.memory = SharedMemory(create=self.owner, size=size, name=name, **options)
        self.bytes_used = 0

    def pack(self, value):
        self.bytes_used = 0

        def visit(item):
            if isinstance(item, np.ndarray) and not item.dtype.hasobject and item.dtype.fields is None:
                offset = (self.bytes_used + 63) // 64 * 64
                if offset + item.nbytes <= self.memory.size:
                    view = np.ndarray(item.shape, dtype=item.dtype, buffer=self.memory.buf, offset=offset)
                    np.copyto(view, item)
                    self.bytes_used = offset + item.nbytes
                    return ("shared_array", offset, item.shape, item.dtype.str)
                # Unusually large optional extras may use the socket fallback.
                return item
            if isinstance(item, dict):
                return {key: visit(entry) for key, entry in item.items()}
            if isinstance(item, tuple):
                return tuple(visit(entry) for entry in item)
            if isinstance(item, list):
                return [visit(entry) for entry in item]
            return item

        return visit(value)

    def unpack(self, value):
        if isinstance(value, tuple) and value and value[0] == "shared_array":
            _, offset, shape, dtype = value
            # Returned tensors must survive arbitrary subsequent env.step calls.
            return np.ndarray(shape, dtype=np.dtype(dtype), buffer=self.memory.buf, offset=offset).copy()
        if isinstance(value, dict):
            return {key: self.unpack(entry) for key, entry in value.items()}
        if isinstance(value, tuple):
            return tuple(self.unpack(entry) for entry in value)
        if isinstance(value, list):
            return [self.unpack(entry) for entry in value]
        return value

    def close(self):
        self.memory.close()
        if self.owner:
            self.memory.unlink()


class _Connection:
    """Framed local socket with deadlines covering every byte of an operation."""

    def __init__(self, endpoint):
        self.socket = endpoint
        self.socket.setblocking(False)

    def _wait(self, writing, deadline):
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Collector communication timed out")
        readable, writable, _ = select.select(
            [] if writing else [self.socket], [self.socket] if writing else [], [], remaining
        )
        if not readable and not writable:
            raise TimeoutError("Collector communication timed out")

    def send(self, value, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self.last_send_bytes = len(payload) + 8
        for part in (struct.pack("!Q", len(payload)), payload):
            view = memoryview(part)
            while view:
                self._wait(True, deadline)
                try:
                    sent = self.socket.send(view)
                except BlockingIOError:
                    continue
                if sent == 0:
                    raise EOFError("Collector connection closed")
                view = view[sent:]

    def recv(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout

        def read(size):
            data = bytearray(size)
            view = memoryview(data)
            while view:
                self._wait(False, deadline)
                try:
                    count = self.socket.recv_into(view)
                except BlockingIOError:
                    continue
                if count == 0:
                    raise EOFError("Collector connection closed")
                view = view[count:]
            return data

        size = struct.unpack("!Q", read(8))[0]
        self.last_receive_bytes = size + 8
        return pickle.loads(read(size))

    def close(self):
        self.socket.close()


@dataclass
class EnvironmentFactory:
    task_class: type
    cfg: object

    def __call__(self):
        return self.task_class(cfg=self.cfg, physics_engine="mujoco", sim_device="cpu", headless=True)


def _worker(endpoint, factory, profiling):
    connection = _Connection(endpoint)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    random.seed(factory.cfg.seed)
    np.random.seed(factory.cfg.seed)
    # Do not initialize an accelerator in the collector.
    torch.random.default_generator.manual_seed(factory.cfg.seed)
    env = None
    inbound = outbound = None
    try:
        env = factory()
        env.capture_terminal_observations = True
        if profiling:
            from .off_policy_profile import collector_timers

            step_timings = collector_timers(env)
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
            if inbound is not None:
                payload = inbound.unpack(payload)
            if command == "shared":
                inbound = _SharedArrays(name=payload["inbound"])
                outbound = _SharedArrays(name=payload["outbound"])
                connection.send(("ok", None))
                continue
            if command == "close":
                break
            if command == "step":
                if profiling:
                    step_timings.clear()
                before = time.perf_counter() if profiling else 0.0
                result = env.step(_decode(payload))
                step_snapshot = {"lengths": env.episode_length_buf, "state": env.training_state_dict()}
                if profiling:
                    step_snapshot["collector_step_s"] = time.perf_counter() - before
                    step_snapshot["timings"] = dict(step_timings)
                response = (result, step_snapshot)
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
            encoded = _encode(response)
            if outbound is not None:
                encoded = outbound.pack(encoded)
            connection.send(("ok", encoded))
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
        for slot in (inbound, outbound):
            if slot is not None:
                slot.close()


class ProcessEnvironment:
    """CPU VecEnv proxy. Call close() in the caller's finally block."""

    def __init__(self, factory, timeout=120.0, transport="shared_memory", profiling=False):
        if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("collector_timeout must be finite and positive")
        if transport not in ("socket", "shared_memory"):
            raise ValueError("collector_transport must be socket or shared_memory")
        self.transport = transport
        self.profiling = profiling
        self.last_step_metrics = {}
        self._shared_in = self._shared_out = None
        self.cfg = factory.cfg
        self.device = torch.device("cpu")
        self.headless = True
        self.timeout = timeout
        self._closed = False
        self._pending = False
        self._broken = False
        context = mp.get_context("spawn")
        parent, child = socket.socketpair()
        self._connection = _Connection(parent)
        self.process = context.Process(
            target=_worker, args=(child, factory, profiling), name="legged-gym-collector", daemon=True
        )
        try:
            self.process.start()
            child.close()
            metadata, snapshot = self._receive()
            sim = metadata.pop("sim")
            self.__dict__.update(metadata)
            self.sim = SimpleNamespace(asset_digest=sim["asset_digest"], batch=SimpleNamespace(nthread=sim["nthread"]))
            self._accept(snapshot)
            if transport == "shared_memory":
                obs_bytes = sum(v.numel() * v.element_size() for v in self.obs_buf.values())
                size = max(65536, 4 * obs_bytes + self.num_envs * (self.num_actions * 4 + 128))
                self._shared_out = _SharedArrays(size=max(65536, self.num_envs * self.num_actions * 4 + 4096))
                self._shared_in = _SharedArrays(size=size)
                self._send("shared", {"inbound": self._shared_out.memory.name, "outbound": self._shared_in.memory.name})
                self._receive()
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
        if self._closed:
            raise RuntimeError("Environment is closed")
        try:
            status, payload = self._connection.recv(timeout=self.timeout)
            if status == "error":
                raise RuntimeError(f"Collector failed:\n{payload}")
            if self._shared_in is not None:
                payload = self._shared_in.unpack(payload)
            return _decode(payload)
        except BaseException:
            # A partially received message cannot be retried on this channel.
            self._broken = True
            self.close()
            raise

    def _send(self, command, payload=None):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._pending:
            raise RuntimeError("Receive the pending collector step before issuing another command")
        try:
            encoded = _encode(payload)
            if self._shared_out is not None:
                encoded = self._shared_out.pack(encoded)
            self._connection.send((command, encoded), timeout=self.timeout)
        except BaseException:
            self._broken = True
            self.close()
            raise

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
            if self.profiling:
                self.last_step_metrics = {
                    "collector_step": snapshot["collector_step_s"],
                    "control_bytes": self._connection.last_send_bytes + self._connection.last_receive_bytes,
                    **snapshot["timings"],
                }
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
            if self.process.is_alive() and not self._pending and not self._broken:
                self._connection.send(("close", None), timeout=min(self.timeout, 0.1))
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
        for name in ("_shared_in", "_shared_out"):
            slot = getattr(self, name)
            if slot is not None:
                slot.close()
                setattr(self, name, None)

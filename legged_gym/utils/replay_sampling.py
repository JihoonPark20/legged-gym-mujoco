"""Reusable sample buffers for the pinned SAC backends; replay stays backend-owned.

Installed only for a runner's learn() call. A batch lives until its slot is reused
two samples later, after the supported algorithms have completed backward().
No speculative sampling is done: RNG draw order and replay visibility are kept.
"""

import warnings
from contextlib import contextmanager

import torch
from tensordict import TensorDict


class _GatherSlots:
    def __init__(self):
        self.slots = [{}, {}]
        self.cursor = 0

    def begin(self):
        self.current = self.slots[self.cursor]
        self.cursor = 1 - self.cursor

    def gather(self, name, source, indices):
        shape = (indices.numel(), *source.shape[1:])
        out = self.current.get(name)
        if out is None or out.shape != shape or out.dtype != source.dtype or out.device != source.device:
            out = self.current[name] = torch.empty(shape, device=source.device, dtype=source.dtype)
        torch.index_select(source, 0, indices, out=out)
        return out


class _SACSampler(_GatherSlots):
    def __init__(self, storage, profiler):
        super().__init__()
        self.storage, self.profiler = storage, profiler

    def generator(self, num_mini_batch, mini_batch_size, num_epochs=1):
        b = self.storage
        total = b.num_envs * b.num_transitions
        if total < 1:
            raise ValueError("Replay contains no transitions")
        size = min(mini_batch_size, total)
        if size < mini_batch_size:
            warnings.warn(f"Replay has only {total} transitions; using batch size {size}", RuntimeWarning)
        for _ in range(num_epochs):
            for _ in range(num_mini_batch):
                with self.profiler.measure("replay_sample"), torch.no_grad():
                    self.begin()
                    # Same order as the backend's flattened env/time meshgrid,
                    # without materializing a capacity-sized grid every update.
                    sampled = torch.randint(total, (size,), device=b.device)
                    indices = sampled.div(b.num_transitions, rounding_mode="floor") * b.buffer_size
                    indices += sampled.remainder(b.num_transitions)

                    def gather(name, tensor):
                        return self.gather(name, tensor.flatten(0, 1), indices)

                    obs = TensorDict({k: gather("obs/" + k, v) for k, v in b.observations.items()}, [size])
                    nxt = TensorDict({k: gather("next/" + k, v) for k, v in b.next_observations.items()}, [size])
                    steps = self.current.get("steps")
                    if steps is None or steps.shape[0] != size:
                        steps = self.current["steps"] = torch.ones(size, 1, device=b.device, dtype=torch.long)
                    batch = [
                        obs,
                        gather("actions", b.actions),
                        gather("rewards", b.rewards),
                        nxt,
                        gather("dones", b.dones),
                        gather("bootstrap", b.bootstrap),
                        steps,
                    ]
                yield batch


class _FlashSampler(_GatherSlots):
    def __init__(self, storage, profiler):
        super().__init__()
        self.storage, self.profiler = storage, profiler
        self.memory_efficient = type(storage).__name__ == "MemoryEfficientTorchUniformBuffer"

    def sample(self, sample_idxs=None):
        b = self.storage
        with self.profiler.measure("replay_sample"), torch.no_grad():
            self.begin()
            if sample_idxs is None:
                high = b._num_in_buffer - b._n_step * b._add_batch_size if self.memory_efficient else b._num_in_buffer
                idx = torch.randint(0, high, (b._sample_batch_size,), device=b._device)
                if self.memory_efficient and b._num_in_buffer == b._max_length:
                    idx = (idx + b._current_idx) % b._max_length
            else:
                idx = torch.as_tensor(sample_idxs, device=b._device, dtype=torch.long)

            def observations(name, storage, indices):
                values = {key: self.gather(name + key, storage[key], indices) for key in b._store_groups}
                if b._obs_storage_dtype is not None:
                    values = {key: value.float() for key, value in values.items()}
                return TensorDict(values, [idx.numel()], device=b._device)

            batch = {"observation": observations("obs/", b._observations, idx)}
            for key, source in (
                ("action", b._actions),
                ("reward", b._rewards),
                ("terminated", b._terminateds),
                ("truncated", b._truncateds),
            ):
                batch[key] = self.gather(key, source, idx)
            if self.memory_efficient:
                next_idx = (idx + b._n_step * b._add_batch_size) % b._max_length
                nxt = observations("next/", b._observations, next_idx)
                # Keep the backend's exact terminal-state repair semantics.
                if b._episode_end_next_observations:
                    hits = [
                        (pos, b._episode_end_next_observations[int(i)])
                        for pos, i in enumerate(idx.detach().cpu().tolist())
                        if int(i) in b._episode_end_next_observations
                    ]
                    if hits:
                        positions = torch.tensor([pos for pos, _ in hits], device=b._device)
                        for key in b._store_groups:
                            nxt[key][positions] = torch.stack([obs[key] for _, obs in hits]).to(nxt[key])
            else:
                nxt = observations("next/", b._next_observations, idx)
            batch["next_observation"] = nxt
            return batch


def _sampler_class(algorithm):
    if getattr(algorithm, "use_compile", False):
        return None
    storage = getattr(algorithm, "replay_buffer", None)
    cls = type(storage)
    if cls.__module__ == "rsl_rl.storage.replay_buffer" and cls.__name__ == "ReplayBuffer" and storage.n_steps == 1:
        return _SACSampler
    if cls.__module__ == "rsl_rl_flashsac.storage.torch_buffer" and cls.__name__ in (
        "TorchUniformBuffer",
        "MemoryEfficientTorchUniformBuffer",
    ):
        return _FlashSampler
    return None


def resolve_sampling_mode(algorithm, mode):
    """Select only the buffer implementations whose ownership rules we support."""
    if mode not in ("auto", "native", "reusable"):
        raise ValueError(f"Invalid replay sampling mode: {mode}")
    return "reusable" if mode != "native" and _sampler_class(algorithm) is not None else "native"


@contextmanager
def sampling_session(algorithm, mode, profiler):
    """Fall back to native sampling for n-step SAC, compilation or custom storage."""
    mode = resolve_sampling_mode(algorithm, mode)
    profiler.sampling_mode = mode
    if mode == "native" and not profiler.enabled:
        yield
        return
    storage = algorithm.replay_buffer
    method = "sample" if callable(getattr(storage, "sample", None)) else "mini_batch_generator"
    if not callable(getattr(storage, method, None)):
        yield
        return
    original = getattr(storage, method)
    replacement = None
    if mode == "reusable":
        sampler = _sampler_class(algorithm)(storage, profiler)
        replacement = sampler.generator if method == "mini_batch_generator" else sampler.sample
    profiler.sampling_mode = "reusable" if replacement is not None else "native"
    if replacement is None and profiler.enabled:
        if method == "sample":

            def replacement(*args, **kwargs):
                with profiler.measure("replay_sample"):
                    return original(*args, **kwargs)
        else:

            def replacement(*args, **kwargs):
                iterator = iter(original(*args, **kwargs))
                while True:
                    with profiler.measure("replay_sample"):
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            return
                    yield batch

    if replacement is None:
        yield
        return
    previous = storage.__dict__.get(method)
    setattr(storage, method, replacement)
    try:
        yield
    finally:
        if previous is None:
            delattr(storage, method)
        else:
            setattr(storage, method, previous)

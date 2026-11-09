"""Opt-in host timing, with explicit device synchronization only in diagnostic mode."""

import csv
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch


def collector_timers(env):
    """Instrument the spawned collector only; no hooks in the normal env path."""
    totals = defaultdict(float)

    def measured(original, key):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                totals[key] += time.perf_counter() - start

        return call

    for owner, method, key in (
        (env.sim, "step", "physics"),
        (env, "compute_observations", "observations"),
        (env, "compute_reward", "reward"),
    ):
        original = getattr(owner, method, None)
        if callable(original):
            setattr(owner, method, measured(original, key))
    return totals


class OffPolicyProfiler:
    stages = (
        "inference",
        "action_transfer",
        "dispatch",
        "collector_step",
        "physics",
        "observations",
        "reward",
        "receive",
        "observation_transfer",
        "replay_insert",
        "replay_sample",
        "update",
        "iteration",
    )

    def __init__(self, mode, device):
        self.enabled = mode != "off"
        self.mode, self.device = mode, torch.device(device)
        self.sampling_mode = "native"
        self.totals = defaultdict(float)
        self.last_record = None

    def sync(self):
        if self.mode == "synchronized":
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, stage):
        if not self.enabled:
            yield
            return
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.totals[stage] += time.perf_counter() - start

    def record(self, iteration, count, logger):
        if not self.enabled:
            return
        # Nested stages (sample/update) and the child physics duration overlap;
        # these columns must not be summed to estimate wall time.
        record = {"iteration": iteration, "iterations": count, "mode": self.mode, "sampling": self.sampling_mode}
        record.update({stage + "_ms": self.totals[stage] * 1000 / count for stage in self.stages})
        record["control_bytes"] = self.totals["control_bytes"] / count
        self.last_record = record
        if logger.writer is not None:
            for key, value in record.items():
                if key.endswith("_ms") or key == "control_bytes":
                    logger.writer.add_scalar("Perf/" + key, value, iteration)
        if logger.log_dir is not None:
            path = Path(logger.log_dir) / "performance.csv"
            header = not path.exists()
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=record.keys())
                if header:
                    writer.writeheader()
                writer.writerow(record)
        self.totals.clear()

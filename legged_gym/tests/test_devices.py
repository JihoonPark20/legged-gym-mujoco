"""Device selection checks run even without CUDA hardware; no GPU tensors are mocked."""

import pytest
import torch

from legged_gym.utils.mujoco_helpers import get_args, resolve_rl_device


@pytest.fixture
def hardware(monkeypatch):
    def configure(cuda=False, mps=False, count=2, current=0):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count if cuda else 0)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    return configure


@pytest.mark.parametrize(
    "requested,expected",
    [("cuda", "cuda:1"), ("cuda:0", "cuda:0"), ("cuda:1", "cuda:1"), (torch.device("cuda:1"), "cuda:1")],
)
def test_cuda_selection_preserves_visible_device_ordinal(hardware, requested, expected):
    hardware(cuda=True, current=1)
    assert resolve_rl_device(requested) == expected


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda:0"), (True, False, "cuda:0"), (False, True, "mps"), (False, False, "cpu")]
)
def test_default_and_explicit_auto_selection(hardware, cuda, mps, expected):
    hardware(cuda=cuda, mps=mps)
    assert resolve_rl_device("auto") == expected
    assert resolve_rl_device() == expected
    assert get_args([]).rl_device == expected


def test_explicit_selection_overrides_auto_and_collection_stays_on_cpu(hardware):
    hardware(cuda=True, mps=True)
    assert get_args([]).rl_device == "cuda:0"
    assert get_args(["--rl_device", "mps"]).rl_device == "mps"
    assert get_args(["--rl_device", "cpu"]).rl_device == "cpu"
    assert get_args(["--rl_device", "cuda:1"]).rl_device == "cuda:1"
    assert get_args([], default_rl_device="cpu").rl_device == "cpu"
    hardware()
    assert resolve_rl_device() == "cpu"
    assert get_args([], default_rl_device="cpu").rl_device == "cpu"


def test_cli_accepts_cuda_and_keeps_mujoco_on_cpu(hardware):
    hardware(cuda=True)
    args = get_args(["--rl_device", "cuda:1"])
    assert args.rl_device == "cuda:1" and args.sim_device == "cpu"
    with pytest.raises(SystemExit):
        get_args(["--sim_device", "cuda"])


@pytest.mark.parametrize("device", ["cuda:2", "cuda:-1", "cuda:x", "cuda:0:1", "mps:1", "cpu:1", "xpu", "meta", ""])
def test_invalid_or_out_of_range_devices_are_rejected(hardware, device):
    hardware(cuda=True, mps=True)
    with pytest.raises(ValueError):
        resolve_rl_device(device)


@pytest.mark.parametrize("device", ["mps", "cuda", "cuda:0"])
def test_explicit_unavailable_accelerator_does_not_silently_fall_back(hardware, device):
    hardware()
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_rl_device(device)
    with pytest.raises(SystemExit):
        get_args(["--rl_device", device])

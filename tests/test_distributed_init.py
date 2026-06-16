"""Tests for misc.init_distributed_mode single-process fallbacks.

Regression: running `mamp-ml predict` on a SLURM compute node crashed with
``ValueError: ... MASTER_ADDR expected, but not set`` because SLURM_PROCID was
present in the environment, so init_distributed_mode tried to bring up a
torch.distributed process group for what is really a single-process inference
run. It must instead detect that there's no genuine distributed launch and run
non-distributed.
"""

from __future__ import annotations

import types

import pytest

# init_distributed_mode lives in misc, which imports torch.
torch = pytest.importorskip("torch")
from mamp_ml import misc  # noqa: E402


_DIST_ENV_VARS = [
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
]


@pytest.fixture
def clean_dist_env(monkeypatch):
    """Strip any ambient distributed env vars so tests are deterministic, and
    stub setup_for_distributed (which otherwise monkeypatches builtins.print)."""
    for var in _DIST_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(misc, "setup_for_distributed", lambda *a, **k: None)
    return monkeypatch


def _args():
    return types.SimpleNamespace(
        dist_on_itp=False,
        dist_url="env://",
        distributed=None,
        rank=None,
        world_size=1,
        gpu=None,
        dist_backend="nccl",
    )


def _forbid_process_group(monkeypatch):
    """Make any attempt to actually init a process group fail the test loudly."""
    def boom(*a, **k):  # pragma: no cover - only hit on regression
        raise AssertionError("init_process_group must NOT be called in single-process mode")

    monkeypatch.setattr(torch.distributed, "init_process_group", boom)


def test_no_distributed_env_runs_single_process(clean_dist_env) -> None:
    _forbid_process_group(clean_dist_env)
    args = _args()
    misc.init_distributed_mode(args)
    assert args.distributed is False


def test_slurm_single_task_is_not_distributed(clean_dist_env) -> None:
    """SLURM_PROCID present but a single task -> single-process (the bug case)."""
    _forbid_process_group(clean_dist_env)
    clean_dist_env.setenv("SLURM_PROCID", "0")
    clean_dist_env.setenv("SLURM_NTASKS", "1")
    args = _args()
    misc.init_distributed_mode(args)
    assert args.distributed is False


def test_slurm_procid_without_ntasks_is_not_distributed(clean_dist_env) -> None:
    """SLURM_PROCID set, SLURM_NTASKS unset (defaults to 1) -> single-process."""
    _forbid_process_group(clean_dist_env)
    clean_dist_env.setenv("SLURM_PROCID", "0")
    args = _args()
    misc.init_distributed_mode(args)
    assert args.distributed is False


def test_multitask_without_master_addr_falls_back(clean_dist_env) -> None:
    """Even a multi-task job with env:// but no MASTER_ADDR must not crash."""
    _forbid_process_group(clean_dist_env)
    clean_dist_env.setenv("SLURM_PROCID", "0")
    clean_dist_env.setenv("SLURM_NTASKS", "2")
    # A genuine multi-task SLURM job runs on GPU nodes; simulate that so the
    # branch's `rank % device_count()` doesn't divide by zero on a CPU-only
    # test host. MASTER_ADDR is intentionally absent.
    clean_dist_env.setattr(torch.cuda, "device_count", lambda: 2)
    args = _args()
    misc.init_distributed_mode(args)
    assert args.distributed is False


def test_real_torchrun_launch_initialises_distributed(clean_dist_env) -> None:
    """A genuine torchrun launch (RANK/WORLD_SIZE/LOCAL_RANK + MASTER_ADDR)
    still brings up the process group."""
    clean_dist_env.setenv("RANK", "0")
    clean_dist_env.setenv("WORLD_SIZE", "1")
    clean_dist_env.setenv("LOCAL_RANK", "0")
    clean_dist_env.setenv("MASTER_ADDR", "127.0.0.1")
    clean_dist_env.setenv("MASTER_PORT", "29500")

    called = {}

    def fake_init(*a, **k):
        called["init"] = k

    clean_dist_env.setattr(torch.distributed, "init_process_group", fake_init)
    clean_dist_env.setattr(torch.distributed, "barrier", lambda *a, **k: None)
    clean_dist_env.setattr(torch.cuda, "set_device", lambda *a, **k: None)

    args = _args()
    misc.init_distributed_mode(args)
    assert args.distributed is True
    assert called["init"]["world_size"] == 1
    assert called["init"]["rank"] == 0

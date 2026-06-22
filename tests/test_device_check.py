"""GPU compatibility preflight + install-check command.

Guards the V100/sm_70 trap: a PyTorch build without kernels for the GPU's
compute capability must be detected *before* inference (and fall back to CPU)
rather than crashing deep in the forward pass with
``cudaErrorNoKernelImageForDevice``.
"""

from __future__ import annotations

import types

import pytest

from mamp_ml.__main__ import (
    _gpu_compatibility_problem,
    _resolve_inference_device,
    main as cli_main,
)


# ---------------------------------------------------------------------------
# _gpu_compatibility_problem
# ---------------------------------------------------------------------------


def test_non_cuda_devices_have_no_problem() -> None:
    assert _gpu_compatibility_problem(None) is None
    assert _gpu_compatibility_problem("cpu") is None
    assert _gpu_compatibility_problem("mps") is None


def _fake_torch(monkeypatch, *, available, capability=(7, 0), arches=None, name="Tesla V100"):
    """Install a fake ``torch`` with a controllable CUDA surface."""
    import mamp_ml.__main__ as cli

    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda i=0: capability,
        get_device_name=lambda i=0: name,
        get_arch_list=lambda: arches if arches is not None else [],
        device_count=lambda: 1,
    )
    fake = types.ModuleType("torch")
    fake.cuda = cuda
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    return fake


def test_cuda_unavailable_is_flagged(monkeypatch) -> None:
    _fake_torch(monkeypatch, available=False)
    problem = _gpu_compatibility_problem("cuda")
    assert problem and "is_available" in problem


def test_unsupported_compute_capability_is_flagged(monkeypatch) -> None:
    # V100 = CC 7.0 / sm_70, torch built only for sm_75+ -> unusable.
    _fake_torch(
        monkeypatch,
        available=True,
        capability=(7, 0),
        arches=["sm_75", "sm_80", "sm_86", "sm_90"],
        name="Tesla V100-SXM2-32GB",
    )
    problem = _gpu_compatibility_problem("cuda")
    assert problem is not None
    assert "sm_70" in problem
    assert "V100" in problem


def test_supported_gpu_has_no_problem(monkeypatch) -> None:
    _fake_torch(
        monkeypatch,
        available=True,
        capability=(8, 0),
        arches=["sm_70", "sm_75", "sm_80"],
        name="A100",
    )
    assert _gpu_compatibility_problem("cuda") is None


# ---------------------------------------------------------------------------
# _resolve_inference_device
# ---------------------------------------------------------------------------


def test_resolve_falls_back_to_cpu_on_incompatible_gpu(monkeypatch, capsys) -> None:
    _fake_torch(
        monkeypatch,
        available=True,
        capability=(7, 0),
        arches=["sm_75", "sm_80"],
        name="Tesla V100",
    )
    assert _resolve_inference_device("cuda") == "cpu"
    out = capsys.readouterr().out
    assert "cpu" in out.lower()


def test_resolve_keeps_compatible_gpu(monkeypatch) -> None:
    _fake_torch(monkeypatch, available=True, capability=(8, 0), arches=["sm_80"])
    assert _resolve_inference_device("cuda") == "cuda"


def test_resolve_passes_through_cpu() -> None:
    assert _resolve_inference_device("cpu") == "cpu"
    assert _resolve_inference_device(None) == "cpu"


# ---------------------------------------------------------------------------
# install-check command
# ---------------------------------------------------------------------------


def test_install_check_runs_and_reports(capsys) -> None:
    rc = cli_main(["install-check"])
    assert rc in (0, 1)  # 1 only if a hard check fails on this host
    out = capsys.readouterr().out
    assert "install-check" in out
    assert "PyTorch" in out

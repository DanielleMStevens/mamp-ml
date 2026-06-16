"""Tests for the ColabFold install discovery helpers + CLI.

The helpers walk a curated set of common install locations on the host.
We exercise each search branch with a synthetic filesystem under
``tmp_path`` so the tests don't depend on whether the dev's machine
happens to have ColabFold installed.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from mamp_ml.fold.colabfold import (
    build_colabfold_command,
    find_colabfold_installs,
    format_activation_hint,
    run_colabfold_batch,
)


def _make_executable(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


# ---------------------------------------------------------------------------
# format_activation_hint
# ---------------------------------------------------------------------------


def test_format_activation_hint_emits_export_path_line(tmp_path: Path) -> None:
    binary = _make_executable(tmp_path / "bin" / "colabfold_batch")
    hint = format_activation_hint(binary)
    assert hint == f'export PATH="{tmp_path / "bin"}:$PATH"'


# ---------------------------------------------------------------------------
# find_colabfold_installs — explicit search branches via monkeypatch
# ---------------------------------------------------------------------------


def test_returns_empty_when_nothing_found(monkeypatch, tmp_path: Path) -> None:
    """If nothing matches the search candidates, return an empty list."""
    monkeypatch.setenv("PATH", str(tmp_path))         # PATH points at an empty dir
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    # Block every candidate location by pointing them at non-existent paths.
    import mamp_ml.fold.colabfold as mod
    monkeypatch.setattr(mod, "_CONDA_ROOT_CANDIDATES", ())
    monkeypatch.setattr(mod, "_LOCALCOLABFOLD_PARENT_CANDIDATES", ())

    found = find_colabfold_installs()
    assert found == []


def test_finds_install_on_PATH(monkeypatch, tmp_path: Path) -> None:
    """A binary on $PATH must be found via shutil.which."""
    bin_dir = tmp_path / "bin"
    binary = _make_executable(bin_dir / "colabfold_batch")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    import mamp_ml.fold.colabfold as mod
    monkeypatch.setattr(mod, "_CONDA_ROOT_CANDIDATES", ())
    monkeypatch.setattr(mod, "_LOCALCOLABFOLD_PARENT_CANDIDATES", ())

    found = find_colabfold_installs()
    assert len(found) == 1
    path, source = found[0]
    assert path == binary.resolve()
    assert "PATH" in source


def test_finds_install_under_conda_root_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    """A binary under a known conda root's envs/ tree must be found."""
    # Build  fake conda root  with  one env containing colabfold_batch.
    fake_conda = tmp_path / "miniconda"
    binary = _make_executable(
        fake_conda / "envs" / "localfold" / "bin" / "colabfold_batch"
    )

    monkeypatch.setenv("PATH", str(tmp_path))   # empty
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    import mamp_ml.fold.colabfold as mod
    monkeypatch.setattr(mod, "_CONDA_ROOT_CANDIDATES", (str(fake_conda),))
    monkeypatch.setattr(mod, "_LOCALCOLABFOLD_PARENT_CANDIDATES", ())

    found = find_colabfold_installs()
    assert len(found) == 1
    path, source = found[0]
    assert path == binary.resolve()
    assert "localfold" in source


def test_finds_install_under_localcolabfold_layout(
    monkeypatch, tmp_path: Path
) -> None:
    """The localcolabfold/colabfold-conda/bin layout is found."""
    binary = _make_executable(
        tmp_path / "localcolabfold" / "colabfold-conda" / "bin" / "colabfold_batch"
    )

    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    import mamp_ml.fold.colabfold as mod
    monkeypatch.setattr(mod, "_CONDA_ROOT_CANDIDATES", ())
    monkeypatch.setattr(mod, "_LOCALCOLABFOLD_PARENT_CANDIDATES", (str(tmp_path),))

    found = find_colabfold_installs()
    assert len(found) == 1
    path, source = found[0]
    assert path == binary.resolve()
    assert "localcolabfold" in source


def test_deduplicates_across_search_branches(
    monkeypatch, tmp_path: Path
) -> None:
    """When the SAME binary is reachable via multiple branches (e.g. both
    $PATH and a conda env), it appears once in the output."""
    bin_dir = tmp_path / "localcolabfold" / "colabfold-conda" / "bin"
    _make_executable(bin_dir / "colabfold_batch")
    # Also expose the same dir on PATH.
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    import mamp_ml.fold.colabfold as mod
    monkeypatch.setattr(mod, "_CONDA_ROOT_CANDIDATES", ())
    monkeypatch.setattr(
        mod, "_LOCALCOLABFOLD_PARENT_CANDIDATES", (str(tmp_path),)
    )

    found = find_colabfold_installs()
    assert len(found) == 1


def test_search_tolerates_permission_errors(monkeypatch, tmp_path: Path) -> None:
    """An unreadable conda root directory must not crash the search."""
    blocked = tmp_path / "unreadable"
    (blocked / "envs").mkdir(parents=True)
    # Chmod 000 to force PermissionError on iterdir.
    os.chmod(blocked / "envs", 0)
    try:
        monkeypatch.setenv("PATH", "")
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        import mamp_ml.fold.colabfold as mod
        monkeypatch.setattr(mod, "_CONDA_ROOT_CANDIDATES", (str(blocked),))
        monkeypatch.setattr(mod, "_LOCALCOLABFOLD_PARENT_CANDIDATES", ())

        found = find_colabfold_installs()
        # Doesn't crash; returns empty since we didn't add any binary.
        assert found == []
    finally:
        # Restore so pytest can clean up the tmp dir.
        os.chmod(blocked / "envs", 0o700)


# ---------------------------------------------------------------------------
# build_colabfold_command / run_colabfold_batch
# ---------------------------------------------------------------------------


def test_build_colabfold_command_layout(tmp_path: Path) -> None:
    """The argv puts flags first and the positional input/output last, exactly
    as ColabFold expects."""
    binary = tmp_path / "colabfold_batch"
    cmd = build_colabfold_command(
        binary, tmp_path / "in.fasta", tmp_path / "out", num_models=1, num_recycle=1
    )
    assert cmd == [
        str(binary),
        "--num-models",
        "1",
        "--num-recycle",
        "1",
        str(tmp_path / "in.fasta"),
        str(tmp_path / "out"),
    ]


def test_build_colabfold_command_inserts_extra_args(tmp_path: Path) -> None:
    """extra_args land before the positional input/output."""
    cmd = build_colabfold_command(
        "colabfold_batch",
        "in.fasta",
        "out",
        num_models=2,
        num_recycle=3,
        extra_args=["--amber"],
    )
    assert cmd == [
        "colabfold_batch",
        "--num-models",
        "2",
        "--num-recycle",
        "3",
        "--amber",
        "in.fasta",
        "out",
    ]


def test_run_colabfold_batch_invokes_subprocess_and_returns_code(
    monkeypatch, tmp_path: Path
) -> None:
    """run_colabfold_batch shells out by absolute path and propagates the exit
    code; output is streamed (not captured)."""
    import subprocess
    import mamp_ml.fold.colabfold as cf

    recorded: dict = {}

    class _Completed:
        returncode = 7

    def fake_run(cmd, check=False):
        recorded["cmd"] = cmd
        recorded["check"] = check
        return _Completed()

    monkeypatch.setattr(subprocess, "run", fake_run)

    binary = tmp_path / "bin" / "colabfold_batch"
    rc = run_colabfold_batch(binary, tmp_path / "in.fasta", tmp_path / "out")
    assert rc == 7
    assert recorded["cmd"][0] == str(binary)
    assert recorded["check"] is False
    # No stdout/stderr capture kwargs -> ColabFold streams to the terminal.
    assert recorded["cmd"][-2:] == [str(tmp_path / "in.fasta"), str(tmp_path / "out")]


def test_run_colabfold_batch_returns_127_when_binary_unexecutable(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """If the binary can't be launched (OSError), we return 127 and explain."""
    import subprocess
    import mamp_ml.fold.colabfold as cf

    def boom(cmd, check=False):
        raise OSError("Exec format error")

    monkeypatch.setattr(subprocess, "run", boom)

    rc = run_colabfold_batch(tmp_path / "colabfold_batch", "in.fasta", "out")
    assert rc == 127
    assert "Failed to execute" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLI: `mamp-ml find-colabfold`
# ---------------------------------------------------------------------------


def test_cli_find_colabfold_returns_1_when_nothing_found(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """When the search returns empty, the subcommand exits with code 1
    and prints install instructions."""
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    import mamp_ml.fold.colabfold as mod
    monkeypatch.setattr(mod, "_CONDA_ROOT_CANDIDATES", ())
    monkeypatch.setattr(mod, "_LOCALCOLABFOLD_PARENT_CANDIDATES", ())

    from mamp_ml.__main__ import main as cli_main

    rc = cli_main(["find-colabfold"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "No `colabfold_batch` install found" in out
    assert "install_colabbatch_linux.sh" in out


def test_cli_find_colabfold_returns_0_when_found(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """When at least one install is found, the subcommand exits with code 0
    and prints both the path and the activation hint."""
    bin_dir = tmp_path / "bin"
    binary = _make_executable(bin_dir / "colabfold_batch")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    import mamp_ml.fold.colabfold as mod
    monkeypatch.setattr(mod, "_CONDA_ROOT_CANDIDATES", ())
    monkeypatch.setattr(mod, "_LOCALCOLABFOLD_PARENT_CANDIDATES", ())

    from mamp_ml.__main__ import main as cli_main

    rc = cli_main(["find-colabfold"])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(binary.resolve()) in out
    assert f'export PATH="{bin_dir}:$PATH"' in out

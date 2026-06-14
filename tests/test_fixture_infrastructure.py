"""Smoke tests for the checkpoint-4 fixture-generation infrastructure.

The actual golden tests that *use* the ColabFold outputs land in checkpoints
4b through 4e. This file only validates that the infrastructure itself is
wired correctly:

- the fixture-generation script exists, is executable, and is syntactically
  valid bash (so it will at least parse on a developer's machine)
- the .gitignore covers the path where the script writes its outputs so a
  developer who runs the script won't accidentally commit ~30 MB of PDBs
- the ``colabfold_outputs_dir`` pytest fixture skips cleanly with a useful
  message when the outputs are absent (which is the default state in CI)

These tests run on every machine, in seconds, with no external dependencies.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_fixture_script_exists_and_is_executable(fixtures_dir: Path) -> None:
    script = fixtures_dir / "build_colabfold_fixtures.sh"
    assert script.is_file(), f"missing fixture script: {script}"
    # Owner-execute bit must be set so `bash <script>` and `./script` both work.
    assert script.stat().st_mode & 0o100, (
        f"{script} is not executable; run `chmod +x {script}`"
    )


def test_fixture_script_passes_bash_syntax_check(fixtures_dir: Path) -> None:
    """``bash -n`` parses the script without executing it; catches typos."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available on this machine")
    script = fixtures_dir / "build_colabfold_fixtures.sh"
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"shell syntax error in {script}:\n{result.stderr}"
    )


def test_gitignore_covers_colabfold_output_path(repo_root: Path) -> None:
    """The .gitignore must shield the heavy fixture artefacts from accidental commit."""
    gitignore = repo_root / ".gitignore"
    assert gitignore.is_file(), "repository .gitignore is missing"
    contents = gitignore.read_text()
    # The colabfold_output/ directory under tests/fixtures/ is the one place
    # the fixture script writes to.
    assert "tests/fixtures/colabfold_output/" in contents, (
        ".gitignore does not exclude tests/fixtures/colabfold_output/"
    )


def test_colabfold_outputs_dir_fixture_skips_cleanly_when_absent(
    fixtures_dir: Path,
) -> None:
    """When colabfold_output/ is not present, the fixture must skip with an
    actionable message — not raise, not return a non-existent path."""
    # Sentinel: directory must NOT exist when this test runs in CI.
    candidate = fixtures_dir / "colabfold_output"
    if candidate.exists():
        pytest.skip(
            "colabfold_output exists locally; this test only meaningfully "
            "covers the absent-fixture path on CI / a fresh clone"
        )
    # Simulate a test that *depends* on the fixture by re-invoking pytest
    # against an inline ad-hoc test, and confirm the outcome is `skipped`.
    inline = f"""
import pytest

def test_dependent(colabfold_outputs_dir):
    assert colabfold_outputs_dir.is_dir()
"""
    tmpfile = fixtures_dir.parent / "_inline_skip_probe.py"
    tmpfile.write_text(inline)
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", str(tmpfile), "-q", "--no-header"],
            cwd=fixtures_dir.parent.parent,
            capture_output=True,
            text=True,
            env={
                **__import__("os").environ,
                "PYTHONPATH": str(fixtures_dir.parent.parent / "src"),
            },
            timeout=60,
        )
    finally:
        tmpfile.unlink(missing_ok=True)
    # `pytest -q` reports skip with exit code 0 and "skipped" in stdout.
    assert "skipped" in result.stdout.lower(), (
        f"expected fixture to skip the inline test, got:\n{result.stdout}\n{result.stderr}"
    )
    assert "build_colabfold_fixtures.sh" in result.stdout, (
        "skip message must point developers at the fixture-generation script"
    )

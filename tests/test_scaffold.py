"""Smoke tests for the package scaffolding.

These checks confirm that the source layout is wired up correctly:
- the package can be imported
- it advertises a sensible version string
- a few well-known on-disk assets are reachable from the test environment

Real behavioural tests (golden-output diffs against the legacy R + shell
pipeline) land alongside their corresponding rewrites in later checkpoints.
"""

from __future__ import annotations

import re
from pathlib import Path


def test_package_imports() -> None:
    """The top-level package must import without side-effects."""
    import mamp_ml  # noqa: F401  -- import is the assertion


def test_package_advertises_version() -> None:
    """The package must expose a PEP 440-compatible version string."""
    import mamp_ml

    assert isinstance(mamp_ml.__version__, str)
    assert re.match(r"^\d+\.\d+", mamp_ml.__version__), (
        f"unexpected version string: {mamp_ml.__version__!r}"
    )


def test_example_xlsx_exists(example_xlsx: Path) -> None:
    """The canonical example input must be reachable from tests."""
    assert example_xlsx.exists(), f"missing example data file: {example_xlsx}"
    assert example_xlsx.suffix == ".xlsx"


def test_lrr_annotation_cache_present(repo_root: Path) -> None:
    """The pre-computed LRR-Annotation cache backs the post-fold golden tests."""
    cache = repo_root / "LRR_Annotation" / "cache"
    expected = {
        "structures.pickle",
        "breakpoints.pickle",
        "windings.pickle",
    }
    present = {p.name for p in cache.glob("*.pickle")}
    missing = expected - present
    assert not missing, f"missing cache files needed for golden tests: {sorted(missing)}"

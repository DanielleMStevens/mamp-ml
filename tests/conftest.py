"""Shared pytest fixtures for the mamp-ml test suite.

Real fixtures (cached structures, golden outputs from the R+shell pipeline,
example_data.xlsx slices) are introduced alongside the rewrites that they
validate. This module is currently a placeholder so pytest discovers the
package as soon as scaffolding is in place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Path to the repository root, used by fixtures to resolve example data,
# cached LRR-Annotation pickles, and other on-disk assets.
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root checkout."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to the test-fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def example_xlsx(repo_root: Path) -> Path:
    """Path to the canonical example input spreadsheet used in golden tests."""
    return repo_root / "example_data.xlsx"

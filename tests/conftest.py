"""Shared pytest fixtures for the mamp-ml test suite.

Provides the path-resolving fixtures (repo root, test-fixture directory,
canonical example spreadsheet) and the optional ColabFold-output fixture
used by the checkpoint-4 golden tests. Anything that requires running
ColabFold first will skip cleanly with a clear "how to generate" message
on machines where the fixtures have not yet been produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Path to the repository root, used by fixtures to resolve example data,
# cached LRR-Annotation pickles, and other on-disk assets.
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolate_hf_cache(tmp_path_factory, monkeypatch):
    """Keep model-cache configuration from touching the source tree.

    ``mamp-ml``'s CLI steers the HuggingFace cache to a ``model_cache/`` folder
    next to the install when neither ``--cache-dir`` nor ``HF_HOME`` is set
    (see ``_configure_model_cache``). Under test that would create a directory
    inside ``src/mamp_ml/``. Pre-setting ``HF_HOME`` to a throwaway temp dir
    makes that logic a respected-user-environment no-op, so the suite never
    pollutes the package. Tests that exercise the default branch explicitly
    delete these vars themselves.
    """
    monkeypatch.setenv("HF_HOME", str(tmp_path_factory.mktemp("hf_home")))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)


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


# ---------------------------------------------------------------------------
# ColabFold-derived fixtures (checkpoint 4+)
# ---------------------------------------------------------------------------
# The ColabFold output directory is *generated* by
# tests/fixtures/build_colabfold_fixtures.sh rather than committed to git
# (see the .gitignore at the repo root for the reasoning).
#
# Tests that need it should declare a dependency on the
# ``colabfold_outputs_dir`` fixture below. When the directory is missing,
# pytest skips those tests with an actionable message instead of failing,
# so the suite stays runnable on machines without ColabFold installed.

# Files we expect ColabFold to produce for the example data. The presence of
# log.txt is the single best "did it finish" sentinel.
_COLABFOLD_SENTINEL_FILES = ("log.txt",)


@pytest.fixture(scope="session")
def colabfold_outputs_dir(fixtures_dir: Path) -> Path:
    """Path to the ColabFold-generated PDB/log artefacts; skips if absent.

    The directory and its contents are produced by
    ``tests/fixtures/build_colabfold_fixtures.sh`` from
    ``example_data.xlsx``. Tests that read from this fixture will be
    silently skipped on environments where the fixtures have not yet been
    built, keeping the test suite runnable on lightweight CI runners.
    """
    candidate = fixtures_dir / "colabfold_output"
    missing = [
        name
        for name in _COLABFOLD_SENTINEL_FILES
        if not (candidate / name).exists()
    ]
    if not candidate.is_dir() or missing:
        pytest.skip(
            "ColabFold fixtures not present "
            f"(missing: {missing or [str(candidate)]}). "
            "Run `bash tests/fixtures/build_colabfold_fixtures.sh` once "
            "in an env where colabfold_batch is available to enable "
            "these tests."
        )
    return candidate

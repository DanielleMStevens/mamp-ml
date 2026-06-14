"""mamp-ml: deep-learning predictor for plant receptor-ligand immunogenicity.

This package provides a single, installable interface to the MAMP-ml pipeline:
data preprocessing, structure-based feature extraction, and ESM-2 backbone
prediction of receptor-ligand interaction class.

Public API surface is intentionally minimal during scaffolding (checkpoint #1);
real preprocessing and prediction entry points are introduced in subsequent
checkpoints. See README.md for the current usage instructions.
"""

from __future__ import annotations

__all__ = ["__version__"]

# Version is kept in lock-step with pyproject.toml's `[project] version`.
# Single source-of-truth via importlib.metadata would require the package to
# be installed; we keep a fallback string for editable / source-tree use.
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        __version__: str = _pkg_version("mamp-ml")
    except PackageNotFoundError:  # pragma: no cover - source tree, not installed
        __version__ = "0.2.0"
except ImportError:  # pragma: no cover - Python < 3.8 (unsupported)
    __version__ = "0.2.0"

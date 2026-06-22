"""mamp-ml: deep-learning predictor for plant receptor-ligand immunogenicity.

This package provides a single, installable interface to the MAMP-ml pipeline:
data preprocessing, structure-based feature extraction, and ESM-2 backbone
prediction of receptor-ligand interaction class.

Public API surface is intentionally minimal during scaffolding (checkpoint #1);
real preprocessing and prediction entry points are introduced in subsequent
checkpoints. See README.md for the current usage instructions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["__version__", "example_data_path"]


def example_data_path() -> "Path":
    """Return the filesystem path to the bundled ``example_data.xlsx`` sample.

    This is the same spreadsheet referenced throughout the README; it ships
    inside the installed package so a ``pip install mamp-ml`` user can smoke-
    test their install without cloning the repo::

        mamp-ml predict --example

    Returns
    -------
    pathlib.Path
        Absolute path to the bundled ``example_data.xlsx``.

    Raises
    ------
    FileNotFoundError
        If the file is missing from the install (a packaging error).
    """
    from importlib import resources
    from pathlib import Path

    # `resources.files` (Python 3.9+) returns a Traversable rooted at the
    # installed `mamp_ml.examples` package; for a regular wheel / editable
    # install this resolves to a real on-disk path.
    resource = resources.files("mamp_ml.examples") / "example_data.xlsx"
    path = Path(str(resource))
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled example_data.xlsx not found at {path}; this indicates a "
            "broken install. Reinstall mamp-ml, or pass an explicit .xlsx path."
        )
    return path

# Version is kept in lock-step with pyproject.toml's `[project] version`.
# Single source-of-truth via importlib.metadata would require the package to
# be installed; we keep a fallback string for editable / source-tree use.
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        __version__: str = _pkg_version("mamp-ml")
    except PackageNotFoundError:  # pragma: no cover - source tree, not installed
        __version__ = "0.2.8"
except ImportError:  # pragma: no cover - Python < 3.8 (unsupported)
    __version__ = "0.2.0"

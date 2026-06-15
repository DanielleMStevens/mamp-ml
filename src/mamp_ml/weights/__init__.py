"""Bundled model-weight resolution for mamp_ml.

The 33 MB pretrained checkpoint ``mamp_ml_weights.pth`` ships with the
package so users don't need to download it separately. This module exposes
:func:`default_weights_path` which returns the absolute path to the
checkpoint, working transparently for both an editable source install
(``pip install -e .``) and a regular wheel install (``pip install mamp-ml``).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["default_weights_path", "BUNDLED_WEIGHTS_FILENAME"]

#: Filename of the checkpoint shipped in this directory.
BUNDLED_WEIGHTS_FILENAME = "mamp_ml_weights.pth"


def default_weights_path() -> Path:
    """Return the absolute path to the bundled MAMP-ml checkpoint.

    Because the package is installed as a regular (unpacked) Python
    package — not from a zipped wheel — the bundled file is always
    reachable as a plain on-disk path. We resolve it relative to this
    ``__init__.py`` so the lookup works from any working directory.

    Returns
    -------
    pathlib.Path
        Absolute path to ``mamp_ml_weights.pth``. The file may or may
        not exist on disk (it should, but the caller is responsible for
        the actual existence check before passing the path to ``torch.load``).
    """
    return Path(__file__).resolve().parent / BUNDLED_WEIGHTS_FILENAME

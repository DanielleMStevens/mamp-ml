"""Bundled sample data shipped inside the installed package.

This subpackage exists so that a ``pip install mamp-ml`` user can smoke-test
their install without first cloning the repository or supplying real data:

    mamp-ml predict --example --device cuda

The sample spreadsheet (:file:`example_data.xlsx`) is the same file kept at the
repository root; :func:`mamp_ml.example_data_path` resolves the installed copy
via :mod:`importlib.resources`, and a test asserts the two stay byte-identical.
"""

from __future__ import annotations

__all__: list[str] = []

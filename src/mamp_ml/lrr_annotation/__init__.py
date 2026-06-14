"""LRR-Annotation: geometric analysis of leucine-rich-repeat receptor structures.

This package contains the legacy ``geom_lrr`` analyzer in a (now nested) layout
under ``mamp_ml.lrr_annotation``. The convenience re-exports below preserve the
public symbol surface that ``from mamp_ml.lrr_annotation import ...`` is
expected to expose, while pointing at the modules that actually define them
(inside the nested ``geom_lrr`` package).

Note on history: the original top-level ``LRR_Annotation/__init__.py`` shipped
``from .loader import Loader`` etc., but those sibling modules never existed at
that level — the real implementations always lived inside ``geom_lrr/``. The
package was therefore only usable via the explicit ``from geom_lrr import ...``
import path that ``scripts/02_alphafold_to_lrr_annotation.py`` used. We make
those re-exports actually resolve here so the dotted public API works as
documented; the underlying behaviour is unchanged.
"""

from __future__ import annotations

from .geom_lrr.analyzer import (
    Analyzer,
    compute_laplacian_circular_coords,
    compute_lrr_discrepancy,
    compute_lrr_std,
    compute_lrr_winding_laplacian,
    compute_regression,
    compute_winding,
    median_slope,
)
from .geom_lrr.loader import Loader
from .geom_lrr.plotter import (
    Plotter,
    plot_regression,
    plot_residue_annotations_3d,
)

__all__ = [
    "Analyzer",
    "Loader",
    "Plotter",
    "compute_laplacian_circular_coords",
    "compute_lrr_discrepancy",
    "compute_lrr_std",
    "compute_lrr_winding_laplacian",
    "compute_regression",
    "compute_winding",
    "median_slope",
    "plot_regression",
    "plot_residue_annotations_3d",
]

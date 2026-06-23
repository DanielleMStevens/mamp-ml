"""Validate that every public symbol moved in checkpoint #2 resolves at its
new location inside the `mamp_ml` package.

These tests are intentionally pure-import checks. They confirm:

1.  the `src/` layout is discoverable on `sys.path`,
2.  every cross-module reference inside the package was rewritten correctly
    (no leftover `from models...`, `from losses...`, `import misc`),
3.  the legacy LRR-Annotation subpackage works under its new dotted path,
4.  the dataset, loss, model, training-engine, and metric utilities can be
    imported independently of each other.

The tests skip cleanly when a heavy optional dependency (torch, transformers,
fair-esm, wandb) is not installed, so they are runnable on light CI runners
without a GPU build of PyTorch.
"""

from __future__ import annotations

import importlib

import pytest


def _try_import(module_name: str):
    """Import a module, skipping the test if a heavy optional dep is missing.

    We do not want a missing `torch` / `transformers` install to be reported as
    a *failure* of the move; only as a skipped check. Anything else (e.g. a
    NameError due to a botched rewrite) is a genuine bug and must surface.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        skipped_deps = {
            "torch",
            "transformers",
            "esm",
            "wandb",
            "matplotlib",
            "sklearn",
            "scipy",
            "Bio",
            "pandas",
            "openpyxl",
            "tqdm",
        }
        if exc.name in skipped_deps:
            pytest.skip(f"optional runtime dep '{exc.name}' not installed")
        raise


def test_lrr_annotation_subpackage() -> None:
    """`mamp_ml.lrr_annotation` re-exports the geometric LRR API."""
    lrr = _try_import("mamp_ml.lrr_annotation")
    for name in ("Loader", "Analyzer", "Plotter", "compute_winding"):
        assert hasattr(lrr, name), f"mamp_ml.lrr_annotation is missing {name}"


def test_lrr_annotation_geom_lrr_submodule() -> None:
    """The nested `geom_lrr` package resolves under the new dotted path."""
    mod = _try_import("mamp_ml.lrr_annotation.geom_lrr")
    for name in ("Loader", "Analyzer", "compute_winding"):
        assert hasattr(mod, name), f"mamp_ml.lrr_annotation.geom_lrr is missing {name}"


def test_lrr_features_module() -> None:
    """`mamp_ml.lrr_features` (B-factor bandpass) is importable.

    Replaces the obsolete ``analyze_bfactor_peaks`` smoke test — that
    module was deleted in checkpoint 7 in favour of the fresh
    :mod:`mamp_ml.lrr_features` implementation.
    """
    mod = _try_import("mamp_ml.lrr_features")
    assert hasattr(mod, "compute_bfactor_lrr_segments")
    assert hasattr(mod, "write_bfactor_lrr_segments")


def test_losses_subpackage() -> None:
    """Loss classes resolve from `mamp_ml.losses.*`."""
    ce = _try_import("mamp_ml.losses.cross_entropy")
    sc = _try_import("mamp_ml.losses.supcon")
    assert hasattr(ce, "CrossEntropyLoss")
    assert hasattr(sc, "SupConLoss")


def test_datasets_subpackage() -> None:
    """The dataset module imports cleanly from the new location."""
    mod = _try_import("mamp_ml.datasets.seq_with_receptor_dataset")
    assert mod is not None


def test_models_subpackage() -> None:
    """The ESM2 model module imports cleanly from the new location."""
    mod = _try_import("mamp_ml.models.esm_positon_weighted")
    for name in (
        "BFactorWeightGenerator",
        "ESMBfactorWeightedFeatures",
        "PeptideSeqWithReceptorDataset",
    ):
        assert hasattr(mod, name), f"mamp_ml.models.esm_positon_weighted missing {name}"


def test_engine_train_module() -> None:
    """The training/eval engine resolves from `mamp_ml.engine_train`."""
    mod = _try_import("mamp_ml.engine_train")
    for name in ("train_one_epoch", "evaluate"):
        assert hasattr(mod, name), f"mamp_ml.engine_train missing {name}"


def test_misc_module() -> None:
    """`mamp_ml.misc` resolves and exposes the original public helpers."""
    mod = _try_import("mamp_ml.misc")
    # Smoke-check a couple of well-known helpers; full API surface is unchanged.
    assert hasattr(mod, "init_distributed_mode") or hasattr(mod, "get_rank")


def test_train_entry_module() -> None:
    """The CLI training entry point resolves from `mamp_ml.train`."""
    mod = _try_import("mamp_ml.train")
    # `get_args_parser` and `main` are the canonical public entry points.
    assert hasattr(mod, "get_args_parser")
    assert hasattr(mod, "main")

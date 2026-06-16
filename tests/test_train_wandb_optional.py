"""Tests that wandb is treated as an optional, training-only dependency.

Regression: `mamp-ml predict` crashed with ``ModuleNotFoundError: No module
named 'wandb'`` because ``mamp_ml/train.py`` imported wandb at module top level,
even though inference always runs with ``--disable_wandb`` and never touches it.
The import is now lazy, and a requested-but-missing wandb degrades to disabled
logging with a warning instead of crashing.
"""

from __future__ import annotations

import importlib
import types

import pytest


def _import_train():
    """Import mamp_ml.train, skipping only if a *heavy* dep (not wandb) is absent.

    The whole point of the fix is that a missing ``wandb`` must NOT prevent the
    import, so wandb is deliberately excluded from the skip set.
    """
    try:
        return importlib.import_module("mamp_ml.train")
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "transformers", "esm", "sklearn", "scipy", "matplotlib"}:
            pytest.skip(f"optional heavy dep '{exc.name}' not installed")
        raise


def test_train_imports_without_wandb() -> None:
    """`from mamp_ml import train` must succeed regardless of wandb being present.

    This is the exact import that crashed `mamp-ml predict` in the field.
    """
    train = _import_train()
    # The module-level symbol exists; it is either the real module or None,
    # never an unbound name that would NameError when referenced.
    assert hasattr(train, "wandb")
    assert hasattr(train, "main")


def test_disable_wandb_if_unavailable_flips_and_warns(monkeypatch) -> None:
    """wandb requested (disable_wandb=False) but absent -> disabled + warning."""
    train = _import_train()
    monkeypatch.setattr(train, "wandb", None)

    args = types.SimpleNamespace(disable_wandb=False)
    with pytest.warns(UserWarning, match="wandb is not installed"):
        returned = train._disable_wandb_if_unavailable(args)
    assert args.disable_wandb is True
    assert returned is args  # returns the same namespace for convenience


def test_disable_wandb_if_unavailable_noop_when_already_disabled(
    monkeypatch, recwarn
) -> None:
    """Already-disabled stays disabled and emits no warning even if wandb is absent."""
    train = _import_train()
    monkeypatch.setattr(train, "wandb", None)

    args = types.SimpleNamespace(disable_wandb=True)
    train._disable_wandb_if_unavailable(args)
    assert args.disable_wandb is True
    assert len(recwarn) == 0


def test_disable_wandb_if_unavailable_noop_when_wandb_present(
    monkeypatch, recwarn
) -> None:
    """When wandb IS importable, an enabled run is left enabled (no warning)."""
    train = _import_train()
    # Stand-in for an installed wandb module.
    monkeypatch.setattr(train, "wandb", types.ModuleType("wandb"))

    args = types.SimpleNamespace(disable_wandb=False)
    train._disable_wandb_if_unavailable(args)
    assert args.disable_wandb is False
    assert len(recwarn) == 0

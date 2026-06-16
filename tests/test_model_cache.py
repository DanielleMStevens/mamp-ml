"""Tests for HuggingFace model-cache steering (_configure_model_cache).

By default mamp-ml caches model weights next to its install (so they land on
whatever filesystem you installed onto, not a small-quota HOME), while still
respecting an explicit `--cache-dir` or `$HF_HOME`.
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import mamp_ml
from mamp_ml.__main__ import (
    _build_parser,
    _configure_model_cache,
    _default_model_cache_dir,
)


def test_default_cache_dir_is_next_to_install() -> None:
    """The default cache sits beside the installed package, not in HOME."""
    default = _default_model_cache_dir()
    pkg_dir = Path(mamp_ml.__file__).resolve().parent
    assert default == pkg_dir / "model_cache"


def test_configure_uses_explicit_cache_dir(tmp_path, monkeypatch) -> None:
    """--cache-dir wins and the directory is created + exported as HF_HOME."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    target = tmp_path / "my_cache"
    args = types.SimpleNamespace(cache_dir=str(target))

    _configure_model_cache(args)

    assert target.is_dir()
    assert os.environ["HF_HOME"] == str(target)


def test_configure_respects_existing_hf_home(tmp_path, monkeypatch) -> None:
    """An HF_HOME the user already set is never overridden, and no default
    cache dir is created."""
    preset = tmp_path / "user_hf"
    monkeypatch.setenv("HF_HOME", str(preset))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    args = types.SimpleNamespace(cache_dir=None)

    _configure_model_cache(args)

    # The user's HF_HOME is left exactly as they set it (not overridden, and no
    # install-adjacent default substituted in).
    assert os.environ["HF_HOME"] == str(preset)


def test_configure_defaults_next_to_install_when_unset(tmp_path, monkeypatch) -> None:
    """With no --cache-dir and no HF env, fall back to the install-adjacent
    default. We redirect the 'default' to tmp so the test never writes into the
    real package directory."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    fake_default = tmp_path / "install_cache"
    monkeypatch.setattr(
        "mamp_ml.__main__._default_model_cache_dir", lambda: fake_default
    )
    args = types.SimpleNamespace(cache_dir=None)

    _configure_model_cache(args)

    assert fake_default.is_dir()
    assert os.environ["HF_HOME"] == str(fake_default)


def test_configure_is_noop_when_dir_uncreatable(tmp_path, monkeypatch) -> None:
    """If the cache dir can't be created, HF_HOME is left untouched (HF keeps
    its own default) rather than crashing."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    # A path *under a regular file* can't be mkdir'd -> OSError, swallowed.
    blocker = tmp_path / "iamafile"
    blocker.write_text("x")
    args = types.SimpleNamespace(cache_dir=str(blocker / "nope"))

    _configure_model_cache(args)

    assert "HF_HOME" not in os.environ


def test_predict_subparser_accepts_cache_dir(example_xlsx) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(["predict", str(example_xlsx), "--cache-dir", "/tmp/x"])
    assert parsed.cache_dir == "/tmp/x"
    # Default is None (runtime then picks the install-adjacent dir).
    assert parser.parse_args(["predict", str(example_xlsx)]).cache_dir is None


def test_prepare_and_fold_subparsers_accept_cache_dir(example_xlsx, tmp_path) -> None:
    parser = _build_parser()
    p = parser.parse_args(["prepare", str(example_xlsx), "--cache-dir", "/tmp/a"])
    assert p.cache_dir == "/tmp/a"
    f = parser.parse_args(
        ["fold", str(example_xlsx), str(tmp_path / "out"), "--cache-dir", "/tmp/b"]
    )
    assert f.cache_dir == "/tmp/b"

"""Unique per-run working/output directories + structure reuse.

The default working directory is now ``intermediate_files/<input>_<disc>_<rand>``
(unique per run) so concurrent runs — e.g. SLURM array tasks launched from the
same directory — never share a working dir or clobber each other. The output
folder shares the same run token. Previously-folded structures can be reused via
``--structures`` to skip the structural-modeling stage.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

from mamp_ml.__main__ import (
    _make_run_token,
    _resolve_working_dir,
    _run_discriminator,
    _sanitize_token_part,
)


# ---------------------------------------------------------------------------
# token building
# ---------------------------------------------------------------------------


def test_sanitize_token_part_strips_unsafe_chars_and_caps_length() -> None:
    assert _sanitize_token_part("Cabernet filtered/RLKs!") == "Cabernet_filtered_RLKs"
    assert _sanitize_token_part("") == "run"
    assert _sanitize_token_part("/// ") == "run"
    assert len(_sanitize_token_part("x" * 200)) == 40


def test_make_run_token_includes_stem_and_random_suffix() -> None:
    args = types.SimpleNamespace(xlsx="/data/Cabernet_filtered_RLKs.xlsx")
    token = _make_run_token(args)
    assert token.startswith("Cabernet_filtered_RLKs_")
    assert re.search(r"_[0-9a-f]{4}$", token)


def test_make_run_token_is_unique_across_calls() -> None:
    args = types.SimpleNamespace(xlsx="x.xlsx")
    tokens = {_make_run_token(args) for _ in range(50)}
    assert len(tokens) == 50  # random suffix guarantees uniqueness


def test_run_discriminator_prefers_slurm(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "4471823")
    assert _run_discriminator() == "job4471823"

    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "999")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    assert _run_discriminator() == "job999_7"


def test_run_discriminator_falls_back_to_timestamp_off_cluster(monkeypatch) -> None:
    for var in (
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", _run_discriminator())


# ---------------------------------------------------------------------------
# working-dir resolution
# ---------------------------------------------------------------------------


def test_resolve_working_dir_is_unique_and_created(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    a = types.SimpleNamespace(xlsx="data.xlsx", out_dir=None)
    b = types.SimpleNamespace(xlsx="data.xlsx", out_dir=None)
    da = _resolve_working_dir(a)
    db = _resolve_working_dir(b)
    assert da.is_dir() and db.is_dir()
    assert da != db  # two concurrent runs -> two distinct dirs
    assert da.parent.name == "intermediate_files"
    # The resolver records the run token + chosen dir on args.
    assert a._run_token and a.out_dir == str(da)


def test_resolve_working_dir_honours_explicit_out_dir(tmp_path) -> None:
    explicit = tmp_path / "my_scratch"
    args = types.SimpleNamespace(xlsx="data.xlsx", out_dir=str(explicit))
    resolved = _resolve_working_dir(args)
    assert resolved == explicit
    assert explicit.is_dir()
    # No run token minted when the user pinned the dir.
    assert not getattr(args, "_run_token", None)


def test_resolve_working_dir_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = types.SimpleNamespace(xlsx="data.xlsx", out_dir=None)
    first = _resolve_working_dir(args)
    second = _resolve_working_dir(args)  # out_dir now set -> same dir back
    assert first == second

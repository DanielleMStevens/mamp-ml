"""Tests for the bundled ``example_data.xlsx`` sample + ``predict --example``.

A ``pip install mamp-ml`` user has no repository checkout, so the canonical
sample spreadsheet must ship *inside* the package (under ``mamp_ml/examples/``)
and be reachable both programmatically (:func:`mamp_ml.example_data_path`) and
from the CLI (``mamp-ml predict --example``). These tests guard that contract,
including that the packaged copy never silently drifts from the repo-root copy
the rest of the test-suite treats as golden.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mamp_ml
from mamp_ml.__main__ import _build_parser


# ---------------------------------------------------------------------------
# example_data_path() accessor
# ---------------------------------------------------------------------------


def test_example_data_path_returns_existing_xlsx() -> None:
    """The accessor resolves to a real, package-internal .xlsx file."""
    path = mamp_ml.example_data_path()
    assert isinstance(path, Path)
    assert path.is_file()
    assert path.name == "example_data.xlsx"
    # It must live under the installed package, not the repo root, so it works
    # for a pip user with no checkout.
    assert path.parent.name == "examples"
    assert path.parent.parent.name == "mamp_ml"


def test_example_data_path_is_exported() -> None:
    """`example_data_path` is part of the package's public API surface."""
    assert "example_data_path" in mamp_ml.__all__


def test_bundled_example_matches_repo_root_copy(repo_root: Path) -> None:
    """The packaged copy must stay byte-identical to the repo-root golden input
    so the two never diverge (the rest of the suite reads the repo-root copy)."""
    bundled = mamp_ml.example_data_path()
    repo_copy = repo_root / "example_data.xlsx"
    assert repo_copy.is_file(), "repo-root example_data.xlsx disappeared"
    assert bundled.read_bytes() == repo_copy.read_bytes(), (
        "bundled mamp_ml/examples/example_data.xlsx has drifted from the "
        "repo-root example_data.xlsx; re-sync them."
    )


# ---------------------------------------------------------------------------
# predict --example CLI wiring
# ---------------------------------------------------------------------------


def test_predict_xlsx_is_optional_with_example_flag() -> None:
    """`predict --example` parses with no positional; a bare path still works."""
    parser = _build_parser()

    parsed = parser.parse_args(["predict", "--example"])
    assert parsed.example is True
    assert parsed.xlsx is None

    parsed_path = parser.parse_args(["predict", "data.xlsx"])
    assert parsed_path.example is False
    assert parsed_path.xlsx == "data.xlsx"

    parsed_default = parser.parse_args(["predict", "data.xlsx"])
    assert parsed_default.example is False


def test_run_predict_example_flag_uses_bundled_dataset(monkeypatch, capsys) -> None:
    """`predict --example` swaps the bundled sample into args.xlsx before the
    pipeline runs. We stub _run_prepare to capture the resolved path (and stop
    before the heavy folding/inference stages)."""
    import mamp_ml.__main__ as cli

    captured: dict = {}

    def fake_prepare(args) -> int:
        captured["xlsx"] = args.xlsx
        return 2  # gate-style early return so predict stops here

    monkeypatch.setattr(cli, "_run_prepare", fake_prepare)

    rc = cli.main(["predict", "--example"])
    assert rc == 2
    assert captured["xlsx"] == str(mamp_ml.example_data_path())
    assert "Using bundled example dataset" in capsys.readouterr().out


def test_run_predict_without_xlsx_or_example_errors(monkeypatch, capsys) -> None:
    """A bare `predict` (no path, no --example) is a clean usage error (rc 2)
    that never reaches the pipeline."""
    import mamp_ml.__main__ as cli

    def fail_prepare(args) -> int:  # pragma: no cover - must not be called
        raise AssertionError("_run_prepare should not run without an input")

    monkeypatch.setattr(cli, "_run_prepare", fail_prepare)

    rc = cli.main(["predict"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "no input spreadsheet given" in out
    assert "--example" in out


# ---------------------------------------------------------------------------
# `mamp-ml example` subcommand
# ---------------------------------------------------------------------------


def test_cli_example_copies_sample_into_out_dir(tmp_path: Path, capsys) -> None:
    """`mamp-ml example --out-dir DIR` writes a byte-identical copy of the
    bundled sample and returns 0."""
    from mamp_ml.__main__ import main as cli_main

    rc = cli_main(["example", "--out-dir", str(tmp_path)])
    assert rc == 0
    written = tmp_path / "example_data.xlsx"
    assert written.is_file()
    assert written.read_bytes() == mamp_ml.example_data_path().read_bytes()
    out = capsys.readouterr().out
    assert "Wrote example_data.xlsx" in out
    assert "mamp-ml predict" in out


def test_cli_example_path_prints_bundled_location_without_copying(
    tmp_path: Path, capsys
) -> None:
    """`--path` prints the bundled file's location and copies nothing."""
    from mamp_ml.__main__ import main as cli_main

    rc = cli_main(["example", "--path", "--out-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == str(mamp_ml.example_data_path())
    # Nothing was written into the target dir.
    assert not (tmp_path / "example_data.xlsx").exists()


def test_cli_example_refuses_to_clobber_without_force(
    tmp_path: Path, capsys
) -> None:
    """An existing target is preserved unless --force is given (rc 1)."""
    from mamp_ml.__main__ import main as cli_main

    target = tmp_path / "example_data.xlsx"
    target.write_text("do not overwrite me")

    rc = cli_main(["example", "--out-dir", str(tmp_path)])
    assert rc == 1
    assert target.read_text() == "do not overwrite me"
    assert "already exists" in capsys.readouterr().out


def test_cli_example_force_overwrites(tmp_path: Path) -> None:
    """--force replaces an existing target with the bundled sample."""
    from mamp_ml.__main__ import main as cli_main

    target = tmp_path / "example_data.xlsx"
    target.write_text("stale")

    rc = cli_main(["example", "--out-dir", str(tmp_path), "--force"])
    assert rc == 0
    assert target.read_bytes() == mamp_ml.example_data_path().read_bytes()


def test_cli_example_reports_missing_bundled_file(monkeypatch, capsys) -> None:
    """If the bundled sample is missing (broken install), the command fails
    cleanly with rc 3 rather than a traceback."""
    import mamp_ml
    import mamp_ml.__main__ as cli

    def boom() -> Path:
        raise FileNotFoundError("Bundled example_data.xlsx not found")

    monkeypatch.setattr(mamp_ml, "example_data_path", boom)

    rc = cli.main(["example"])
    assert rc == 3
    assert "Error:" in capsys.readouterr().out

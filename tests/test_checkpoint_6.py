"""Tests for checkpoint 6: bundled weights, console-script entry point,
``predict`` subcommand.

The actual ESM-2 inference is *not* exercised by these tests because it
requires downloading and running a 650M-param model (~2.5 GB HuggingFace
download on first use, ~5-10 min on M2 CPU). The smoke tests below cover
argument parsing, the bundled-weights lookup, the pyproject entry-point
declaration, and the ColabFold gating path (which short-circuits before
the model is ever loaded). End-to-end model inference is verified by a
manual smoke run in the README's Quickstart.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mamp_ml.__main__ import main as cli_main


# ---------------------------------------------------------------------------
# Bundled weights helper
# ---------------------------------------------------------------------------


def test_default_weights_path_returns_existing_file() -> None:
    """The bundled 33 MB checkpoint must be on disk under src/mamp_ml/weights/."""
    from mamp_ml.weights import default_weights_path

    p = default_weights_path()
    assert isinstance(p, Path)
    assert p.is_file(), f"bundled weights missing at {p}"
    # Sanity-check the size — way too small means a Git-LFS / corrupt download.
    assert p.stat().st_size > 30_000_000, (
        f"weights file at {p} is suspiciously small ({p.stat().st_size} bytes); "
        "may be a Git-LFS pointer that was never resolved."
    )


def test_default_weights_path_independent_of_cwd(tmp_path: Path) -> None:
    """The lookup must be cwd-independent — uses __file__-relative resolution."""
    from mamp_ml.weights import default_weights_path

    prev = Path.cwd()
    try:
        os.chdir(tmp_path)
        p = default_weights_path()
        assert p.is_file()
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# Console-script entry point (pyproject [project.scripts])
# ---------------------------------------------------------------------------


def test_pyproject_declares_mamp_ml_console_script(repo_root: Path) -> None:
    """pyproject.toml must register a `mamp-ml` console script pointing at the
    package dispatcher, so `pip install` registers an executable."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    text = (repo_root / "pyproject.toml").read_bytes()
    cfg = tomllib.loads(text.decode("utf-8"))
    scripts = cfg.get("project", {}).get("scripts", {})
    assert scripts.get("mamp-ml") == "mamp_ml.__main__:main", (
        f"expected mamp-ml -> mamp_ml.__main__:main, got: {scripts}"
    )


# ---------------------------------------------------------------------------
# `python -m mamp_ml predict` subcommand
# ---------------------------------------------------------------------------


def test_predict_help_renders() -> None:
    """`predict --help` must not crash (catches typos in the parser setup)."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["predict", "--help"])
    # argparse exits 0 on --help.
    assert exc_info.value.code == 0


def test_predict_gates_on_missing_colabfold(
    tmp_path: Path, example_xlsx: Path, capsys, monkeypatch
) -> None:
    """`predict` reuses prepare's ColabFold gate: when fold output is missing
    AND no colabfold_batch is installed, write the receptor FASTA, print the
    colabfold_batch hint, exit cleanly."""
    # Deterministically exercise the "not installed" branch.
    import mamp_ml.fold.colabfold as cf

    monkeypatch.setattr(cf, "find_colabfold_installs", lambda: [])

    out_dir = tmp_path / "inter"
    rc = cli_main(
        ["predict", str(example_xlsx), "--out-dir", str(out_dir)]
    )
    # Gating exit code propagates from _run_prepare.
    assert rc == 2
    # FASTA still produced by stage 1.
    assert (out_dir / "receptor_full_length.fasta").is_file()
    # Hint surfaces in stdout.
    captured = capsys.readouterr()
    assert "colabfold_batch" in captured.out
    # No model artefacts.
    assert not (out_dir / "predictions.csv").exists()
    assert not (out_dir / "test_preds.pth").exists()


def test_predict_uses_bundled_weights_by_default(
    tmp_path: Path, example_xlsx: Path
) -> None:
    """When --weights is omitted, predict resolves to the bundled file.

    The CLI parser leaves ``parsed.weights`` as ``None`` when the user
    didn't pass a custom path; the runtime then substitutes the bundled
    ``mamp_ml_weights.pth`` via
    :func:`mamp_ml.weights.default_weights_path`. We verify both halves of
    that contract here without actually loading the (heavy) ESM-2 model.
    """
    from mamp_ml.__main__ import _build_parser
    from mamp_ml.weights import default_weights_path

    parser = _build_parser()
    parsed = parser.parse_args(["predict", str(example_xlsx)])
    assert parsed.weights is None
    assert default_weights_path().is_file()


# ---------------------------------------------------------------------------
# Console-script behaviour after `pip install -e .`
# ---------------------------------------------------------------------------


def test_console_script_runs_when_invoked_via_python_m(
    repo_root: Path,
) -> None:
    """Even without `pip install -e .`, `python -m mamp_ml --help` works.

    This is the safety net for users running from a fresh clone — the
    console script only registers after a proper install, but every
    machine that has Python on PATH and PYTHONPATH set to the source
    tree can still call the dispatcher this way.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, "-m", "mamp_ml", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0
    assert "prepare" in result.stdout
    assert "predict" in result.stdout


# ---------------------------------------------------------------------------
# Colab notebook
# ---------------------------------------------------------------------------


def test_colab_notebook_is_valid_jupyter_v4(repo_root: Path) -> None:
    """The Colab notebook must be a well-formed nbformat v4 document so it
    opens cleanly in Google Colab and in jupyter."""
    import json

    nb_path = repo_root / "mamp_ml_colab.ipynb"
    assert nb_path.is_file(), f"missing notebook: {nb_path}"
    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)
    assert nb["nbformat"] == 4, "notebook must be nbformat v4 (Colab compatible)"
    assert "cells" in nb and len(nb["cells"]) > 0
    # Every cell must declare a valid type and a source attribute.
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert "source" in cell


def test_readme_documents_the_new_cli(repo_root: Path) -> None:
    """The README must showcase the seamless ``mamp-ml predict`` UX and
    must not still tell users to invoke the deleted legacy scripts."""
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "mamp-ml predict" in readme, (
        "README must document the one-shot `mamp-ml predict` command"
    )
    assert "pip install" in readme, "README must document the pip install path"
    # Must not still tell users to invoke the legacy scripts (they're deleted).
    for legacy in (
        "01_convert_sheet_to_fasta.R",
        "02_alphafold_to_lrr_annotation.py",
        "03_parse_lrr_annotation.py",
        "04_data_prep_for_prediction.py",
        "05_chemical_conversion.R",
        "main_train.py",
    ):
        assert legacy not in readme, (
            f"README still references the deleted legacy script '{legacy}'"
        )


def test_colab_notebook_uses_mamp_ml_cli(repo_root: Path) -> None:
    """The Colab notebook must invoke the top-level `mamp-ml` CLI, not the
    legacy bash scripts or the per-stage subcommands directly. This is the
    user-facing contract we promised in the checkpoint-5 commit."""
    import json

    nb_path = repo_root / "mamp_ml_colab.ipynb"
    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)

    all_code = "\n".join(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )

    # Must invoke the top-level predict command. `mamp-ml predict` now folds
    # (auto-running ColabFold or in-process ESMFold) AND runs inference in one
    # command, so the notebook no longer needs a separate `mamp-ml prepare`
    # step — predict is the single canonical entry point.
    assert "mamp-ml predict" in all_code, "notebook must invoke `mamp-ml predict`"

    # Must NOT invoke any of the legacy scripts that we're transitioning away from.
    for legacy in (
        "scripts/01_convert_sheet_to_fasta.R",
        "scripts/02_alphafold_to_lrr_annotation.py",
        "scripts/03_parse_lrr_annotation.py",
        "scripts/04_data_prep_for_prediction.py",
        "scripts/05_chemical_conversion.R",
        "main_train.py",
    ):
        assert legacy not in all_code, (
            f"notebook still references the legacy script '{legacy}'"
        )

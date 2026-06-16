"""Tests for the ``python -m mamp_ml`` CLI dispatcher (checkpoint 5).

Two layers of coverage:

1. Per-subcommand smoke tests that invoke the dispatcher in-process and
   confirm each stage produces its expected output file. These run on
   every machine and exercise the argument parsing + delegation layer.

2. A full end-to-end shell test that drives the two updated shell scripts
   (``prepare_input_data.sh`` then ``run_preparation_pipeline.sh``) on a
   staging copy of the repository, then diffs all six produced
   intermediate-pipeline outputs against the committed goldens. Skipped
   when the locally folded ColabFold output dir is absent.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from mamp_ml.__main__ import main as cli_main


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Per-subcommand smoke tests (in-process)
# ---------------------------------------------------------------------------


def test_cli_prepare_fasta(tmp_path: Path, example_xlsx: Path) -> None:
    out = tmp_path / "receptor.fasta"
    rc = cli_main(["prepare-fasta", str(example_xlsx), str(out)])
    assert rc == 0
    assert out.is_file()
    assert out.read_text().count(">") == 2


def test_cli_prepare_fasta_missing_xlsx_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cli_main(
            ["prepare-fasta", str(tmp_path / "no.xlsx"), str(tmp_path / "out.fasta")]
        )


def test_cli_lrr_domain_fasta(
    tmp_path: Path, repo_root: Path, example_xlsx: Path
) -> None:
    # Stage 1: build the receptor FASTA we'll pass in.
    receptor_fasta = tmp_path / "receptor.fasta"
    cli_main(["prepare-fasta", str(example_xlsx), str(receptor_fasta)])

    out = tmp_path / "lrr_domain.fasta"
    golden_results = (
        repo_root / "tests" / "fixtures" / "golden" / "lrr_annotation_results.txt"
    )
    rc = cli_main(
        [
            "lrr-domain-fasta",
            str(golden_results),
            str(receptor_fasta),
            str(out),
        ]
    )
    assert rc == 0
    # Must SHA1-equal the committed LRR-domain FASTA golden.
    golden_fasta = (
        repo_root / "tests" / "fixtures" / "golden" / "lrr_domain_sequences.fasta"
    )
    assert _sha1(out) == _sha1(golden_fasta)


def test_cli_assemble_test_data(
    tmp_path: Path, repo_root: Path, example_xlsx: Path
) -> None:
    golden_fasta = (
        repo_root / "tests" / "fixtures" / "golden" / "lrr_domain_sequences.fasta"
    )
    out = tmp_path / "test_data.csv"
    rc = cli_main(
        [
            "assemble-test-data",
            str(example_xlsx),
            str(golden_fasta),
            str(out),
        ]
    )
    assert rc == 0
    golden_test_data = (
        repo_root / "tests" / "fixtures" / "golden" / "test_data.csv"
    )
    pd.testing.assert_frame_equal(pd.read_csv(out), pd.read_csv(golden_test_data))


def test_cli_chemical_features(tmp_path: Path, repo_root: Path) -> None:
    golden_test_data = (
        repo_root / "tests" / "fixtures" / "golden" / "test_data.csv"
    )
    out = tmp_path / "ready_test_data.csv"
    rc = cli_main(["chemical-features", str(golden_test_data), str(out)])
    assert rc == 0
    golden_ready = (
        repo_root / "tests" / "fixtures" / "golden" / "ready_test_data.csv"
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(out), pd.read_csv(golden_ready)
    )


def test_cli_bfactor(
    tmp_path: Path,
    repo_root: Path,
    colabfold_outputs_dir: Path,
) -> None:
    """Bfactor CLI subcommand must produce the same numeric DataFrame as the
    direct API call in checkpoint 4e (which already golden-tested)."""
    production_cache = repo_root / "src" / "mamp_ml" / "lrr_annotation" / "cache"
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    for src in colabfold_outputs_dir.glob(
        "*_unrelaxed_rank_001_alphafold2_ptm_model_*_seed_*.pdb"
    ):
        receptor = src.name.split("_unrelaxed_rank_")[0]
        (pdb_dir / f"{receptor}.pdb").write_bytes(src.read_bytes())

    out = tmp_path / "bfactor.csv"
    rc = cli_main(
        ["bfactor", str(pdb_dir), str(production_cache), str(out)]
    )
    assert rc == 0

    golden = (
        repo_root
        / "tests"
        / "fixtures"
        / "golden"
        / "bfactor_winding_lrr_segments.csv"
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(out).reset_index(drop=True),
        pd.read_csv(golden).reset_index(drop=True),
        check_dtype=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_cli_structure_stage(
    tmp_path: Path,
    repo_root: Path,
    colabfold_outputs_dir: Path,
) -> None:
    """Structure-stage CLI subcommand must produce both golden text outputs."""
    scores = tmp_path / "alphafold_scores.txt"
    pdb_target = tmp_path / "pdb_target"
    lrr_results = tmp_path / "lrr_annotation_results.txt"
    cache_dir = tmp_path / "cache"

    rc = cli_main(
        [
            "structure-stage",
            str(colabfold_outputs_dir),
            str(scores),
            str(pdb_target),
            str(lrr_results),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    assert rc == 0

    golden_scores = (
        repo_root / "tests" / "fixtures" / "golden" / "alphafold_scores.txt"
    )
    golden_lrr = (
        repo_root
        / "tests"
        / "fixtures"
        / "golden"
        / "lrr_annotation_results.txt"
    )
    assert _sha1(scores) == _sha1(golden_scores)
    assert _sha1(lrr_results) == _sha1(golden_lrr)


def test_cli_no_subcommand_errors() -> None:
    """Calling the dispatcher with no subcommand must fail with non-zero exit."""
    with pytest.raises(SystemExit):
        cli_main([])


# ---------------------------------------------------------------------------
# `prepare` one-shot — the user-facing seamless command
# ---------------------------------------------------------------------------


def test_cli_prepare_gates_on_missing_colabfold(
    tmp_path: Path, example_xlsx: Path, capsys, monkeypatch
) -> None:
    """When ColabFold is *not installed* and has not been run yet, ``prepare``
    must:
    - write the receptor FASTA (stage 1),
    - print the suggested colabfold_batch invocation,
    - exit cleanly with code 2 so a CI pipeline can detect "needs colabfold".
    """
    # Force the "nothing installed" branch deterministically (the host running
    # the test suite may legitimately have colabfold_batch installed).
    import mamp_ml.fold.colabfold as cf

    monkeypatch.setattr(cf, "find_colabfold_installs", lambda: [])

    out_dir = tmp_path / "inter"
    rc = cli_main(
        ["prepare", str(example_xlsx), "--out-dir", str(out_dir)]
    )
    assert rc == 2
    # FASTA exists.
    assert (out_dir / "receptor_full_length.fasta").is_file()
    # User-facing hint surfaces the colabfold_batch command.
    captured = capsys.readouterr()
    assert "colabfold_batch" in captured.out
    assert "has not been run yet" in captured.out
    # No downstream artefacts should have been created.
    assert not (out_dir / "alphafold_scores.txt").exists()


def test_cli_prepare_auto_runs_discovered_colabfold(
    tmp_path: Path,
    example_xlsx: Path,
    colabfold_outputs_dir: Path,
    capsys,
    monkeypatch,
) -> None:
    """When a colabfold_batch install IS found, ``prepare`` runs it
    automatically (no manual re-invocation) and continues the pipeline with the
    freshly produced outputs."""
    import shutil

    import mamp_ml.fold.colabfold as cf

    fake_binary = tmp_path / "localcolabfold" / "bin" / "colabfold_batch"
    monkeypatch.setattr(
        cf, "find_colabfold_installs", lambda: [(fake_binary, "on $PATH")]
    )

    captured_call: dict = {}

    def fake_run(binary, fasta_path, output_dir, *, num_models, num_recycle, **kw):
        # Simulate a successful ColabFold run by dropping the golden fixture
        # outputs into the requested directory, then report success.
        captured_call["binary"] = binary
        captured_call["fasta"] = fasta_path
        captured_call["output_dir"] = output_dir
        captured_call["num_models"] = num_models
        captured_call["num_recycle"] = num_recycle
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for f in colabfold_outputs_dir.iterdir():
            if f.is_file():
                shutil.copyfile(f, out / f.name)
        return 0

    monkeypatch.setattr(cf, "run_colabfold_batch", fake_run)

    out_dir = tmp_path / "inter"
    rc = cli_main(
        [
            "prepare",
            str(example_xlsx),
            "--out-dir",
            str(out_dir),
            "--structure-cache-dir",
            str(tmp_path / "fresh_cache"),
        ]
    )

    # Pipeline ran to completion using the auto-generated outputs.
    assert rc == 0
    assert captured_call["binary"] == fake_binary
    assert captured_call["num_models"] == 1
    assert captured_call["num_recycle"] == 1
    out = capsys.readouterr().out
    assert "running the discovered colabfold_batch automatically" in out
    assert "ColabFold finished ->" in out


def test_cli_prepare_reports_colabfold_failure(
    tmp_path: Path, example_xlsx: Path, capsys, monkeypatch
) -> None:
    """If the auto-run colabfold_batch exits non-zero, prepare surfaces the
    status, prints the manual fallback, and exits 2 without running downstream
    stages."""
    import mamp_ml.fold.colabfold as cf

    fake_binary = tmp_path / "cf" / "colabfold_batch"
    monkeypatch.setattr(
        cf, "find_colabfold_installs", lambda: [(fake_binary, "on $PATH")]
    )
    monkeypatch.setattr(
        cf, "run_colabfold_batch", lambda *a, **k: 1
    )

    out_dir = tmp_path / "inter"
    rc = cli_main(["prepare", str(example_xlsx), "--out-dir", str(out_dir)])
    assert rc == 2
    out = capsys.readouterr().out
    assert "ColabFold exited with status 1" in out
    assert "colabfold_batch --num-models 1" in out
    # Downstream artefacts must not exist after a failed fold.
    assert not (out_dir / "alphafold_scores.txt").exists()


def test_cli_prepare_one_shot_full_pipeline(
    tmp_path: Path,
    repo_root: Path,
    example_xlsx: Path,
    colabfold_outputs_dir: Path,
    capsys,
) -> None:
    """When ColabFold outputs are present, ``prepare`` runs every stage in
    one call and produces all six golden-equivalent outputs."""
    out_dir = tmp_path / "inter"
    out_dir.mkdir()
    cf_dir = out_dir / "receptor_only"
    cf_dir.mkdir()
    for f in colabfold_outputs_dir.iterdir():
        if f.is_file():
            shutil.copyfile(f, cf_dir / f.name)

    rc = cli_main(
        [
            "prepare",
            str(example_xlsx),
            "--out-dir",
            str(out_dir),
            "--structure-cache-dir",
            str(tmp_path / "fresh_cache"),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    # All six stages must have logged a banner line.
    for stage_marker in (
        "[1/6]",
        "[2/6]",
        "[3/6]",
        "[4/6]",
        "[5/6]",
        "[6/6]",
    ):
        assert stage_marker in captured.out, (
            f"prepare did not log {stage_marker} marker in stdout"
        )

    # Confirm every expected output landed.
    for name in (
        "receptor_full_length.fasta",
        "alphafold_scores.txt",
        "lrr_annotation_results.txt",
        "lrr_domain_sequences.fasta",
        "bfactor_winding_lrr_segments.csv",
        "test_data.csv",
        "ready_test_data.csv",
    ):
        assert (out_dir / name).is_file(), f"prepare did not produce {name}"

    # Spot-check goldens match.
    goldens_dir = repo_root / "tests" / "fixtures" / "golden"
    assert _sha1(out_dir / "alphafold_scores.txt") == _sha1(
        goldens_dir / "alphafold_scores.txt"
    )
    assert _sha1(out_dir / "lrr_annotation_results.txt") == _sha1(
        goldens_dir / "lrr_annotation_results.txt"
    )
    assert _sha1(out_dir / "lrr_domain_sequences.fasta") == _sha1(
        goldens_dir / "lrr_domain_sequences.fasta"
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(out_dir / "ready_test_data.csv"),
        pd.read_csv(goldens_dir / "ready_test_data.csv"),
    )


# ---------------------------------------------------------------------------
# End-to-end shell-pipeline test
# ---------------------------------------------------------------------------


def _copy_repo_files_into(work: Path, repo_root: Path) -> None:
    """Stage a minimal copy of the repo into ``work`` so the shell scripts
    can run without polluting the developer's checkout.

    We copy only what the pipeline needs: the two shell scripts, the new
    src/ tree (including the production cache + weights), the example
    spreadsheet, and the legacy scripts directory (kept on PATH for full
    repo authenticity even though the new pipeline doesn't invoke it).
    """
    # Shell entry points
    for name in ("prepare_input_data.sh", "run_preparation_pipeline.sh"):
        shutil.copyfile(repo_root / name, work / name)
        os.chmod(work / name, 0o755)
    # Python source tree (includes lrr_annotation cache pickles)
    shutil.copytree(repo_root / "src", work / "src")
    # Example data
    shutil.copyfile(repo_root / "example_data.xlsx", work / "example_data.xlsx")


def test_shell_pipeline_end_to_end(
    tmp_path: Path,
    repo_root: Path,
    colabfold_outputs_dir: Path,
) -> None:
    """Run the updated shell scripts end-to-end on a staged copy of the
    repo and confirm every produced intermediate file matches its golden.

    Validates the seamless user path: ``bash run_preparation_pipeline.sh``
    is now a one-line wrapper that delegates to ``python -m mamp_ml prepare``,
    so this test exercises BOTH the shell layer and the new one-shot
    orchestrator together.
    """
    work = tmp_path / "repo"
    work.mkdir()
    _copy_repo_files_into(work, repo_root)

    # Stage the ColabFold outputs where the legacy convention expects them.
    receptor_only = work / "intermediate_files" / "receptor_only"
    receptor_only.mkdir(parents=True)
    for f in colabfold_outputs_dir.iterdir():
        if f.is_file():
            shutil.copyfile(f, receptor_only / f.name)

    # run_preparation_pipeline.sh now runs everything from receptor FASTA
    # through ready_test_data.csv in a single bash invocation.
    cp = subprocess.run(
        ["bash", "run_preparation_pipeline.sh", "example_data.xlsx"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert cp.returncode == 0, (
        f"run_preparation_pipeline.sh failed:\nstdout: {cp.stdout}\nstderr: {cp.stderr}"
    )

    # Six golden outputs to confirm.
    inter = work / "intermediate_files"
    goldens_dir = repo_root / "tests" / "fixtures" / "golden"

    # Byte-identical FASTA + TSV outputs
    for name in (
        "alphafold_scores.txt",
        "lrr_annotation_results.txt",
        "lrr_domain_sequences.fasta",
    ):
        produced = inter / name
        golden = goldens_dir / name
        assert _sha1(produced) == _sha1(golden), (
            f"{name} SHA1 mismatch.\n"
            f"  produced ({_sha1(produced)}): {produced.read_bytes()!r}\n"
            f"  golden   ({_sha1(golden)}): {golden.read_bytes()!r}"
        )

    # DataFrame-identical CSV outputs
    for name in ("test_data.csv", "ready_test_data.csv"):
        produced = inter / name
        golden = goldens_dir / name
        pd.testing.assert_frame_equal(pd.read_csv(produced), pd.read_csv(golden))

    # bfactor uses ~1e-10 tolerance because of float roundoff in Winding Number.
    produced = inter / "bfactor_winding_lrr_segments.csv"
    golden = goldens_dir / "bfactor_winding_lrr_segments.csv"
    pd.testing.assert_frame_equal(
        pd.read_csv(produced).reset_index(drop=True),
        pd.read_csv(golden).reset_index(drop=True),
        check_dtype=False,
        rtol=1e-10,
        atol=1e-12,
    )

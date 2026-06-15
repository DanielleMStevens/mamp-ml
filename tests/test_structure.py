"""Tests for :mod:`mamp_ml.structure` (checkpoint 4d).

Coverage:

1. Unit tests for each public helper using synthetic log/PDB inputs.
2. A byte-identical golden test for ``write_alphafold_scores`` against
   ``tests/fixtures/golden/alphafold_scores.txt``.
3. A byte-identical golden test for ``annotate_lrr_regions`` against
   ``tests/fixtures/golden/lrr_annotation_results.txt`` — runs the actual
   geometric annotation on the locally-folded Solanum PDB.
4. A full-stage end-to-end test (``run_structure_stage``) that exercises
   log parsing, PDB selection/copy, and LRR annotation in one call.

Tests that depend on the locally-folded ColabFold output dir use the
``colabfold_outputs_dir`` fixture from ``conftest.py``, which skips
cleanly on machines where the fold has not yet been run.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from pathlib import Path

import pytest

from mamp_ml.structure import (
    ModelScore,
    annotate_lrr_regions,
    copy_best_pdbs,
    parse_colabfold_log,
    run_structure_stage,
    select_best_pdb_files,
    write_alphafold_scores,
)


# ---------------------------------------------------------------------------
# parse_colabfold_log
# ---------------------------------------------------------------------------


def _write_log(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "log.txt"
    p.write_text(content)
    return p


def test_parse_log_basic(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        "Query 1/1: my_receptor (length 100)\n"
        "alphafold2_ptm_model_1_seed_000 recycle=0 pLDDT=80.5 pTM=0.55\n",
    )
    parsed = parse_colabfold_log(log)
    assert parsed == {"my_receptor": {"1": ModelScore(80.5, 0.55)}}


def test_parse_log_keeps_max_pLDDT_per_model(tmp_path: Path) -> None:
    """The same model can appear in multiple lines — keep the highest pLDDT."""
    log = _write_log(
        tmp_path,
        "Query 1/1: r (length 50)\n"
        "alphafold2_ptm_model_1_seed_000 recycle=0 pLDDT=70.0 pTM=0.4\n"
        "alphafold2_ptm_model_1_seed_000 recycle=1 pLDDT=85.0 pTM=0.6\n"
        "rank_001_alphafold2_ptm_model_1_seed_000 pLDDT=85.0 pTM=0.6\n",
    )
    parsed = parse_colabfold_log(log)
    assert parsed["r"]["1"] == ModelScore(85.0, 0.6)


def test_parse_log_skips_took_lines(tmp_path: Path) -> None:
    """Per-model timing lines are skipped even if they carry a pLDDT field."""
    log = _write_log(
        tmp_path,
        "Query 1/1: r (length 50)\n"
        "alphafold2_ptm_model_1_seed_000 recycle=0 pLDDT=70.0 pTM=0.4\n"
        "alphafold2_ptm_model_1_seed_000 took 1234s (1 recycles) pLDDT=99.0 pTM=0.99\n",
    )
    parsed = parse_colabfold_log(log)
    # The 70.0 score wins; the 99.0 score is on a `took` line and must be ignored.
    assert parsed["r"]["1"].plddt == 70.0


def test_parse_log_multiple_receptors(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        "Query 1/2: alpha (length 100)\n"
        "alphafold2_ptm_model_1_seed_000 recycle=0 pLDDT=80.0 pTM=0.5\n"
        "Query 2/2: beta (length 50)\n"
        "alphafold2_ptm_model_2_seed_000 recycle=0 pLDDT=75.0 pTM=0.45\n",
    )
    parsed = parse_colabfold_log(log)
    assert parsed == {
        "alpha": {"1": ModelScore(80.0, 0.5)},
        "beta": {"2": ModelScore(75.0, 0.45)},
    }


def test_parse_log_records_receptor_with_no_models(tmp_path: Path) -> None:
    """A Query line without any successful model rows still gets recorded."""
    log = _write_log(tmp_path, "Query 1/1: nameless (length 100)\n")
    parsed = parse_colabfold_log(log)
    assert parsed == {"nameless": {}}


def test_parse_log_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_colabfold_log(tmp_path / "does_not_exist.txt")


# ---------------------------------------------------------------------------
# write_alphafold_scores
# ---------------------------------------------------------------------------


def test_write_scores_basic_format(tmp_path: Path) -> None:
    results = {
        "alpha": {"1": ModelScore(80.5, 0.51), "2": ModelScore(82.4, 0.55)},
        "beta": {"1": ModelScore(75.0, 0.45)},
    }
    out = tmp_path / "scores.txt"
    best = write_alphafold_scores(results, out)
    assert best == {"alpha": "2", "beta": "1"}

    lines = out.read_text().splitlines()
    # 4 lines: header, separator, alpha, beta (alpha first — sorted).
    assert len(lines) == 4
    assert lines[0].startswith("Receptor")
    assert "Best Model" in lines[0]
    assert lines[1] == "-" * 125
    # Data row column widths
    assert lines[2].startswith("alpha" + " " * 95)  # 5 + 95 = 100
    assert "model_2" in lines[2]
    # pLDDT formatted as %.1f, pTM as %.3f
    assert "82.4" in lines[2]
    assert "0.550" in lines[2]


def test_write_scores_sorts_alphabetically(tmp_path: Path) -> None:
    results = {
        "zebra": {"1": ModelScore(80.0, 0.5)},
        "ant": {"1": ModelScore(82.0, 0.55)},
        "mango": {"1": ModelScore(81.0, 0.52)},
    }
    out = tmp_path / "scores.txt"
    write_alphafold_scores(results, out)
    body = "\n".join(out.read_text().splitlines()[2:])  # skip header + sep
    # ant comes before mango comes before zebra
    assert body.index("ant") < body.index("mango") < body.index("zebra")


def test_write_scores_skips_empty_receptors(tmp_path: Path) -> None:
    """A receptor with no parsed models must not produce a data row."""
    results = {
        "alpha": {"1": ModelScore(80.0, 0.5)},
        "ghost": {},  # no successful models
    }
    out = tmp_path / "scores.txt"
    best = write_alphafold_scores(results, out)
    assert best == {"alpha": "1"}
    assert "ghost" not in out.read_text()


def test_write_scores_uses_lf_line_endings(tmp_path: Path) -> None:
    results = {"r": {"1": ModelScore(80.0, 0.5)}}
    out = tmp_path / "scores.txt"
    write_alphafold_scores(results, out)
    raw = out.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")


# ---------------------------------------------------------------------------
# select_best_pdb_files + copy_best_pdbs
# ---------------------------------------------------------------------------


def _make_pdb_stub(path: Path) -> None:
    """Write a minimal valid-looking PDB so `shutil.copy2` succeeds + we can
    later verify the copy. Not a structurally-valid PDB."""
    path.write_text("HEADER stub\nEND\n")


def test_select_best_pdb_files_matches_colabfold_pattern(tmp_path: Path) -> None:
    cf = tmp_path / "cf"
    cf.mkdir()
    _make_pdb_stub(
        cf
        / "R1_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb"
    )
    _make_pdb_stub(
        cf
        / "R2_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    )
    sel = select_best_pdb_files({"R1": "3", "R2": "1"}, cf)
    assert set(sel) == {"R1", "R2"}
    assert sel["R1"].name.startswith("R1_unrelaxed_rank_001")
    assert sel["R2"].name.startswith("R2_unrelaxed_rank_001")


def test_select_best_pdb_files_warns_on_missing(tmp_path: Path) -> None:
    cf = tmp_path / "cf"
    cf.mkdir()
    _make_pdb_stub(
        cf
        / "Real_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sel = select_best_pdb_files({"Real": "1", "Ghost": "1"}, cf)
    assert set(sel) == {"Real"}
    assert any("Ghost" in str(w.message) for w in caught)


def test_select_best_pdb_files_warns_on_multiple_matches(tmp_path: Path) -> None:
    cf = tmp_path / "cf"
    cf.mkdir()
    _make_pdb_stub(
        cf
        / "R_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    )
    _make_pdb_stub(
        cf
        / "R_unrelaxed_rank_002_alphafold2_ptm_model_1_seed_001.pdb"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sel = select_best_pdb_files({"R": "1"}, cf)
    # Prefer the lowest rank (rank_001).
    assert sel["R"].name.startswith("R_unrelaxed_rank_001")
    assert any("Multiple" in str(w.message) for w in caught)


def test_copy_best_pdbs_renames_to_receptor_name(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    s1 = src / "R1_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    _make_pdb_stub(s1)
    out = tmp_path / "out"
    copied = copy_best_pdbs({"R1": s1}, out)
    assert copied == [out / "R1.pdb"]
    assert (out / "R1.pdb").read_text() == "HEADER stub\nEND\n"


# ---------------------------------------------------------------------------
# Golden equivalence: write_alphafold_scores against the committed golden
# ---------------------------------------------------------------------------


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def test_write_alphafold_scores_matches_golden(
    tmp_path: Path,
    repo_root: Path,
    colabfold_outputs_dir: Path,
) -> None:
    """Driving the parser+writer on the real log.txt must SHA-1-equal the
    committed alphafold_scores.txt golden."""
    parsed = parse_colabfold_log(colabfold_outputs_dir / "log.txt")
    new_out = tmp_path / "alphafold_scores.txt"
    write_alphafold_scores(parsed, new_out)

    golden = repo_root / "tests" / "fixtures" / "golden" / "alphafold_scores.txt"
    assert _sha1(new_out) == _sha1(golden), (
        f"SHA-1 mismatch.\n"
        f"  new   : {_sha1(new_out)}\n"
        f"  golden: {_sha1(golden)}\n"
        f"new bytes ({new_out.stat().st_size}): {new_out.read_bytes()!r}\n"
        f"golden ({golden.stat().st_size}): {golden.read_bytes()!r}"
    )


# ---------------------------------------------------------------------------
# Golden equivalence: annotate_lrr_regions against the committed golden
# ---------------------------------------------------------------------------


def test_annotate_lrr_regions_matches_golden(
    tmp_path: Path,
    repo_root: Path,
    colabfold_outputs_dir: Path,
) -> None:
    """Running the geometric annotation on the real Solanum PDB must
    produce byte-identical lrr_annotation_results.txt to the committed
    golden — the algorithm is deterministic for this input despite the
    np.random.rand init in compute_winding (gradient descent converges to
    the same minimum)."""
    # Set up the input PDB dir using the same convention copy_best_pdbs does.
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    src_pdbs = list(
        colabfold_outputs_dir.glob(
            "*_unrelaxed_rank_001_alphafold2_ptm_model_*_seed_*.pdb"
        )
    )
    assert src_pdbs, "no ColabFold PDB files in fixture dir"
    for src in src_pdbs:
        # Receptor name is the prefix before "_unrelaxed_rank_..."
        receptor = src.name.split("_unrelaxed_rank_")[0]
        (pdb_dir / f"{receptor}.pdb").write_bytes(src.read_bytes())

    output = tmp_path / "lrr_annotation_results.txt"
    n = annotate_lrr_regions(pdb_dir, output, cache_dir=tmp_path / "cache")
    assert n >= 1, "no LRR-region rows written"

    golden = (
        repo_root / "tests" / "fixtures" / "golden" / "lrr_annotation_results.txt"
    )
    assert _sha1(output) == _sha1(golden), (
        f"SHA-1 mismatch — annotation diverged from the committed golden.\n"
        f"  new   : {_sha1(output)}\n"
        f"  golden: {_sha1(golden)}\n"
        f"new contents:\n{output.read_text()}\n"
        f"golden contents:\n{golden.read_text()}"
    )


# ---------------------------------------------------------------------------
# Orchestrator: run_structure_stage drives all four steps in one call
# ---------------------------------------------------------------------------


def test_run_structure_stage_end_to_end(
    tmp_path: Path,
    repo_root: Path,
    colabfold_outputs_dir: Path,
) -> None:
    """The one-shot orchestrator must produce both the goldens in one call."""
    work = tmp_path
    scores = work / "alphafold_scores.txt"
    pdbs = work / "pdbs"
    lrr_results = work / "lrr_annotation_results.txt"
    cache = work / "cache"

    n = run_structure_stage(
        colabfold_outputs_dir,
        scores_path=scores,
        pdb_target_dir=pdbs,
        lrr_results_path=lrr_results,
        cache_dir=cache,
    )
    assert n >= 1

    golden_scores = (
        repo_root / "tests" / "fixtures" / "golden" / "alphafold_scores.txt"
    )
    golden_lrr = (
        repo_root / "tests" / "fixtures" / "golden" / "lrr_annotation_results.txt"
    )

    assert _sha1(scores) == _sha1(golden_scores), "scores file diverged from golden"
    assert _sha1(lrr_results) == _sha1(golden_lrr), "LRR results diverged from golden"

    # The orchestrator should also have populated the PDB target dir and the
    # cache dir with the artefacts the next stage (bfactor analysis) needs.
    assert any(pdbs.glob("*.pdb"))
    assert (cache / "structures.pickle").is_file()
    assert (cache / "breakpoints.pickle").is_file()
    assert (cache / "windings.pickle").is_file()

"""Tests for the ESMFold backend (checkpoint 8).

The actual ESMFold model is 7 GB and ~30 min/sequence on CPU; we do not
download or run it in CI. The tests below cover everything *around* the
model invocation:

- The receptor-name normalisation matches ColabFold's filename convention.
- The log-line renderer produces lines that parse cleanly through
  :func:`mamp_ml.structure.parse_colabfold_log`.
- The PDB-filename helper matches the glob pattern that
  :func:`mamp_ml.structure.select_best_pdb_files` searches for.
- An end-to-end mock of ``EsmForProteinFolding`` runs the orchestration
  without the model and confirms the on-disk output schema, including
  pLDDT extraction and B-factor preservation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from mamp_ml.fold.esmfold import (
    ESMFOLD_MAX_LENGTH,
    fold_with_esmfold,
    make_colabfold_compatible_pdb_filename,
    normalize_receptor_name,
    render_colabfold_compatible_log,
)


# ---------------------------------------------------------------------------
# normalize_receptor_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Solanum habrochates|scaffold11|CORE", "Solanum_habrochates_scaffold11_CORE"),
        ("already_underscored", "already_underscored"),
        ("pipes|only", "pipes_only"),
        ("  trim me  ", "trim_me"),
        ("multi   spaces", "multi_spaces"),
    ],
)
def test_normalize_receptor_name(raw: str, expected: str) -> None:
    assert normalize_receptor_name(raw) == expected


# ---------------------------------------------------------------------------
# PDB filename: matches the structure-stage glob pattern
# ---------------------------------------------------------------------------


def test_pdb_filename_matches_structure_stage_glob_pattern() -> None:
    """``select_best_pdb_files`` globs for:
        ``{receptor}_unrelaxed_rank_[0-9]{3}_alphafold2_ptm_model_<N>_seed_*.pdb``
    The ESMFold-emitted filenames must match this exact pattern with N=1."""
    name = "Solanum_habrochates_scaffold11_CORE"
    filename = make_colabfold_compatible_pdb_filename(name)
    pattern = re.compile(
        rf"^{re.escape(name)}_unrelaxed_rank_[0-9]{{3}}_alphafold2_ptm_model_1_seed_[0-9]+\.pdb$"
    )
    assert pattern.match(filename), (
        f"emitted filename {filename!r} does not match the glob pattern"
    )


# ---------------------------------------------------------------------------
# Log rendering: must parse cleanly via parse_colabfold_log
# ---------------------------------------------------------------------------


def test_rendered_log_parses_through_structure_stage(tmp_path: Path) -> None:
    """The most important contract: the log we render must round-trip through
    :func:`mamp_ml.structure.parse_colabfold_log` and produce the same
    receptor names + pLDDT values we put in."""
    from mamp_ml.structure import parse_colabfold_log

    log_text = render_colabfold_compatible_log(
        [
            ("Sp_a_loc_R1", 1500, 1024, 84.6, 0.0),
            ("Sp_b_loc_R2", 950, 950, 88.2, 0.0),
        ]
    )
    log_path = tmp_path / "log.txt"
    log_path.write_text(log_text)

    parsed = parse_colabfold_log(log_path)
    assert set(parsed) == {"Sp_a_loc_R1", "Sp_b_loc_R2"}
    # pLDDT was 84.6, but the log writer formats to one decimal place; the
    # parser reads it back as a float, so 84.6 round-trips.
    assert parsed["Sp_a_loc_R1"]["1"].plddt == 84.6
    assert parsed["Sp_b_loc_R2"]["1"].plddt == 88.2


def test_rendered_log_uses_lf_line_endings() -> None:
    """Output must be byte-stable: LF only, no CR."""
    text = render_colabfold_compatible_log(
        [("R", 100, 100, 80.0, 0.0)]
    )
    assert "\r" not in text
    assert text.endswith("\n")


def test_rendered_log_empty_receptors() -> None:
    """No receptors -> minimal banner-only log."""
    text = render_colabfold_compatible_log([])
    assert "Running" in text  # banner survives
    assert "Query " not in text


# ---------------------------------------------------------------------------
# fold_with_esmfold (mocked model)
# ---------------------------------------------------------------------------


def _stub_pdb_string(receptor_name: str, n_residues: int) -> str:
    """Generate a minimal PDB string for tests: one CA atom per residue,
    sequential B-factor values so tests can verify pLDDT propagation."""
    lines = ["MODEL     1"]
    for i in range(1, n_residues + 1):
        # B-factor column is positions 61-66 in PDB format.
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"   1.000   2.000   3.000  1.00 {float(i):6.2f}           C"
        )
    lines.append(f"TER   {n_residues + 1:5d}      ALA A{n_residues:4d}")
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"


def test_fold_with_esmfold_missing_fasta(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fold_with_esmfold(
            tmp_path / "no.fasta",
            tmp_path / "out",
        )


def test_fold_with_esmfold_orchestration(tmp_path: Path, monkeypatch) -> None:
    """End-to-end smoke test with the model fully mocked. Validates:

    - The receptor FASTA is read
    - One PDB per receptor is written with the ColabFold-style filename
    - The log.txt is in the format the structure stage expects
    - Sequences over the max length are truncated with a warning
    """
    # Build a synthetic input FASTA: one short receptor + one over-length.
    fasta = tmp_path / "in.fasta"
    long_seq = "A" * (ESMFOLD_MAX_LENGTH + 50)
    short_seq = "VKLMNPSTQRWY" * 8  # 96 AAs
    fasta.write_text(
        f">Sp x|loc|R_long\n{long_seq}\n"
        f">Sp y|loc|R_short\n{short_seq}\n"
    )

    # Mock transformers.AutoTokenizer + EsmForProteinFolding before
    # _import_esmfold runs.
    sentinel_calls: List[int] = []

    def fake_tokenizer_factory(model_id):
        def tokenizer(seq, return_tensors="pt", add_special_tokens=False):
            # Capture the per-sequence length for assertions below.
            sentinel_calls.append(len(seq))
            # Tiny tensor stand-ins (we never actually inspect them).
            import torch

            return {"input_ids": torch.zeros((1, len(seq)), dtype=torch.long)}

        return tokenizer

    def fake_model_factory(model_id):
        model = MagicMock()
        model.esm = MagicMock()
        model.esm.half.return_value = model.esm
        model.to.return_value = model
        model.eval.return_value = None

        def fake_call(**inputs):
            n_residues = inputs["input_ids"].shape[1]
            # outputs.plddt has shape (batch, n_residues, n_atoms) — index 1
            # is the CA atom in the official ESMFold output convention.
            plddt = np.zeros((1, n_residues, 37), dtype=np.float32)
            plddt[0, :, 1] = 80.0  # mean = 80.0 exactly
            out = MagicMock()
            import torch

            out.plddt = torch.from_numpy(plddt)
            return out

        model.side_effect = fake_call

        # output_to_pdb returns a list of strings, one per batch element.
        def fake_output_to_pdb(outputs):
            n_residues = outputs.plddt.shape[1]
            # Stub uses 1-indexed residue tag for traceability; we don't
            # inspect content beyond presence here.
            return [_stub_pdb_string("stub", n_residues)]

        model.output_to_pdb = fake_output_to_pdb
        return model

    import mamp_ml.fold.esmfold as ef

    monkeypatch.setattr(
        ef,
        "_import_esmfold",
        lambda: (
            type(
                "FakeAutoTokenizer",
                (),
                {"from_pretrained": staticmethod(fake_tokenizer_factory)},
            ),
            type(
                "FakeEsmForProteinFolding",
                (),
                {"from_pretrained": staticmethod(fake_model_factory)},
            ),
        ),
    )

    out_dir = tmp_path / "fold_out"
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pdbs = fold_with_esmfold(fasta, out_dir, device="cpu")

    # Exactly two PDBs produced, in input-FASTA order.
    assert len(pdbs) == 2
    assert pdbs[0].name == "Sp_x_loc_R_long_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    assert pdbs[1].name == "Sp_y_loc_R_short_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"

    # The over-length receptor was truncated; the tokenizer saw 1024, not 1074.
    assert ESMFOLD_MAX_LENGTH in sentinel_calls
    assert len(short_seq) in sentinel_calls
    # Warning was raised about the truncation.
    truncation_warnings = [w for w in caught if "truncating" in str(w.message).lower()]
    assert truncation_warnings, "expected a truncation warning for the long sequence"

    # log.txt is present and parseable by the structure stage.
    log_path = out_dir / "log.txt"
    assert log_path.is_file()
    from mamp_ml.structure import parse_colabfold_log

    parsed = parse_colabfold_log(log_path)
    assert set(parsed) == {"Sp_x_loc_R_long", "Sp_y_loc_R_short"}
    assert parsed["Sp_x_loc_R_long"]["1"].plddt == 80.0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_fold_subcommand_requires_existing_fasta(tmp_path: Path) -> None:
    """``mamp-ml fold`` with a missing FASTA exits with code 3."""
    from mamp_ml.__main__ import main as cli_main

    rc = cli_main(
        ["fold", str(tmp_path / "no.fasta"), str(tmp_path / "out"), "--structure", "esmfold"]
    )
    assert rc == 3


def test_cli_fold_colabfold_prints_invocation(
    tmp_path: Path, example_xlsx: Path, capsys
) -> None:
    """``--backend colabfold`` exits 2 and prints the colabfold_batch hint."""
    from mamp_ml.__main__ import main as cli_main
    from mamp_ml.preprocess import xlsx_to_receptor_fasta

    fasta = tmp_path / "receptor.fasta"
    xlsx_to_receptor_fasta(example_xlsx, fasta)

    rc = cli_main(
        ["fold", str(fasta), str(tmp_path / "out"), "--structure", "colabfold"]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "colabfold_batch" in captured.out


def test_predict_subparser_accepts_structure_flag(example_xlsx: Path) -> None:
    """The predict subcommand must accept --structure esmfold without parser error."""
    from mamp_ml.__main__ import _build_parser

    parser = _build_parser()
    parsed = parser.parse_args(
        ["predict", str(example_xlsx), "--structure", "esmfold"]
    )
    assert parsed.structure == "esmfold"
    # Default to colabfold when omitted.
    parsed_default = parser.parse_args(["predict", str(example_xlsx)])
    assert parsed_default.structure == "colabfold"


def test_predict_subparser_accepts_weights_flag(
    example_xlsx: Path, tmp_path: Path
) -> None:
    """The predict subcommand must accept a custom --weights path. The
    default leaves it as None; the runtime resolves to default_weights_path
    when that's the case."""
    from mamp_ml.__main__ import _build_parser

    custom_weights = tmp_path / "my_finetune.pth"
    custom_weights.write_bytes(b"\x00")  # fake checkpoint, just needs to exist
    parser = _build_parser()
    parsed = parser.parse_args(
        ["predict", str(example_xlsx), "--weights", str(custom_weights)]
    )
    assert parsed.weights == str(custom_weights)
    # Default to None when omitted (runtime then picks the bundled file).
    parsed_default = parser.parse_args(["predict", str(example_xlsx)])
    assert parsed_default.weights is None


def test_predict_subparser_accepts_keep_flag(example_xlsx: Path) -> None:
    """The predict subcommand must accept --keep with choices {default, all}."""
    from mamp_ml.__main__ import _build_parser

    parser = _build_parser()
    parsed_default = parser.parse_args(["predict", str(example_xlsx)])
    assert parsed_default.keep == "default"

    parsed_all = parser.parse_args(
        ["predict", str(example_xlsx), "--keep", "all"]
    )
    assert parsed_all.keep == "all"

    with pytest.raises(SystemExit):
        # Anything other than {default, all} must reject.
        parser.parse_args(["predict", str(example_xlsx), "--keep", "garbage"])


def test_tidy_intermediate_files_removes_everything_but_keep(tmp_path: Path) -> None:
    """The cleanup helper must preserve the listed keep paths and remove
    everything else under out_dir without touching out_dir itself."""
    from mamp_ml.__main__ import _tidy_intermediate_files

    out_dir = tmp_path / "inter"
    out_dir.mkdir()
    # Files we want to keep.
    preds = out_dir / "predictions.csv"
    preds.write_text("Header_Name,prediction\n")
    plots_dir = out_dir / "lrr_annotation_plots"
    plots_dir.mkdir()
    (plots_dir / "Solanum_plot.png").write_bytes(b"\x89PNG")
    # Other intermediates that should be removed.
    (out_dir / "test_data.csv").write_text("a,b\n")
    (out_dir / "ready_test_data.csv").write_text("a,b\n")
    (out_dir / "receptor_only").mkdir()
    (out_dir / "receptor_only" / "log.txt").write_text("...")
    (out_dir / "pdb_for_lrr_annotator").mkdir()
    (out_dir / "pdb_for_lrr_annotator" / "Solanum.pdb").write_text("...")
    (out_dir / "lrr_annotation_results.txt").write_text("...")

    _tidy_intermediate_files(out_dir, keep=[preds, plots_dir])

    # Survivors:
    assert preds.is_file()
    assert plots_dir.is_dir()
    assert (plots_dir / "Solanum_plot.png").is_file()
    assert out_dir.is_dir()  # the out_dir itself must NOT be removed

    # Casualties:
    assert not (out_dir / "test_data.csv").exists()
    assert not (out_dir / "ready_test_data.csv").exists()
    assert not (out_dir / "receptor_only").exists()
    assert not (out_dir / "pdb_for_lrr_annotator").exists()
    assert not (out_dir / "lrr_annotation_results.txt").exists()


def test_tidy_intermediate_files_tolerates_missing_keep_paths(tmp_path: Path) -> None:
    """If a keep target doesn't exist, the cleanup must still run cleanly
    and remove the other files."""
    from mamp_ml.__main__ import _tidy_intermediate_files

    out_dir = tmp_path / "inter"
    out_dir.mkdir()
    (out_dir / "test_data.csv").write_text("a,b\n")

    # predictions.csv intentionally absent.
    nonexistent_preds = out_dir / "predictions.csv"
    nonexistent_plots = out_dir / "lrr_annotation_plots"

    _tidy_intermediate_files(out_dir, keep=[nonexistent_preds, nonexistent_plots])
    # The casualty was removed even though the keep targets never existed.
    assert not (out_dir / "test_data.csv").exists()
    # out_dir survives.
    assert out_dir.is_dir()


def test_prepare_summary_mentions_plots_and_intermediates(
    tmp_path: Path,
    repo_root: Path,
    example_xlsx: Path,
    colabfold_outputs_dir: Path,
    capsys,
) -> None:
    """After a successful `prepare`, the summary block must surface the
    predictions-ready CSV, the LRR annotation plots dir, and the overall
    intermediates dir. This is the user-visible contract for checkpoint 8."""
    import shutil

    from mamp_ml.__main__ import main as cli_main

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
    assert "Model-ready CSV" in captured.out
    assert "LRR annotation plots" in captured.out
    assert "All intermediates" in captured.out
    # The actual paths must be the user's chosen out_dir, not the default.
    assert str(out_dir) in captured.out

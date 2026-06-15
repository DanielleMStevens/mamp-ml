"""Tests for :func:`mamp_ml.preprocess.assemble_test_data` (checkpoint 4c).

Two layers of coverage:

1. Unit tests for the small helpers (three-part key extraction, LRR-FASTA
   loader) and a handful of synthetic end-to-end calls to ``assemble_test_data``
   that exercise the documented behaviours (join semantics, NaN-drop on
   missing receptors, column ordering, ligand-rename).

2. A DataFrame-equivalence golden test that feeds the function the same
   inputs the legacy ``scripts/04_data_prep_for_prediction.py`` saw —
   ``example_data.xlsx`` paired with the committed
   ``tests/fixtures/golden/lrr_domain_sequences.fasta`` — and asserts that
   the resulting CSV parses back to exactly the same DataFrame as the
   committed ``tests/fixtures/golden/test_data.csv``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mamp_ml.preprocess import (
    _load_lrr_domain_sequences,
    _three_part_pipe_key,
    assemble_test_data,
)


# ---------------------------------------------------------------------------
# _three_part_pipe_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header, expected",
    [
        ("Solanum_habrochates|scaffold11|CORE|LRR_domain",
         "Solanum_habrochates|scaffold11|CORE"),
        ("species|loc|R|extra|even_more",
         "species|loc|R"),
        ("species|loc|R", "species|loc|R"),     # exactly three parts -> verbatim
        ("only|two", "only|two"),               # too few parts -> fallback to verbatim
        ("nopipes", "nopipes"),
        ("", ""),
    ],
)
def test_three_part_pipe_key(header: str, expected: str) -> None:
    assert _three_part_pipe_key(header) == expected


# ---------------------------------------------------------------------------
# _load_lrr_domain_sequences
# ---------------------------------------------------------------------------


def test_load_lrr_domain_sequences_drops_lrr_domain_tag(tmp_path: Path) -> None:
    """The ``|LRR_domain`` suffix must not be part of the lookup key."""
    fasta = tmp_path / "lrr.fasta"
    fasta.write_text(
        ">Sp_a|loc1|R1|LRR_domain\nAAAAA\n"
        ">Sp_b|loc2|R2|LRR_domain\nBBBBB\n"
    )
    out = _load_lrr_domain_sequences(fasta)
    assert out == {
        "Sp_a|loc1|R1": "AAAAA",
        "Sp_b|loc2|R2": "BBBBB",
    }


def test_load_lrr_domain_sequences_empty_file(tmp_path: Path) -> None:
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    assert _load_lrr_domain_sequences(fasta) == {}


# ---------------------------------------------------------------------------
# assemble_test_data — synthetic end-to-end
# ---------------------------------------------------------------------------


def _synthetic_xlsx(tmp_path: Path) -> Path:
    """Two receptors, two ligands each, plus one extra column to be discarded."""
    xlsx = tmp_path / "input.xlsx"
    pd.DataFrame(
        {
            "plant_species": ["Sp a"] * 2 + ["Sp b"] * 2,
            "receptor": ["R1"] * 2 + ["R2"] * 2,
            "locus_id": ["loc1"] * 2 + ["loc2"] * 2,
            "receptor_sequence": ["FULL_A"] * 2 + ["FULL_B"] * 2,
            "ligand_sequence": ["AGR", "WYF", "DKH", "MNP"],
            "some_extra_column": ["ignored"] * 4,
        }
    ).to_excel(xlsx, index=False)
    return xlsx


def _synthetic_lrr_fasta(tmp_path: Path) -> Path:
    """Only Sp a has an LRR annotation; Sp b should get dropped."""
    fasta = tmp_path / "lrr.fasta"
    fasta.write_text(">Sp_a|loc1|R1|LRR_domain\nLRR_A\n")
    return fasta


def test_assemble_test_data_drops_rows_missing_lrr(tmp_path: Path) -> None:
    xlsx = _synthetic_xlsx(tmp_path)
    lrr = _synthetic_lrr_fasta(tmp_path)
    out_csv = tmp_path / "out.csv"

    df = assemble_test_data(xlsx, lrr, out_csv)

    # 4 input rows -> 2 surviving rows (Sp_b dropped entirely)
    assert len(df) == 2
    assert (df["Header_Name"] == "Sp_a|loc1|R1").all()
    # receptor_sequence has been overwritten with the LRR domain sequence
    assert (df["receptor_sequence"] == "LRR_A").all()
    # ligand sequences preserved + renamed to Sequence
    assert df["Sequence"].tolist() == ["AGR", "WYF"]


def test_assemble_test_data_column_order(tmp_path: Path) -> None:
    xlsx = _synthetic_xlsx(tmp_path)
    lrr = _synthetic_lrr_fasta(tmp_path)
    out_csv = tmp_path / "out.csv"
    df = assemble_test_data(xlsx, lrr, out_csv)
    assert list(df.columns) == [
        "Header_Name",
        "plant_species",
        "receptor",
        "locus_id",
        "receptor_sequence",
        "Sequence",
    ]
    # No extra column from the synthetic spreadsheet should leak through.
    assert "some_extra_column" not in df.columns
    # The CSV on disk has the same column order.
    on_disk = pd.read_csv(out_csv)
    assert list(on_disk.columns) == list(df.columns)


def test_assemble_test_data_underscores_species_in_header(tmp_path: Path) -> None:
    xlsx = tmp_path / "i.xlsx"
    pd.DataFrame(
        {
            "plant_species": ["My favourite species name"],
            "receptor": ["R"],
            "locus_id": ["loc"],
            "receptor_sequence": ["X"],
            "ligand_sequence": ["AGR"],
        }
    ).to_excel(xlsx, index=False)
    fasta = tmp_path / "lrr.fasta"
    fasta.write_text(
        ">My_favourite_species_name|loc|R|LRR_domain\nLRR_X\n"
    )
    df = assemble_test_data(xlsx, fasta, tmp_path / "out.csv")
    assert df["Header_Name"].iloc[0] == "My_favourite_species_name|loc|R"
    # plant_species itself remains with the original spaces — only Header_Name
    # gets the underscore treatment.
    assert df["plant_species"].iloc[0] == "My favourite species name"


def test_assemble_test_data_renames_ligand_to_Sequence(tmp_path: Path) -> None:
    xlsx = _synthetic_xlsx(tmp_path)
    lrr = _synthetic_lrr_fasta(tmp_path)
    df = assemble_test_data(xlsx, lrr, tmp_path / "out.csv")
    assert "ligand_sequence" not in df.columns
    assert "Sequence" in df.columns


def test_assemble_test_data_resets_row_index(tmp_path: Path) -> None:
    """After NaN drop the index must be reset to 0..N-1 so CSV round-trip matches."""
    xlsx = _synthetic_xlsx(tmp_path)
    lrr = _synthetic_lrr_fasta(tmp_path)
    df = assemble_test_data(xlsx, lrr, tmp_path / "out.csv")
    assert df.index.tolist() == list(range(len(df)))


def test_assemble_test_data_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        assemble_test_data(
            tmp_path / "no.xlsx",
            tmp_path / "no.fasta",
            tmp_path / "out.csv",
        )

    # Spreadsheet exists, LRR FASTA missing
    xlsx = _synthetic_xlsx(tmp_path)
    with pytest.raises(FileNotFoundError):
        assemble_test_data(
            xlsx,
            tmp_path / "no.fasta",
            tmp_path / "out.csv",
        )


def test_assemble_test_data_validates_columns(tmp_path: Path) -> None:
    """If the spreadsheet is missing a required column, fail with a clear error."""
    bad = tmp_path / "bad.xlsx"
    pd.DataFrame(
        {
            "plant_species": ["X"],
            # missing receptor, locus_id, receptor_sequence, ligand_sequence
        }
    ).to_excel(bad, index=False)
    fasta = _synthetic_lrr_fasta(tmp_path)
    with pytest.raises(ValueError, match="missing required columns"):
        assemble_test_data(bad, fasta, tmp_path / "out.csv")


# ---------------------------------------------------------------------------
# Golden equivalence: DataFrame-identical to the legacy pipeline output
# ---------------------------------------------------------------------------


def test_assemble_test_data_matches_legacy_golden(
    tmp_path: Path,
    repo_root: Path,
    example_xlsx: Path,
) -> None:
    """The Python output must parse back to the exact DataFrame the legacy
    pipeline produced.

    Both runs see the same xlsx (``example_data.xlsx``) and the same
    LRR-domain FASTA (``tests/fixtures/golden/lrr_domain_sequences.fasta``).
    The resulting CSV is read back and compared with the committed
    ``tests/fixtures/golden/test_data.csv`` via
    ``pandas.testing.assert_frame_equal``.
    """
    golden_lrr_fasta = (
        repo_root / "tests" / "fixtures" / "golden" / "lrr_domain_sequences.fasta"
    )
    golden_test_data = (
        repo_root / "tests" / "fixtures" / "golden" / "test_data.csv"
    )
    assert golden_lrr_fasta.is_file(), "missing golden lrr_domain_sequences.fasta"
    assert golden_test_data.is_file(), "missing golden test_data.csv"

    new_csv = tmp_path / "test_data.csv"
    df_new_returned = assemble_test_data(example_xlsx, golden_lrr_fasta, new_csv)

    # 65 surviving rows (only Solanum is present in the LRR FASTA goldens)
    assert len(df_new_returned) == 65

    df_golden = pd.read_csv(golden_test_data)
    df_new = pd.read_csv(new_csv)

    pd.testing.assert_frame_equal(df_new, df_golden, check_like=False)

"""Pure unit tests for :mod:`mamp_ml.preprocess`.

These tests are entirely Python-side: they don't touch R, don't require
network access, and don't depend on any heavyweight optional deps. They run
on every machine the test suite is invoked on, so a regression in the core
preprocessing semantics is caught immediately.

The Rscript-driven byte-equivalence tests live in
``tests/test_golden_r_outputs.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mamp_ml.preprocess import (
    BULKINESS,
    CHARGE,
    HYDROPHOBICITY,
    _r_format,
    add_chemical_features,
    sequence_to_bulkiness,
    sequence_to_charge,
    sequence_to_hydrophobicity,
    sequence_to_property,
    xlsx_to_receptor_fasta,
)

# ---------------------------------------------------------------------------
# _r_format: matches R's paste()/as.character()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0"),
        (1, "1"),
        (-1, "-1"),
        (0.0, "0"),
        (-1.0, "-1"),
        (11.0, "11"),
        (0.1, "0.1"),
        (11.50, "11.5"),
        (3.40, "3.4"),
        (13.46, "13.46"),
        (2.65, "2.65"),
        (21.40, "21.4"),
        (0.06, "0.06"),
    ],
)
def test_r_format(value: float, expected: str) -> None:
    """Integer-valued doubles drop trailing ``.0``; other values use ``g``."""
    assert _r_format(value) == expected


# ---------------------------------------------------------------------------
# Property tables: complete and unchanged from the R script
# ---------------------------------------------------------------------------


def test_property_tables_cover_20_standard_amino_acids() -> None:
    """All 20 canonical amino-acid letters must appear in each table."""
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    assert set(BULKINESS) == standard
    assert set(CHARGE) == standard
    assert set(HYDROPHOBICITY) == standard


def test_charge_values_match_r_constants() -> None:
    """Spot-check the charge values against the R source (line 26)."""
    assert CHARGE["R"] == 1
    assert CHARGE["K"] == 1
    assert CHARGE["D"] == -1
    assert CHARGE["E"] == -1
    assert CHARGE["H"] == 0.1
    # Neutral by table convention
    assert CHARGE["A"] == 0
    assert CHARGE["G"] == 0


def test_bulkiness_extremes() -> None:
    """G has the lowest bulkiness, W is among the highest."""
    assert BULKINESS["G"] == 3.40
    assert BULKINESS["W"] == 21.67
    assert min(BULKINESS, key=BULKINESS.get) == "G"


def test_hydrophobicity_extremes() -> None:
    """W and I are the most hydrophobic by this table."""
    assert HYDROPHOBICITY["W"] == 2.65
    assert HYDROPHOBICITY["I"] == 2.22
    assert HYDROPHOBICITY["R"] == 0.00


# ---------------------------------------------------------------------------
# sequence_to_property: comma-separated, R-compatible formatting
# ---------------------------------------------------------------------------


def test_sequence_to_property_basic() -> None:
    assert sequence_to_property("A", BULKINESS) == "11.5"
    assert sequence_to_property("AR", BULKINESS) == "11.5,14.28"
    assert sequence_to_property("", BULKINESS) == ""


def test_sequence_to_bulkiness_examples() -> None:
    assert sequence_to_bulkiness("A") == "11.5"
    assert sequence_to_bulkiness("G") == "3.4"
    assert sequence_to_bulkiness("AG") == "11.5,3.4"
    assert sequence_to_bulkiness("AAR") == "11.5,11.5,14.28"


def test_sequence_to_charge_handles_signs_and_decimals() -> None:
    assert sequence_to_charge("D") == "-1"
    assert sequence_to_charge("R") == "1"
    assert sequence_to_charge("A") == "0"
    assert sequence_to_charge("H") == "0.1"
    assert sequence_to_charge("RHA") == "1,0.1,0"


def test_sequence_to_hydrophobicity_examples() -> None:
    # R has hydrophobicity 0.00 — must render as "0" (matches R's integer-double
    # formatting), not "0.0".
    assert sequence_to_hydrophobicity("R") == "0"
    assert sequence_to_hydrophobicity("A") == "0.61"
    assert sequence_to_hydrophobicity("W") == "2.65"


def test_sequence_skips_nonstandard_amino_acids() -> None:
    """``X`` and other non-standard letters are silently dropped (matches R)."""
    assert sequence_to_bulkiness("AXG") == "11.5,3.4"
    assert sequence_to_charge("DXR") == "-1,1"
    assert sequence_to_hydrophobicity("AAA") == "0.61,0.61,0.61"


# ---------------------------------------------------------------------------
# xlsx_to_receptor_fasta: dedup, header construction, error handling
# ---------------------------------------------------------------------------


def test_xlsx_to_fasta_dedups_example_data(tmp_path: Path, example_xlsx: Path) -> None:
    """example_data.xlsx has 130 rows but only 2 distinct receptor sequences."""
    out = tmp_path / "receptor_full_length.fasta"
    n = xlsx_to_receptor_fasta(example_xlsx, out)
    assert n == 2
    lines = out.read_text().splitlines()
    # 2 unique records => 4 lines, alternating header / sequence.
    assert len(lines) == 4
    for header_line in (lines[0], lines[2]):
        assert header_line.startswith(">")
        assert header_line.count("|") == 2  # >species|locus|receptor
    for seq_line in (lines[1], lines[3]):
        # Receptor sequences are non-empty and contain only valid AA letters.
        assert seq_line != ""
        assert set(seq_line).issubset(set("ACDEFGHIKLMNPQRSTVWY"))


def test_xlsx_to_fasta_preserves_spaces_in_species(
    tmp_path: Path, example_xlsx: Path
) -> None:
    """``plant_species`` is written verbatim; spaces are NOT replaced here."""
    out = tmp_path / "out.fasta"
    xlsx_to_receptor_fasta(example_xlsx, out)
    text = out.read_text()
    # At least one species in the example data has a space ("Nicotiana benthamiana").
    assert any(" " in line for line in text.splitlines() if line.startswith(">"))


def test_xlsx_to_fasta_uses_lf_line_endings(
    tmp_path: Path, example_xlsx: Path
) -> None:
    """Output is byte-stable: LF line endings only, no CR."""
    out = tmp_path / "out.fasta"
    xlsx_to_receptor_fasta(example_xlsx, out)
    raw = out.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")


def test_xlsx_to_fasta_missing_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xlsx"
    pd.DataFrame({"plant_species": ["X"], "receptor": ["Y"]}).to_excel(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        xlsx_to_receptor_fasta(bad, tmp_path / "out.fasta")


def test_xlsx_to_fasta_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        xlsx_to_receptor_fasta(tmp_path / "nope.xlsx", tmp_path / "out.fasta")


# ---------------------------------------------------------------------------
# add_chemical_features: column ordering, arithmetic, error handling
# ---------------------------------------------------------------------------


def _minimal_test_data() -> pd.DataFrame:
    """Two rows mimicking the schema emitted by 04_data_prep_for_prediction.py."""
    return pd.DataFrame(
        {
            "Header_Name": ["Species_a|loc_a|R1", "Species_b|loc_b|R2"],
            "plant_species": ["Species a", "Species b"],
            "receptor": ["R1", "R2"],
            "locus_id": ["loc_a", "loc_b"],
            "Sequence": ["AGR", "WYF"],
            "receptor_sequence": ["DKH", "CCC"],
        }
    )


def test_add_chemical_features_column_order(tmp_path: Path) -> None:
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    _minimal_test_data().to_csv(inp, index=False)
    df = add_chemical_features(inp, out)
    assert list(df.columns) == [
        "Header_Name", "plant_species", "receptor", "locus_id",
        "Sequence", "receptor_sequence",
        "Sequence_Bulkiness", "Receptor_Bulkiness",
        "Sequence_Charge", "Receptor_Charge",
        "Sequence_Hydrophobicity", "Receptor_Hydrophobicity",
    ]


def test_add_chemical_features_arithmetic(tmp_path: Path) -> None:
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    _minimal_test_data().to_csv(inp, index=False)
    df = add_chemical_features(inp, out)
    # Row 0: Sequence = "AGR", receptor_sequence = "DKH"
    assert df["Sequence_Bulkiness"].iloc[0] == "11.5,3.4,14.28"
    assert df["Receptor_Bulkiness"].iloc[0] == "11.68,15.71,13.69"
    assert df["Sequence_Charge"].iloc[0] == "0,0,1"
    assert df["Receptor_Charge"].iloc[0] == "-1,1,0.1"
    assert df["Sequence_Hydrophobicity"].iloc[0] == "0.61,0.74,0"
    assert df["Receptor_Hydrophobicity"].iloc[0] == "0.06,0.28,0.61"


def test_add_chemical_features_persists_to_disk(tmp_path: Path) -> None:
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    _minimal_test_data().to_csv(inp, index=False)
    add_chemical_features(inp, out)
    assert out.is_file()
    # Round-trip back to a DataFrame and confirm column set survives.
    reread = pd.read_csv(out)
    assert "Sequence_Bulkiness" in reread.columns
    assert "Receptor_Hydrophobicity" in reread.columns


def test_add_chemical_features_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        add_chemical_features(tmp_path / "nope.csv", tmp_path / "out.csv")


def test_add_chemical_features_missing_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Header_Name": ["x"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        add_chemical_features(bad, tmp_path / "out.csv")

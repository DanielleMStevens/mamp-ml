"""Tests for :func:`mamp_ml.preprocess.build_lrr_domain_fasta` (checkpoint 4b).

Two layers of coverage:

1. Unit tests for the small helpers (PDB-stem key normalisation, FASTA
   parser, LRR-annotation-row parser) and a handful of synthetic
   end-to-end calls. These run on every machine in milliseconds.

2. A byte-identical golden test that feeds the function the same
   ``lrr_annotation_results.txt`` and the same ``receptor_full_length.fasta``
   that the legacy pipeline saw (the FASTA is regenerated on the fly from
   ``example_data.xlsx`` via :func:`xlsx_to_receptor_fasta` to keep the
   suite self-contained), then SHA-1-compares the output against
   ``tests/fixtures/golden/lrr_domain_sequences.fasta``.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pytest

from mamp_ml.preprocess import (
    _normalize_pdb_lookup_key,
    _read_fasta_records,
    _read_lrr_annotation_results,
    build_lrr_domain_fasta,
    xlsx_to_receptor_fasta,
)


# ---------------------------------------------------------------------------
# _normalize_pdb_lookup_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Solanum habrochates|scaffold11|CORE", "Solanum_habrochates_scaffold11_CORE"),
        ("Solanum_habrochates_scaffold11_CORE", "Solanum_habrochates_scaffold11_CORE"),
        ("species name|loc|R", "species_name_loc_R"),
        ("", ""),
        ("|||", "___"),
        ("no pipes here", "no_pipes_here"),
    ],
)
def test_normalize_pdb_lookup_key(raw: str, expected: str) -> None:
    """Spaces and pipes both become underscores; idempotent on already-normalised."""
    assert _normalize_pdb_lookup_key(raw) == expected


def test_normalize_pdb_lookup_key_is_idempotent() -> None:
    """Applying twice gives the same answer as applying once."""
    for raw in ("a b|c", "a_b_c", "x", "x|y z"):
        once = _normalize_pdb_lookup_key(raw)
        assert _normalize_pdb_lookup_key(once) == once


# ---------------------------------------------------------------------------
# _read_fasta_records
# ---------------------------------------------------------------------------


def test_read_fasta_two_records(tmp_path: Path) -> None:
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">first\nABCDEFG\n>second|x|y\nWXYZ\n")
    recs = _read_fasta_records(fasta)
    assert recs == [("first", "ABCDEFG"), ("second|x|y", "WXYZ")]


def test_read_fasta_multiline_sequence(tmp_path: Path) -> None:
    """Multi-line sequences must be concatenated into one string."""
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">a\nAA\nBB\nCC\n>b\nDDD\n")
    recs = _read_fasta_records(fasta)
    assert recs == [("a", "AABBCC"), ("b", "DDD")]


def test_read_fasta_tolerates_blank_lines_and_crlf(tmp_path: Path) -> None:
    """Blank lines and Windows-style line endings should be silently consumed."""
    fasta = tmp_path / "in.fasta"
    fasta.write_bytes(b">a\r\nABCD\r\n\r\n>b\r\nXYZ\r\n")
    recs = _read_fasta_records(fasta)
    assert recs == [("a", "ABCD"), ("b", "XYZ")]


def test_read_fasta_empty_file(tmp_path: Path) -> None:
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    assert _read_fasta_records(fasta) == []


# ---------------------------------------------------------------------------
# _read_lrr_annotation_results
# ---------------------------------------------------------------------------


def _annotation_header() -> str:
    return (
        "PDB_Filename\tRegion_Number\tStart_Position\tEnd_Position\t"
        "Sequence_Length\tFull_Sequence_Length\tTotal_LRR_Regions\tSequence\n"
    )


def test_read_lrr_annotation_results_strips_pdb_suffix(tmp_path: Path) -> None:
    p = tmp_path / "lrr.txt"
    p.write_text(_annotation_header() + "Foo.pdb\t1\t10\t30\t20\t100\t1\tABCDEF\n")
    rows = _read_lrr_annotation_results(p)
    assert rows == [("Foo", "ABCDEF")]


def test_read_lrr_annotation_results_keeps_stem_when_no_pdb_extension(
    tmp_path: Path,
) -> None:
    p = tmp_path / "lrr.txt"
    p.write_text(_annotation_header() + "Bar\t1\t10\t30\t20\t100\t1\tQRST\n")
    rows = _read_lrr_annotation_results(p)
    assert rows == [("Bar", "QRST")]


def test_read_lrr_annotation_results_rejects_unexpected_header(
    tmp_path: Path,
) -> None:
    p = tmp_path / "lrr.txt"
    p.write_text("Wrong\tHeader\tHere\n")
    with pytest.raises(ValueError, match="Unexpected header"):
        _read_lrr_annotation_results(p)


def test_read_lrr_annotation_results_warns_on_short_row(tmp_path: Path) -> None:
    """A row with fewer than 8 columns must warn and skip — not crash."""
    p = tmp_path / "lrr.txt"
    p.write_text(_annotation_header() + "Foo.pdb\t1\t10\t30\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = _read_lrr_annotation_results(p)
    assert rows == []
    assert any("malformed" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# build_lrr_domain_fasta — synthetic end-to-end
# ---------------------------------------------------------------------------


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    """Helper: write a list of (header, sequence) tuples as a FASTA file."""
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        for header, seq in records:
            fh.write(f">{header}\n{seq}\n")


def test_build_lrr_domain_fasta_writes_expected_header_format(
    tmp_path: Path,
) -> None:
    receptor_fasta = tmp_path / "receptors.fasta"
    _write_fasta(
        receptor_fasta,
        [
            ("Species one|loc1|R1", "AAAAAAAAA"),
            ("Species two|loc2|R2", "BBBBBBBBB"),
        ],
    )
    annotation = tmp_path / "lrr.txt"
    annotation.write_text(
        _annotation_header()
        + "Species_one_loc1_R1.pdb\t1\t1\t9\t9\t9\t1\tAAA\n"
        + "Species_two_loc2_R2.pdb\t1\t1\t9\t9\t9\t1\tBBB\n"
    )
    out = tmp_path / "lrr_out.fasta"

    n = build_lrr_domain_fasta(annotation, receptor_fasta, out)

    assert n == 2
    contents = out.read_text()
    # Spaces in plant_species become underscores; pipes survive; |LRR_domain
    # gets appended.
    assert ">Species_one|loc1|R1|LRR_domain\nAAA\n" in contents
    assert ">Species_two|loc2|R2|LRR_domain\nBBB\n" in contents


def test_build_lrr_domain_fasta_uses_lf_line_endings(tmp_path: Path) -> None:
    receptor_fasta = tmp_path / "receptors.fasta"
    _write_fasta(receptor_fasta, [("Sp x|loc|R", "AAA")])
    annotation = tmp_path / "lrr.txt"
    annotation.write_text(
        _annotation_header() + "Sp_x_loc_R.pdb\t1\t1\t3\t3\t3\t1\tAAA\n"
    )
    out = tmp_path / "out.fasta"
    build_lrr_domain_fasta(annotation, receptor_fasta, out)
    raw = out.read_bytes()
    assert b"\r" not in raw, "must not write CR; LF-only for byte stability"
    assert raw.endswith(b"\n")


def test_build_lrr_domain_fasta_skips_unmatched_pdb_with_warning(
    tmp_path: Path,
) -> None:
    """When a PDB stem has no matching receptor header, skip + warn cleanly."""
    receptor_fasta = tmp_path / "receptors.fasta"
    _write_fasta(receptor_fasta, [("Real species|loc|R", "AAAA")])
    annotation = tmp_path / "lrr.txt"
    annotation.write_text(
        _annotation_header()
        + "Real_species_loc_R.pdb\t1\t1\t4\t4\t4\t1\tAAAA\n"
        + "Ghost_species_loc_R.pdb\t1\t1\t4\t4\t4\t1\tBBBB\n"
    )
    out = tmp_path / "out.fasta"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = build_lrr_domain_fasta(annotation, receptor_fasta, out)
    assert n == 1
    assert "Ghost_species_loc_R" in str(caught[0].message)
    text = out.read_text()
    assert "Real_species|loc|R|LRR_domain" in text
    assert "Ghost" not in text


def test_build_lrr_domain_fasta_preserves_input_order(tmp_path: Path) -> None:
    """Output order follows the annotation file order, not the FASTA order."""
    receptor_fasta = tmp_path / "receptors.fasta"
    _write_fasta(
        receptor_fasta,
        [
            ("Sp a|loc|R", "AAA"),
            ("Sp b|loc|R", "BBB"),
            ("Sp c|loc|R", "CCC"),
        ],
    )
    annotation = tmp_path / "lrr.txt"
    annotation.write_text(
        _annotation_header()
        + "Sp_c_loc_R.pdb\t1\t1\t3\t3\t3\t1\tCCC\n"
        + "Sp_a_loc_R.pdb\t1\t1\t3\t3\t3\t1\tAAA\n"
    )
    out = tmp_path / "out.fasta"
    build_lrr_domain_fasta(annotation, receptor_fasta, out)
    text = out.read_text()
    # Sp_c must appear before Sp_a — annotation file order, not FASTA order.
    assert text.index("Sp_c") < text.index("Sp_a")
    assert "Sp_b" not in text  # not in annotation -> not in output


def test_build_lrr_domain_fasta_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_lrr_domain_fasta(
            tmp_path / "no_lrr.txt",
            tmp_path / "no_fasta.fasta",
            tmp_path / "out.fasta",
        )

    # LRR present, FASTA missing
    annotation = tmp_path / "lrr.txt"
    annotation.write_text(_annotation_header())
    with pytest.raises(FileNotFoundError):
        build_lrr_domain_fasta(
            annotation,
            tmp_path / "no_fasta.fasta",
            tmp_path / "out.fasta",
        )


# ---------------------------------------------------------------------------
# Golden equivalence: byte-identical to the legacy pipeline output
# ---------------------------------------------------------------------------


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def test_build_lrr_domain_fasta_matches_legacy_golden(
    tmp_path: Path,
    repo_root: Path,
    example_xlsx: Path,
) -> None:
    """The Python output must SHA-1-equal the legacy pipeline's output.

    We regenerate the receptor FASTA on the fly from ``example_data.xlsx``
    via :func:`xlsx_to_receptor_fasta` (already golden-tested against the R
    script in checkpoint 3), then feed it together with the committed
    ``lrr_annotation_results.txt`` golden into the new function. The result
    must be byte-identical to the committed ``lrr_domain_sequences.fasta``
    that the legacy ``scripts/03_parse_lrr_annotation.py`` produced.
    """
    golden_lrr_results = (
        repo_root / "tests" / "fixtures" / "golden" / "lrr_annotation_results.txt"
    )
    golden_lrr_domain = (
        repo_root / "tests" / "fixtures" / "golden" / "lrr_domain_sequences.fasta"
    )
    assert golden_lrr_results.is_file(), "golden lrr_annotation_results.txt missing"
    assert golden_lrr_domain.is_file(), "golden lrr_domain_sequences.fasta missing"

    receptor_fasta = tmp_path / "receptor_full_length.fasta"
    xlsx_to_receptor_fasta(example_xlsx, receptor_fasta)

    new_output = tmp_path / "lrr_domain_sequences.fasta"
    n = build_lrr_domain_fasta(golden_lrr_results, receptor_fasta, new_output)
    assert n == 1, "Solanum is the only receptor with a PDB in the goldens"

    assert _sha1(new_output) == _sha1(golden_lrr_domain), (
        f"SHA-1 mismatch.\n"
        f"  new:    {_sha1(new_output)}\n"
        f"  golden: {_sha1(golden_lrr_domain)}\n"
        f"new bytes ({new_output.stat().st_size}): {new_output.read_bytes()!r}\n"
        f"golden bytes ({golden_lrr_domain.stat().st_size}): {golden_lrr_domain.read_bytes()!r}"
    )

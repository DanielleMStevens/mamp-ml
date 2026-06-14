"""Golden-output equivalence tests: legacy R scripts vs new Python.

This module compares the output of the original R preprocessing scripts to
the output of their Python replacements, on identical inputs:

* ``scripts/01_convert_sheet_to_fasta.R`` ↔ :func:`mamp_ml.preprocess.xlsx_to_receptor_fasta`
* ``scripts/05_chemical_conversion.R``   ↔ :func:`mamp_ml.preprocess.add_chemical_features`

The R scripts are invoked through ``Rscript`` in an isolated temporary working
directory; the output files are then compared byte-for-byte (FASTA) or
DataFrame-for-DataFrame (CSV). If ``Rscript`` is not on PATH, or the required
R packages (``readxl``, ``tidyverse``, ``Peptides``) cannot be loaded, the
tests are skipped — the suite stays portable on machines without R.

These tests exist specifically to gate the eventual removal of the R scripts:
they must continue to pass for as long as the R scripts remain part of the
distribution, providing a hard guarantee that the Python and R pipelines
produce indistinguishable results.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from mamp_ml.preprocess import add_chemical_features, xlsx_to_receptor_fasta


# ---------------------------------------------------------------------------
# Probe helpers — decide whether the R-driven tests can run on this machine
# ---------------------------------------------------------------------------


def _rscript_runs(packages: tuple[str, ...]) -> bool:
    """Return True if ``Rscript`` is on PATH and can load *all* of ``packages``.

    A failure to load (missing R install, missing package, version skew) is
    treated as "skip cleanly" rather than "test failure" so the suite stays
    runnable on machines that don't have R configured.
    """
    if shutil.which("Rscript") is None:
        return False
    library_calls = "; ".join(f"library({pkg})" for pkg in packages)
    probe = f"suppressMessages({{{library_calls}}})"
    try:
        result = subprocess.run(
            ["Rscript", "-e", probe],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


_HAS_R_FASTA = _rscript_runs(("readxl", "tidyverse"))
_HAS_R_CHEM = _rscript_runs(("Peptides", "tidyverse"))


# ---------------------------------------------------------------------------
# Test 1 — FASTA byte-equivalence
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_R_FASTA,
    reason="Rscript or readxl+tidyverse not available — cannot diff against R",
)
def test_xlsx_to_fasta_byte_identical_to_R(
    tmp_path: Path,
    repo_root: Path,
    example_xlsx: Path,
) -> None:
    """Running the legacy R script and the new Python on the same xlsx must
    produce byte-identical FASTA output."""
    # Run the R script in an isolated working directory so it doesn't touch
    # the developer's intermediate_files/ tree.
    work = tmp_path / "r_workdir"
    (work / "intermediate_files").mkdir(parents=True)
    r_script = repo_root / "scripts" / "01_convert_sheet_to_fasta.R"

    completed = subprocess.run(
        ["Rscript", str(r_script), str(example_xlsx)],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        pytest.fail(
            "R fasta script failed:\n"
            f"  stdout: {completed.stdout}\n"
            f"  stderr: {completed.stderr}"
        )
    r_out = work / "intermediate_files" / "receptor_full_length.fasta"
    assert r_out.is_file(), "R script did not produce the expected FASTA file"
    r_bytes = r_out.read_bytes()

    # Run the Python implementation alongside.
    py_out = tmp_path / "python.fasta"
    xlsx_to_receptor_fasta(example_xlsx, py_out)
    py_bytes = py_out.read_bytes()

    # Normalize CR (in case the R install on a Windows host wrote CRLF) so the
    # check stays meaningful across operating systems while still catching any
    # genuine content divergence.
    r_normalized = r_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    py_normalized = py_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

    if r_normalized != py_normalized:
        # Surface a useful diff on failure.
        r_lines = r_normalized.decode("utf-8").splitlines()
        py_lines = py_normalized.decode("utf-8").splitlines()
        diff = []
        for i, (rline, pyline) in enumerate(zip(r_lines, py_lines)):
            if rline != pyline:
                diff.append(f"  line {i}: R={rline!r}  PY={pyline!r}")
        if len(r_lines) != len(py_lines):
            diff.append(f"  line counts differ: R={len(r_lines)}, PY={len(py_lines)}")
        pytest.fail("FASTA outputs diverge:\n" + "\n".join(diff))


# ---------------------------------------------------------------------------
# Test 2 — chemical-features DataFrame equivalence
# ---------------------------------------------------------------------------


def _write_synthetic_test_data_csv(csv_path: Path) -> None:
    """Write a small but representative test_data.csv covering sign, decimal,
    and trailing-zero formatting edge cases."""
    df = pd.DataFrame(
        {
            "Header_Name": [
                "Species_a|loc_a|R1",
                "Species_b|loc_b|R2",
                "Species_c|loc_c|R3",
            ],
            "plant_species": ["Species a", "Species b", "Species c"],
            "receptor": ["R1", "R2", "R3"],
            "locus_id": ["loc_a", "loc_b", "loc_c"],
            # Mix all 20 standard amino acids across the rows so every entry of
            # every property table gets exercised.
            "Sequence": ["AGRDKHCEQNVLI", "WYFPSTM", "ACDEFGHIKLMNPQRSTVWY"],
            "receptor_sequence": [
                "MNPQRSTVWY",
                "ACDEFGHIKL",
                "AGAGAGRDKHCEQNVLI",
            ],
        }
    )
    df.to_csv(csv_path, index=False)


@pytest.mark.skipif(
    not _HAS_R_CHEM,
    reason="Rscript or Peptides+tidyverse not available — cannot diff against R",
)
def test_chemical_features_dataframe_identical_to_R(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    """The Python add_chemical_features output must parse back to the exact
    same DataFrame as the legacy R script's output."""
    work = tmp_path / "r_workdir"
    intermediate = work / "intermediate_files"
    intermediate.mkdir(parents=True)
    test_data_csv = intermediate / "test_data.csv"
    _write_synthetic_test_data_csv(test_data_csv)

    # The R script expects to be run from a CWD that has intermediate_files/
    # as a sibling because of its hard-coded relative path.
    r_script = repo_root / "scripts" / "05_chemical_conversion.R"
    completed = subprocess.run(
        ["Rscript", str(r_script), "test_data.csv"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        pytest.fail(
            "R chemical-conversion script failed:\n"
            f"  stdout: {completed.stdout}\n"
            f"  stderr: {completed.stderr}"
        )
    r_out = intermediate / "ready_test_data.csv"
    assert r_out.is_file(), "R script did not produce ready_test_data.csv"

    py_out = tmp_path / "ready_python.csv"
    add_chemical_features(test_data_csv, py_out)
    assert py_out.is_file()

    # Compare via parsed DataFrames: pandas will read either quoting style
    # transparently, but the parsed cell values must be identical.
    r_df = pd.read_csv(r_out)
    py_df = pd.read_csv(py_out)

    # Sanity: same column set, same ordering.
    assert list(r_df.columns) == list(py_df.columns), (
        f"R columns: {list(r_df.columns)}\nPY columns: {list(py_df.columns)}"
    )

    # The feature columns store comma-separated number strings; the row-level
    # equality test catches divergence at the per-residue value level.
    pd.testing.assert_frame_equal(r_df, py_df, check_like=False)

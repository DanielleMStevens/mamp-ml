"""mamp_ml.preprocess — single-language data preparation pipeline.

This module is the consolidated Python replacement for the legacy preprocessing
scripts that originally drove the MAMP-ml pipeline:

* ``scripts/01_convert_sheet_to_fasta.R``    (this checkpoint)
* ``scripts/03_parse_lrr_annotation.py``     (added in checkpoint 4)
* ``scripts/04_data_prep_for_prediction.py`` (added in checkpoint 4)
* ``scripts/05_chemical_conversion.R``       (this checkpoint)

All functions are *bit-for-bit faithful* to the legacy scripts they replace —
either byte-identical output (FASTA, fixed text format) or DataFrame-identical
output (CSV, parsed back through ``pandas``). Equivalence is enforced by the
golden tests in ``tests/test_golden_r_outputs.py``, which invoke the original
R scripts via ``Rscript`` and diff the result against the new Python output.

The module is intentionally a single file (per the consolidation requirement)
with clearly demarcated sections. Each section's public API stays at the top
level so callers see one flat namespace:

    >>> from mamp_ml.preprocess import xlsx_to_receptor_fasta, add_chemical_features

Stability notes
---------------
* Numeric formatting matches R's ``as.character`` / ``paste`` semantics so the
  per-residue feature strings are byte-identical to those emitted by the R
  pipeline.
* FASTA output uses unconditional LF (``\\n``) line endings to remain
  byte-stable across operating systems.
* Non-standard amino acids are silently skipped during chemical-feature
  generation, matching R's behaviour where ``sapply`` returns ``NULL`` and
  ``paste(..., collapse=",")`` drops those NULLs.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]

__all__ = [
    "REQUIRED_INPUT_COLUMNS",
    "BULKINESS",
    "CHARGE",
    "HYDROPHOBICITY",
    "read_receptor_xlsx",
    "xlsx_to_receptor_fasta",
    "build_lrr_domain_fasta",
    "sequence_to_property",
    "sequence_to_bulkiness",
    "sequence_to_charge",
    "sequence_to_hydrophobicity",
    "add_chemical_features",
]

# =============================================================================
# Section 1 :: Input spreadsheet  ->  FASTA of unique receptor sequences
# Replaces scripts/01_convert_sheet_to_fasta.R
# =============================================================================

#: Columns that the canonical MAMP input spreadsheet must contain. Matches the
#: column names referenced by the legacy R script verbatim.
REQUIRED_INPUT_COLUMNS = (
    "plant_species",
    "receptor",
    "locus_id",
    "receptor_sequence",
    "ligand_sequence",
)


def read_receptor_xlsx(
    xlsx_path: PathLike,
    sheet_name: str = "Sheet1",
) -> pd.DataFrame:
    """Load and validate the canonical MAMP input spreadsheet.

    Parameters
    ----------
    xlsx_path
        Path to the input ``.xlsx`` file.
    sheet_name
        Sheet to read from the workbook (default ``"Sheet1"``, matching the
        legacy R script).

    Returns
    -------
    pandas.DataFrame
        The loaded spreadsheet with rows in their original order.

    Raises
    ------
    FileNotFoundError
        If ``xlsx_path`` does not exist.
    ValueError
        If any column listed in :data:`REQUIRED_INPUT_COLUMNS` is missing.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Input spreadsheet not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input spreadsheet {xlsx_path} is missing required columns: {missing}"
        )
    return df


def xlsx_to_receptor_fasta(
    xlsx_path: PathLike,
    output_fasta: PathLike,
    *,
    sheet_name: str = "Sheet1",
) -> int:
    """Convert the MAMP input spreadsheet into a FASTA of unique receptors.

    Python replacement for ``scripts/01_convert_sheet_to_fasta.R``. For each
    row in the spreadsheet a FASTA record is constructed with header
    ``>{plant_species}|{locus_id}|{receptor}`` and sequence
    ``{receptor_sequence}``. Records are deduplicated by sequence, keeping the
    first occurrence, exactly matching ``dplyr::distinct(Sequence, .keep_all=TRUE)``
    from the legacy script.

    Parameters
    ----------
    xlsx_path
        Path to the input ``.xlsx`` file.
    output_fasta
        Path where the FASTA file will be written. Parent directories are
        created on demand.
    sheet_name
        Sheet to read from the workbook (default ``"Sheet1"``).

    Returns
    -------
    int
        The number of unique FASTA records written.

    Notes
    -----
    * The ``plant_species`` value is written verbatim in the FASTA header
      (spaces preserved). Downstream tooling normalises spaces and pipes
      separately when matching against PDB filenames.
    * Output uses unconditional LF line endings so the file is byte-stable
      across operating systems.
    """
    df = read_receptor_xlsx(xlsx_path, sheet_name=sheet_name)
    # dplyr's distinct(Sequence, .keep_all=TRUE) keeps the first occurrence of
    # each unique value in the "Sequence" column while retaining every other
    # column from that first row. pandas' drop_duplicates with keep="first" has
    # the same semantics.
    unique = df.drop_duplicates(subset="receptor_sequence", keep="first")

    output_fasta = Path(output_fasta)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    # `newline="\n"` disables Windows-style line-ending translation so the
    # written bytes match the legacy R output byte-for-byte regardless of host.
    with open(output_fasta, "w", newline="\n", encoding="utf-8") as fh:
        for _, row in unique.iterrows():
            header = f">{row['plant_species']}|{row['locus_id']}|{row['receptor']}"
            fh.write(f"{header}\n{row['receptor_sequence']}\n")
            n_written += 1
    return n_written


# =============================================================================
# Section 2 :: Per-residue physico-chemical features
# Replaces scripts/05_chemical_conversion.R
# =============================================================================

#: Per-amino-acid bulkiness values. Copied verbatim from
#: ``scripts/05_chemical_conversion.R`` (line 16). Stored as a plain ``dict``
#: so the per-residue lookup remains a single hash op.
BULKINESS: Dict[str, float] = {
    "A": 11.50, "R": 14.28, "N": 12.82, "D": 11.68, "C": 13.46,
    "Q": 14.45, "E": 13.57, "G": 3.40,  "H": 13.69, "I": 21.40,
    "L": 21.40, "K": 15.71, "M": 16.25, "F": 19.80, "P": 17.43,
    "S": 9.47,  "T": 15.77, "W": 21.67, "Y": 18.03, "V": 21.57,
}

#: Per-amino-acid net side-chain charge at physiological pH. Copied verbatim
#: from ``scripts/05_chemical_conversion.R`` (line 26).
CHARGE: Dict[str, float] = {
    "A": 0,    "R": 1,    "N": 0,    "D": -1,   "C": 0,
    "Q": 0,    "E": -1,   "G": 0,    "H": 0.1,  "I": 0,
    "L": 0,    "K": 1,    "M": 0,    "F": 0,    "P": 0,
    "S": 0,    "T": 0,    "W": 0,    "Y": 0,    "V": 0,
}

#: Per-amino-acid hydrophobicity. Copied verbatim from
#: ``scripts/05_chemical_conversion.R`` (line 36).
HYDROPHOBICITY: Dict[str, float] = {
    "A": 0.61, "R": 0.00, "N": 0.06, "D": 0.06, "C": 1.07,
    "Q": 0.00, "E": 0.01, "G": 0.74, "H": 0.61, "I": 2.22,
    "L": 1.53, "K": 0.28, "M": 1.18, "F": 2.02, "P": 1.95,
    "S": 0.46, "T": 0.45, "W": 2.65, "Y": 1.88, "V": 1.32,
}

# Column order written by 05_chemical_conversion.R (line 58).
_CHEMICAL_OUTPUT_COLUMNS = [
    "Header_Name", "plant_species", "receptor", "locus_id",
    "Sequence", "receptor_sequence",
    "Sequence_Bulkiness", "Receptor_Bulkiness",
    "Sequence_Charge", "Receptor_Charge",
    "Sequence_Hydrophobicity", "Receptor_Hydrophobicity",
]

# Required input columns for add_chemical_features() — must match the columns
# emitted by scripts/04_data_prep_for_prediction.py.
_CHEMICAL_INPUT_COLUMNS = (
    "Header_Name", "plant_species", "receptor", "locus_id",
    "Sequence", "receptor_sequence",
)


def _r_format(value: float) -> str:
    """Format a number the way R's ``paste()`` / ``as.character()`` would.

    R stores all entries of a numeric vector as doubles, but its default
    string conversion strips trailing zeros from integer-valued doubles
    (``0`` not ``"0.0"``, ``-1`` not ``"-1.0"``). Python's ``str(0.0)``
    instead yields ``"0.0"``. This helper bridges the gap so the
    comma-separated feature strings are byte-identical to R's output.

    Examples
    --------
    >>> _r_format(0)
    '0'
    >>> _r_format(-1.0)
    '-1'
    >>> _r_format(11.50)
    '11.5'
    >>> _r_format(13.46)
    '13.46'
    >>> _r_format(0.1)
    '0.1'
    """
    int_value = int(value)
    if value == int_value:
        return str(int_value)
    return format(value, "g")


def sequence_to_property(seq: str, table: Mapping[str, float]) -> str:
    """Render a protein sequence as comma-separated per-residue property values.

    Non-standard amino acids (anything not present in ``table``) are silently
    skipped, matching the legacy R behaviour where ``sapply`` returns
    ``NULL`` for missing entries and ``paste(..., collapse=",")`` drops those
    NULLs from the joined output.

    Parameters
    ----------
    seq
        Amino-acid sequence in one-letter code (uppercase).
    table
        Mapping from amino-acid letter to numeric property value.

    Returns
    -------
    str
        Comma-separated property values, formatted to match R's
        ``as.character`` output.
    """
    return ",".join(_r_format(table[aa]) for aa in seq if aa in table)


def sequence_to_bulkiness(seq: str) -> str:
    """Comma-separated bulkiness values for each residue of ``seq``."""
    return sequence_to_property(seq, BULKINESS)


def sequence_to_charge(seq: str) -> str:
    """Comma-separated side-chain charge values for each residue of ``seq``."""
    return sequence_to_property(seq, CHARGE)


def sequence_to_hydrophobicity(seq: str) -> str:
    """Comma-separated hydrophobicity values for each residue of ``seq``."""
    return sequence_to_property(seq, HYDROPHOBICITY)


def add_chemical_features(
    input_csv: PathLike,
    output_csv: PathLike,
) -> pd.DataFrame:
    """Annotate the preprocessed test-data CSV with chemical-feature columns.

    Python replacement for ``scripts/05_chemical_conversion.R``. Reads the CSV
    produced by ``04_data_prep_for_prediction.py``, reorders the columns to
    place ``Sequence`` before ``receptor_sequence`` (matching the legacy R
    script), and appends six comma-separated columns of per-residue
    physico-chemical features for both the ligand (``Sequence``) and the
    receptor (``receptor_sequence``).

    Parameters
    ----------
    input_csv
        Path to the upstream ``test_data.csv``. Must contain the columns
        ``Header_Name``, ``plant_species``, ``receptor``, ``locus_id``,
        ``Sequence``, and ``receptor_sequence``.
    output_csv
        Where to write the augmented CSV. Parent directories are created on
        demand.

    Returns
    -------
    pandas.DataFrame
        The fully-annotated DataFrame written to disk.

    Raises
    ------
    FileNotFoundError
        If ``input_csv`` does not exist.
    ValueError
        If any required column is missing from the input.

    Notes
    -----
    Numeric formatting follows R's ``as.character`` semantics (see
    :func:`_r_format`), so each feature value is written without trailing
    zeros (``0`` not ``0.0``, ``11.5`` not ``11.500``).
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    missing = [c for c in _CHEMICAL_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV {input_csv} is missing required columns: {missing}"
        )

    # Reorder to put `Sequence` before `receptor_sequence` (matches the
    # column selection on line 11 of the legacy R script).
    df = df[list(_CHEMICAL_INPUT_COLUMNS)].copy()

    # Append the six chemical-feature columns. ``Series.map`` is a per-row
    # Python call, but the inner ``sequence_to_property`` body is tight enough
    # that for typical input sizes (~1000 AAs x ~1000 rows) total wall time
    # stays well under a second; further vectorisation is unnecessary.
    df["Sequence_Bulkiness"] = df["Sequence"].map(sequence_to_bulkiness)
    df["Receptor_Bulkiness"] = df["receptor_sequence"].map(sequence_to_bulkiness)
    df["Sequence_Charge"] = df["Sequence"].map(sequence_to_charge)
    df["Receptor_Charge"] = df["receptor_sequence"].map(sequence_to_charge)
    df["Sequence_Hydrophobicity"] = df["Sequence"].map(sequence_to_hydrophobicity)
    df["Receptor_Hydrophobicity"] = df["receptor_sequence"].map(sequence_to_hydrophobicity)

    df = df[_CHEMICAL_OUTPUT_COLUMNS]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


# =============================================================================
# Section 3 :: LRR-annotation results  +  receptor FASTA  ->  LRR-domain FASTA
# Replaces scripts/03_parse_lrr_annotation.py
# =============================================================================
#
# After the structure-analysis stage emits ``lrr_annotation_results.txt`` (one
# row per detected LRR region, keyed on the source PDB filename), this section
# joins those rows back to the receptor full-length FASTA so the LRR-domain
# sequence can be re-headered with the canonical ``species|locus|receptor``
# identifier and consumed by the downstream data-prep step.
#
# The matching is necessary because the PDB filename uses underscores in
# place of both spaces and pipes (it has to: many filesystems disallow ``|``
# and tools may split on whitespace), while the FASTA header is canonical and
# uses both characters. So the join key is the *common normalised form*
# (everything replaced with underscores), built fresh from both sides.


def _normalize_pdb_lookup_key(text: str) -> str:
    """Reduce a receptor identifier to its PDB-filename-stem normalisation.

    The receptor full-length FASTA uses headers like
    ``Solanum habrochates|scaffold11|CORE`` (space + pipes); the PDB files
    written by the structure-analysis stage replace both separators with
    underscores, yielding ``Solanum_habrochates_scaffold11_CORE``. This
    helper applies the same normalisation to either side so a header can
    be looked up by its PDB stem.

    Parameters
    ----------
    text
        Either a FASTA header (without the leading ``>``) or a PDB
        filename stem (without the trailing ``.pdb``).

    Returns
    -------
    str
        Underscore-joined normalised form.
    """
    return text.replace(" ", "_").replace("|", "_")


def _read_fasta_records(fasta_path: Path) -> List[Tuple[str, str]]:
    """Parse a FASTA file into a list of ``(header, sequence)`` tuples.

    Headers are returned WITHOUT the leading ``>``. Multi-line sequences
    are concatenated. Blank lines and Windows-style ``\\r`` line endings
    are tolerated. The original record order is preserved.

    Parameters
    ----------
    fasta_path
        Path to a FASTA file on disk.

    Returns
    -------
    list of (str, str)
        ``(header, sequence)`` for each record in the file.
    """
    records: List[Tuple[str, str]] = []
    header: Union[str, None] = None
    seq_chunks: List[str] = []
    with open(fasta_path, encoding="utf-8") as fh:
        for raw in fh:
            # Strip both LF and any CR so the parser is robust to CRLF inputs.
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    return records


# Column indices in the LRR-annotation TSV emitted by the structure-analysis
# stage (matches the header line written by scripts/02_alphafold_to_lrr_annotation.py
# and the upcoming mamp_ml.structure module).
_LRR_RESULT_PDB_FILENAME_COL = 0
_LRR_RESULT_SEQUENCE_COL = 7


def _read_lrr_annotation_results(
    results_path: Path,
) -> List[Tuple[str, str]]:
    """Parse the LRR-annotation TSV into ``(pdb_stem, lrr_sequence)`` rows.

    The PDB filename column carries the trailing ``.pdb`` extension; we
    strip it here so the caller can use the bare stem as a lookup key. The
    header row is validated to catch the case where this function is
    pointed at the wrong file by mistake.

    Parameters
    ----------
    results_path
        Path to the tab-separated annotation file (header row plus one row
        per detected LRR region).

    Returns
    -------
    list of (str, str)
        ``(pdb_filename_stem, lrr_sequence)`` per row, in file order.

    Raises
    ------
    ValueError
        If the header row is not the expected ``PDB_Filename\\t…`` line.
    """
    records: List[Tuple[str, str]] = []
    with open(results_path, encoding="utf-8") as fh:
        header_line = fh.readline()
        if not header_line.startswith("PDB_Filename"):
            raise ValueError(
                f"Unexpected header in LRR-annotation results {results_path}: "
                f"{header_line.rstrip()!r}"
            )
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) <= _LRR_RESULT_SEQUENCE_COL:
                warnings.warn(
                    f"Skipping malformed LRR-annotation row in {results_path}: "
                    f"{line!r}",
                    stacklevel=2,
                )
                continue
            pdb_filename = cols[_LRR_RESULT_PDB_FILENAME_COL]
            lrr_sequence = cols[_LRR_RESULT_SEQUENCE_COL]
            stem = (
                pdb_filename[:-4]
                if pdb_filename.endswith(".pdb")
                else pdb_filename
            )
            records.append((stem, lrr_sequence))
    return records


def build_lrr_domain_fasta(
    lrr_annotation_results: PathLike,
    receptor_full_length_fasta: PathLike,
    output_fasta: PathLike,
) -> int:
    """Join LRR-annotation rows with receptor headers into an LRR-domain FASTA.

    Python replacement for ``scripts/03_parse_lrr_annotation.py``. Reads the
    LRR-annotation TSV (which references receptors by their PDB filename stem)
    and matches each row to the canonical FASTA header from the receptor
    full-length file. Output headers preserve the original pipe-separated
    form, replace any spaces in ``plant_species`` with underscores so that
    whitespace-tokenising downstream tools see a single identifier, and
    append a ``|LRR_domain`` suffix to mark the sequence as a region extract.

    Records whose PDB stem cannot be matched to any receptor header are
    skipped with a warning. This is the common case when the structure stage
    failed to produce a PDB for one of the input receptors (e.g. a folding
    job was cancelled or the model crashed for one query).

    Parameters
    ----------
    lrr_annotation_results
        Path to the tab-separated LRR-annotation file
        (``intermediate_files/lrr_annotation_results.txt`` in the legacy
        layout). Column 0 holds the PDB filename, column 7 holds the LRR
        sequence; columns 1-6 are positional metadata and are ignored here.
    receptor_full_length_fasta
        Path to the receptor full-length FASTA produced by
        :func:`xlsx_to_receptor_fasta`. The headers in this file form the
        canonical identifiers re-attached to each LRR region in the output.
    output_fasta
        Where to write the LRR-domain FASTA. Parent directories are created
        on demand. The file uses unconditional LF (``\\n``) line endings.

    Returns
    -------
    int
        Number of records written to ``output_fasta``.

    Raises
    ------
    FileNotFoundError
        If either input path does not exist.
    ValueError
        If the LRR-annotation file has an unexpected header row.
    """
    lrr_annotation_results = Path(lrr_annotation_results)
    receptor_full_length_fasta = Path(receptor_full_length_fasta)
    output_fasta = Path(output_fasta)

    if not lrr_annotation_results.is_file():
        raise FileNotFoundError(
            f"LRR annotation file not found: {lrr_annotation_results}"
        )
    if not receptor_full_length_fasta.is_file():
        raise FileNotFoundError(
            f"Receptor FASTA not found: {receptor_full_length_fasta}"
        )

    # Build the PDB-stem -> (original_header, sequence) lookup. The receptor
    # FASTA carries headers in canonical form (with spaces and pipes); the
    # normalisation maps them to the same form used to derive PDB filenames,
    # so a PDB stem from the annotation file can find its parent record.
    fasta_records = _read_fasta_records(receptor_full_length_fasta)
    lookup: Dict[str, Tuple[str, str]] = {
        _normalize_pdb_lookup_key(header): (header, sequence)
        for header, sequence in fasta_records
    }

    annotation_rows = _read_lrr_annotation_results(lrr_annotation_results)

    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(output_fasta, "w", newline="\n", encoding="utf-8") as fh:
        for pdb_stem, lrr_sequence in annotation_rows:
            key = _normalize_pdb_lookup_key(pdb_stem)
            match = lookup.get(key)
            if match is None:
                # Surface as a runtime warning rather than a hard error so a
                # partial structure-stage run (e.g. one of two receptors
                # folded) still produces usable downstream data for the
                # receptors that succeeded.
                warnings.warn(
                    "No matching receptor header found for PDB "
                    f"'{pdb_stem}.pdb' (normalised key '{key}'); skipping.",
                    stacklevel=2,
                )
                continue
            original_header, _ = match
            # Replace only spaces (not pipes) so the header stays a single
            # whitespace-delimited token, then append the LRR-domain tag.
            output_header = original_header.replace(" ", "_") + "|LRR_domain"
            fh.write(f">{output_header}\n{lrr_sequence}\n")
            n_written += 1

    return n_written

"""ESMFold structure-prediction backend.

Runs HuggingFace's ``facebook/esmfold_v1`` (Meta's single-sequence folding
model, an ESM-2 derivative) over the receptor full-length FASTA produced by
:func:`mamp_ml.preprocess.xlsx_to_receptor_fasta`, writing one PDB per
receptor plus a ``log.txt`` in **ColabFold-compatible format** so the
downstream :mod:`mamp_ml.structure` stage requires no special handling.

Why ColabFold-compatible output
-------------------------------
The structure stage already parses ColabFold's ``log.txt`` (Query / model
score lines) and globs for PDBs matching
``{receptor}_unrelaxed_rank_001_alphafold2_ptm_model_N_seed_*.pdb``. Rather
than build a parallel parsing path for ESMFold, we emit the same line
schema and the same filename convention. The PDB content itself is
ESMFold's output; only the file *names* and the metadata *log* mimic
ColabFold's conventions.

Sequence length cap
-------------------
ESMFold has a hard maximum of 1024 residues (positional embedding limit).
Plant LRR receptors are frequently 1000-1500 AA, with the LRR ectodomain
near the N-terminus. We truncate to 1024 with a clear warning rather than
fail; this matches the user-confirmed convention that the first 1024
residues encode the LRR ectodomain.

Heavy deps
----------
This module's runtime imports of ``transformers`` and ``torch`` are
deliberately lazy, so importing ``mamp_ml.fold.esmfold`` itself is cheap.
Users install the heavy deps via ``pip install mamp-ml[esmfold]`` (which
adds ``accelerate``; transformers + torch are already core deps).
"""

from __future__ import annotations

import datetime
import re
import warnings
from pathlib import Path
from typing import List, Tuple, Union

PathLike = Union[str, Path]

__all__ = [
    "ESMFOLD_MAX_LENGTH",
    "fold_with_esmfold",
    "render_colabfold_compatible_log",
    "make_colabfold_compatible_pdb_filename",
    "normalize_receptor_name",
    "auto_pick_chunk_size",
]

#: Hard upper bound on input sequence length set by ESMFold's positional
#: embedding tables. Sequences longer than this are truncated to 1024 AAs
#: from the N-terminus before folding.
ESMFOLD_MAX_LENGTH: int = 1024

#: Descending ladder of trunk chunk sizes the OOM-backoff loop falls through
#: when a forward pass runs out of GPU memory. Smaller chunks lower the
#: triangular-attention activation peak at the cost of wall-clock; 1 is the
#: slowest-but-leanest setting ESMFold supports. We start from whatever chunk
#: size was chosen (auto-picked, user-supplied, or ``None`` = no chunking) and
#: only try strictly smaller values from this ladder.
_OOM_FALLBACK_LADDER: Tuple[int, ...] = (128, 64, 32, 16, 8, 4, 2, 1)


def _now_str() -> str:
    """Timestamp prefix matching the format ColabFold writes into log.txt."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def normalize_receptor_name(fasta_header: str) -> str:
    """Reduce a FASTA header to a filesystem-safe receptor name.

    Mirrors the convention ColabFold uses when naming output PDBs: spaces
    and pipes both collapse to underscores so the resulting string is a
    valid filename and matches the regex in
    :func:`mamp_ml.structure.select_best_pdb_files`.

    Parameters
    ----------
    fasta_header
        FASTA header WITHOUT the leading ``>`` (e.g. ``"Solanum
        habrochates|scaffold11|CORE"``).

    Returns
    -------
    str
        Normalised filename stem (e.g. ``"Solanum_habrochates_scaffold11_CORE"``).
    """
    return re.sub(r"[\s|]+", "_", fasta_header.strip())


def make_colabfold_compatible_pdb_filename(receptor_name: str) -> str:
    """Build a ColabFold-style PDB filename for a given receptor name.

    The downstream :mod:`mamp_ml.structure` pipeline globs for files
    matching ``{receptor}_unrelaxed_rank_[0-9]{3}_alphafold2_ptm_model_<N>_seed_*.pdb``.
    Since ESMFold produces a single structure per sequence, we always emit
    rank ``001``, model number ``1``, seed ``0`` so the glob picks it up.
    """
    return (
        f"{receptor_name}_unrelaxed_rank_001_"
        f"alphafold2_ptm_model_1_seed_000.pdb"
    )


def render_colabfold_compatible_log(
    receptors: List[Tuple[str, int, int, float, float]],
    *,
    backend_label: str = "mamp-ml ESMFold backend (facebook/esmfold_v1)",
) -> str:
    """Render a ``log.txt`` that ``mamp_ml.structure`` can parse unchanged.

    Parameters
    ----------
    receptors
        One tuple per folded receptor:
        ``(receptor_name, original_length, padded_length, plddt, ptm)``.
    backend_label
        Human-readable backend banner written into the first log line.

    Returns
    -------
    str
        Multi-line log content. Lines mirror what ColabFold writes:

        * ``<ts> Running ...`` banner
        * ``<ts> Query <i>/<N>: <receptor_name> (length <L>)`` per receptor
        * ``<ts> Padding length to <L>`` per receptor
        * ``<ts> alphafold2_ptm_model_1_seed_000 recycle=0 pLDDT=... pTM=...``
        * ``<ts> rank_001_alphafold2_ptm_model_1_seed_000 pLDDT=... pTM=...``
    """
    n = len(receptors)
    lines: List[str] = [f"{_now_str()} Running {backend_label}"]
    for i, (name, orig_len, padded_len, plddt, ptm) in enumerate(receptors, 1):
        lines.append(
            f"{_now_str()} Query {i}/{n}: {name} (length {orig_len})"
        )
        lines.append(f"{_now_str()} Padding length to {padded_len}")
        lines.append(
            f"{_now_str()} alphafold2_ptm_model_1_seed_000 recycle=0 "
            f"pLDDT={plddt:.1f} pTM={ptm:.3f}"
        )
        lines.append(
            f"{_now_str()} rank_001_alphafold2_ptm_model_1_seed_000 "
            f"pLDDT={plddt:.1f} pTM={ptm:.3f}"
        )
    return "\n".join(lines) + "\n"


def _read_fasta_records(fasta_path: Path) -> List[Tuple[str, str]]:
    """Tiny self-contained FASTA reader (avoids a circular import on preprocess)."""
    records: List[Tuple[str, str]] = []
    header: Union[str, None] = None
    seq_chunks: List[str] = []
    with open(fasta_path, encoding="utf-8") as fh:
        for raw in fh:
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


def auto_pick_chunk_size(device: str) -> "int | None":
    """Pick an ESMFold trunk chunk size from the host's free VRAM.

    Called after the model has been moved onto ``device`` so the free-VRAM
    query reflects what's actually available for the per-layer activations
    of the folding-trunk forward pass.

    Thresholds are derived from ESMFold's empirical ~17 GB peak VRAM at the
    1024-AA cap (no chunking, fp32 trunk). We aim for ~20 % headroom over
    whatever ChatGPT-style consensus says fits at a given chunk size:

    ====================  =====================
    Free VRAM             Chunk size returned
    ====================  =====================
    ≥ 24 GB               ``None`` (full speed)
    16-24 GB              ``128``
    10-16 GB              ``64``
    6-10 GB               ``32``
    < 6 GB                ``16`` (last resort)
    ====================  =====================

    Parameters
    ----------
    device
        Torch device string (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``,
        ``"mps"``).

    Returns
    -------
    int or None
        Suggested chunk size, or ``None`` to skip chunking entirely.
        Returns ``None`` for any non-CUDA device — chunking only meaningfully
        helps CUDA where peak VRAM is a hard constraint.
    """
    if not device.startswith("cuda"):
        return None

    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except (RuntimeError, AttributeError):
        # Some PyTorch versions / driver setups don't expose mem_get_info.
        # In that case, default to a conservative 64 — fits most cluster
        # GPUs and won't surprise the user.
        return 64

    free_gb = free_bytes / (1024 ** 3)
    if free_gb >= 24:
        return None
    if free_gb >= 16:
        return 128
    if free_gb >= 10:
        return 64
    if free_gb >= 6:
        return 32
    return 16


def _import_esmfold():
    """Lazily import the heavy ESMFold deps, with a clear install hint on failure.

    Two common cluster failures are caught and re-raised with actionable
    messages:

    * ``ImportError`` — ``transformers`` was never installed. The user is
      told to ``pip install mamp-ml[esmfold]``.
    * ``RuntimeError`` mentioning ``torch._six`` — an outdated ``deepspeed``
      in the user's env tried to import the (PyTorch-2-removed) ``torch._six``
      module while ``transformers`` was loading. The user is told to remove
      or upgrade ``deepspeed`` (which the ESMFold path does not need).
    """
    try:
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError as exc:
        raise ImportError(
            "ESMFold backend requires `transformers` and `accelerate`. "
            "Install with: pip install mamp-ml[esmfold]"
        ) from exc
    except RuntimeError as exc:
        if "torch._six" in str(exc):
            raise RuntimeError(
                "ESMFold could not load `transformers` because an outdated "
                "`deepspeed` in your environment imports the removed "
                "`torch._six` module (removed in PyTorch 2.0). ESMFold "
                "inference does NOT require deepspeed; either uninstall it:\n"
                "    pip uninstall -y deepspeed\n"
                "or upgrade it to a torch-2-compatible release:\n"
                "    pip install --upgrade 'deepspeed>=0.12'\n"
                "then re-run the same `mamp-ml predict` command."
            ) from exc
        raise
    import torch  # noqa: F401  -- pulled into the namespace for the caller

    return AutoTokenizer, EsmForProteinFolding


def _chunk_fallback_sequence(initial: "int | None") -> "List[int | None]":
    """Build the ordered list of trunk chunk sizes to attempt on OOM.

    The first element is always ``initial`` (the chunk size we'd use absent
    any memory pressure — possibly ``None`` to mean "no chunking"). Each
    subsequent element is a value from :data:`_OOM_FALLBACK_LADDER` strictly
    smaller than ``initial``, in descending order. ``None`` is treated as
    "larger than any concrete chunk size", so a ``None`` start falls through
    the entire ladder.

    Examples
    --------
    >>> _chunk_fallback_sequence(None)
    [None, 128, 64, 32, 16, 8, 4, 2, 1]
    >>> _chunk_fallback_sequence(64)
    [64, 32, 16, 8, 4, 2, 1]
    >>> _chunk_fallback_sequence(1)
    [1]
    """
    seq: List["int | None"] = [initial]
    for candidate in _OOM_FALLBACK_LADDER:
        if initial is None or candidate < initial:
            seq.append(candidate)
    return seq


def _is_cuda_oom(exc: BaseException, torch) -> bool:
    """True if ``exc`` is (or reads as) a CUDA out-of-memory error.

    ``torch.cuda.OutOfMemoryError`` (torch >= 1.13) is matched directly. Older
    torch builds — and some code paths — surface OOM as a plain
    ``RuntimeError`` whose message contains "out of memory", so we match that
    too. Anything else is a genuine failure and must propagate unchanged.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_cuda_memory(torch) -> None:
    """Return the allocator's cached blocks to the driver between retries.

    Best-effort: guarded so it is a no-op on CPU/MPS hosts and never masks the
    original OOM with a secondary error.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _forward_with_oom_backoff(
    model,
    inputs,
    *,
    receptor_name: str,
    padded_length: int,
    initial_chunk_size: "int | None",
    can_chunk: bool,
    torch,
) -> "Tuple[object, int | None]":
    """Run ``model(**inputs)`` once, retrying smaller chunk sizes on CUDA OOM.

    Starting from ``initial_chunk_size`` we attempt the forward pass; on a CUDA
    out-of-memory error we free the allocator cache, drop to the next smaller
    chunk size from :func:`_chunk_fallback_sequence`, and retry. Only when the
    whole ladder is exhausted do we raise an actionable
    ``torch.cuda.OutOfMemoryError``.

    Returns the model outputs together with the chunk size that actually
    succeeded, so the caller can start the next sequence from that lower size
    instead of re-discovering the ceiling each time.

    When ``can_chunk`` is False (the installed transformers exposes no
    ``trunk.set_chunk_size``) there is no lever to pull, so we make a single
    attempt and propagate any OOM.
    """
    attempts = (
        _chunk_fallback_sequence(initial_chunk_size)
        if can_chunk
        else [initial_chunk_size]
    )
    last_exc: "BaseException | None" = None

    for i, chunk in enumerate(attempts):
        if can_chunk and chunk is not None:
            model.trunk.set_chunk_size(chunk)
        try:
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs, chunk
        except Exception as exc:  # noqa: BLE001 -- re-raised unless it is OOM
            if not _is_cuda_oom(exc, torch):
                raise
            last_exc = exc
            _free_cuda_memory(torch)
            if i + 1 < len(attempts):
                nxt = attempts[i + 1]
                cur_label = chunk if chunk is not None else "disabled"
                print(
                    f"  ! {receptor_name}: CUDA OOM at chunk_size={cur_label}; "
                    f"retrying at chunk_size={nxt} (lower peak VRAM, slower)..."
                )

    retried = (
        f" even after automatically retrying down to chunk_size={attempts[-1]}"
        if can_chunk and len(attempts) > 1
        else ""
    )
    raise torch.cuda.OutOfMemoryError(
        f"{receptor_name}: ESMFold ran out of GPU memory at "
        f"length={padded_length}{retried}. ESMFold's full-precision trunk "
        "needs ~17 GB VRAM at the 1024-AA cap, which exceeds this GPU's free "
        "memory for this sequence length. Options: free other processes on "
        "this GPU, use a larger-memory GPU, or fall back to CPU with "
        "`--device cpu` (much slower but works). You can also force a "
        "specific chunk size with `--chunk-size 64` (or 32). "
        f"Original message: {last_exc}"
    ) from last_exc


def fold_with_esmfold(
    fasta_path: PathLike,
    output_dir: PathLike,
    *,
    device: str = "cpu",
    max_length: int = ESMFOLD_MAX_LENGTH,
    model_id: str = "facebook/esmfold_v1",
    chunk_size: "int | None" = None,
) -> List[Path]:
    """Fold every sequence in ``fasta_path`` and write ColabFold-style outputs.

    For each FASTA record this function:

    1. Truncates the sequence to ``max_length`` (default 1024) if necessary,
       emitting a warning so the user knows what fraction was kept.
    2. Tokenises and runs ``facebook/esmfold_v1`` (lazy-loaded on first call).
    3. Calls the model's ``output_to_pdb`` helper to produce a PDB string;
       ESMFold writes per-residue pLDDT into the B-factor column, so the
       downstream B-factor bandpass stage works unchanged.
    4. Writes the PDB to ``output_dir/{receptor_name}_unrelaxed_rank_001_*.pdb``
       so :mod:`mamp_ml.structure` can pick it up.
    5. After all receptors finish, writes a single ``log.txt`` in
       ColabFold's format so the structure stage's parser handles it.

    Parameters
    ----------
    fasta_path
        Receptor full-length FASTA produced by
        :func:`mamp_ml.preprocess.xlsx_to_receptor_fasta`.
    output_dir
        Where to write the PDBs and the ``log.txt``. Created on demand.
    device
        Torch device string. ``"cuda"`` is highly recommended;
        ESMFold inference on CPU for a 1024-AA sequence is ~30 min on M2.
    max_length
        Maximum residues to fold; sequences longer than this are truncated
        from the N-terminus. Defaults to ESMFold's positional-embedding cap.
    model_id
        HuggingFace model id; defaults to the canonical ``facebook/esmfold_v1``.
    chunk_size
        If set, applied via ``model.trunk.set_chunk_size(chunk_size)`` before
        the forward pass. This splits the folding-trunk's triangular attention
        into chunks of that many tokens, dramatically reducing peak VRAM at
        the cost of some wall-clock. Typical values: 128 (modest savings),
        64 (~half the peak), 32 (~quarter). If ``None`` (default) AND ``device``
        is a CUDA device, an appropriate chunk size is picked automatically
        from the host's free VRAM via :func:`auto_pick_chunk_size`. Pass an
        explicit integer to override that auto-pick; pass any explicit value
        to skip the auto-pick on CPU/MPS too.

    Returns
    -------
    list of pathlib.Path
        Absolute paths of the PDB files written, in input-FASTA order.

    Raises
    ------
    FileNotFoundError
        If ``fasta_path`` does not exist.
    ImportError
        If ``transformers`` is not installed; the message points at the
        ``pip install mamp-ml[esmfold]`` extra.
    """
    fasta_path = Path(fasta_path)
    output_dir = Path(output_dir)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")

    AutoTokenizer, EsmForProteinFolding = _import_esmfold()
    import torch  # safe: _import_esmfold already validated it

    records = _read_fasta_records(fasta_path)
    if not records:
        warnings.warn(
            f"No records found in {fasta_path}; nothing to fold.",
            stacklevel=2,
        )
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_id} (one-time download: ~7 GB on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmForProteinFolding.from_pretrained(model_id)
    if device == "cpu":
        # ESMFold defaults to fp32; on CPU we cast the ESM backbone to fp16 to
        # roughly halve memory + improve throughput. The folding head stays
        # fp32 since fp16 produces qualitatively wrong structures.
        model.esm = model.esm.half()
    model = model.to(device)
    model.eval()

    # Auto-pick a chunk size on CUDA devices when the user didn't pass one
    # explicitly. Querying free VRAM AFTER the model has landed on device
    # gives a realistic estimate of how much memory is left for activations.
    chunk_size_source = "user-supplied"
    if chunk_size is None and device.startswith("cuda"):
        chunk_size = auto_pick_chunk_size(device)
        chunk_size_source = "auto-selected from free VRAM"

    # Trunk chunking dramatically reduces peak VRAM in the folding trunk's
    # triangular-attention layers at the cost of some wall-clock. ESMFold
    # exposes this via `model.trunk.set_chunk_size(N)`. We detect the API once
    # here; the size is actually applied per-sequence inside
    # `_forward_with_oom_backoff`, so the OOM-backoff loop can keep lowering it
    # when a fold still doesn't fit.
    can_chunk = hasattr(model.trunk, "set_chunk_size")
    if not can_chunk and chunk_size is not None:
        warnings.warn(
            "Installed transformers version does not expose "
            "`model.trunk.set_chunk_size`; chunk-size ignored.",
            stacklevel=2,
        )
        chunk_size = None

    if chunk_size is not None:
        print(f"  (ESMFold trunk chunk size: {chunk_size} [{chunk_size_source}])")
    elif device.startswith("cuda"):
        # CUDA with >= 24 GB free -> no chunking, full speed. Worth surfacing
        # so the user knows the auto-pick decided to skip. If this proves too
        # optimistic for the actual sequence length, the OOM-backoff loop below
        # still kicks in and chunks down automatically.
        print("  (ESMFold trunk chunking: disabled — >= 24 GB free VRAM)")

    log_records: List[Tuple[str, int, int, float, float]] = []
    pdbs_written: List[Path] = []

    # Once a fold succeeds at a reduced chunk size, the GPU has effectively
    # told us its ceiling for this run; start subsequent sequences from there
    # rather than re-discovering the same OOM each time.
    effective_chunk_size = chunk_size

    for header, sequence in records:
        receptor_name = normalize_receptor_name(header)
        original_length = len(sequence)
        if original_length > max_length:
            warnings.warn(
                f"{receptor_name}: truncating from {original_length} to "
                f"{max_length} AAs (ESMFold positional-embedding cap).",
                stacklevel=2,
            )
            sequence = sequence[:max_length]
        padded_length = len(sequence)

        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # The trunk's triangular attention scales quadratically with sequence
        # length; a 1024-AA fold needs ~17 GB VRAM without chunking. When the
        # GPU can't fit the current chunk size, `_forward_with_oom_backoff`
        # frees the cache and retries at progressively smaller chunk sizes
        # before giving up.
        outputs, effective_chunk_size = _forward_with_oom_backoff(
            model,
            inputs,
            receptor_name=receptor_name,
            padded_length=padded_length,
            initial_chunk_size=effective_chunk_size,
            can_chunk=can_chunk,
            torch=torch,
        )

        # `output_to_pdb` returns one PDB string per batch element; we feed one
        # sequence at a time, so we take element 0.
        pdb_string = model.output_to_pdb(outputs)[0]

        # Per-residue pLDDT — ESMFold returns a per-atom array; mean over the
        # CA atoms (atom index 1) gives the residue-level confidence and
        # mean-over-residues gives the model-level pLDDT.
        # outputs.plddt has shape (batch, n_residues, n_atoms)
        per_residue_plddt = outputs.plddt[0, :, 1].cpu().numpy()
        mean_plddt = float(per_residue_plddt.mean())

        # ESMFold does not compute pTM. We emit a sentinel so the log line is
        # well-formed and `parse_colabfold_log` extracts something; the
        # downstream best-model picker only ranks on pLDDT, so this value
        # never influences receptor selection.
        ptm = 0.0

        pdb_filename = make_colabfold_compatible_pdb_filename(receptor_name)
        pdb_path = output_dir / pdb_filename
        pdb_path.write_text(pdb_string)
        pdbs_written.append(pdb_path)

        log_records.append(
            (receptor_name, original_length, padded_length, mean_plddt, ptm)
        )

    log_text = render_colabfold_compatible_log(log_records)
    (output_dir / "log.txt").write_text(log_text, encoding="utf-8")

    return pdbs_written

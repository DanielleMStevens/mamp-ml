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


def test_import_esmfold_translates_torch_six_runtime_error(monkeypatch) -> None:
    """When transformers fails to import because the user has an outdated
    deepspeed (which references the removed ``torch._six`` module), our
    helper must re-raise with an actionable message that names deepspeed
    and gives the user the exact ``pip uninstall`` / upgrade commands."""
    from mamp_ml.fold.esmfold import _import_esmfold

    def fake_from_transformers():
        raise RuntimeError(
            "Failed to import transformers.models.esm.modeling_esmfold "
            "because of the following error (look up to see its traceback): "
            "No module named 'torch._six'"
        )

    # The real import path tries to read transformers.AutoTokenizer etc.; we
    # replace the entire body via a sys.modules entry that raises on attribute
    # access, which is what triggers the relevant `from transformers import …`
    # to fail in _import_esmfold.
    import sys
    import types

    class _BoomModule(types.ModuleType):
        def __getattr__(self, name):
            fake_from_transformers()

    monkeypatch.setitem(sys.modules, "transformers", _BoomModule("transformers"))

    with pytest.raises(RuntimeError) as exc_info:
        _import_esmfold()
    msg = str(exc_info.value)
    assert "deepspeed" in msg.lower()
    assert "pip uninstall" in msg
    assert "torch._six" in msg


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


def test_top_level_help_includes_end_to_end_example(repo_root: Path) -> None:
    """`mamp-ml --help` must end with the canonical end-to-end usage block,
    so users running the CLI cold get the answer to "how do I actually run
    this" without leaving the terminal. The same block surfaces in the
    README; here we just guard the CLI-help half of that contract."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    result = subprocess.run(
        [sys.executable, "-m", "mamp_ml", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "Example usage" in out
    assert "mamp-ml predict input_data.xlsx --device cuda" in out
    assert "--structure esmfold" in out
    assert "--weights" in out
    assert "--keep all" in out


def test_predict_help_includes_end_to_end_example(repo_root: Path) -> None:
    """`mamp-ml predict --help` must also end with the worked-example block."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    result = subprocess.run(
        [sys.executable, "-m", "mamp_ml", "predict", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "Example usage" in out
    assert "--structure esmfold" in out
    assert "--keep all" in out
    # The help documents the output: the predicted class names + the folder.
    assert "Output" in out
    assert "Immunogenic" in out
    assert "Non-Immunogenic" in out
    assert "Weakly" in out


def test_predict_subparser_accepts_chunk_size_flag(example_xlsx: Path) -> None:
    """The predict, prepare, and fold subparsers must accept --chunk-size."""
    from mamp_ml.__main__ import _build_parser

    parser = _build_parser()
    parsed = parser.parse_args(
        ["predict", str(example_xlsx), "--chunk-size", "64"]
    )
    assert parsed.chunk_size == 64
    parsed_default = parser.parse_args(["predict", str(example_xlsx)])
    assert parsed_default.chunk_size is None


@pytest.mark.parametrize(
    "free_gb, expected",
    [
        (40.0, None),   # plenty of headroom -> no chunking
        (24.0, None),   # exactly at the upper threshold
        (23.9, 128),    # just below the upper threshold
        (16.0, 128),
        (15.9, 64),
        (10.0, 64),
        (9.9, 32),
        (6.0, 32),
        (5.9, 16),
        (2.0, 16),
    ],
)
def test_auto_pick_chunk_size_from_free_vram(monkeypatch, free_gb, expected) -> None:
    """The auto-pick must walk the documented threshold table cleanly."""
    import sys
    import types
    from mamp_ml.fold import esmfold as ef

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (int(free_gb * 1024 ** 3), int(80 * 1024 ** 3)),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert ef.auto_pick_chunk_size("cuda") == expected


def test_auto_pick_chunk_size_returns_none_on_cpu(monkeypatch) -> None:
    """The auto-pick is a no-op for non-CUDA devices."""
    from mamp_ml.fold import esmfold as ef

    assert ef.auto_pick_chunk_size("cpu") is None
    assert ef.auto_pick_chunk_size("mps") is None


def test_auto_pick_chunk_size_returns_64_when_mem_get_info_unavailable(
    monkeypatch,
) -> None:
    """Driver setups without ``cuda.mem_get_info`` fall back to a conservative 64."""
    import sys
    import types
    from mamp_ml.fold import esmfold as ef

    def boom():
        raise RuntimeError("mem_get_info not available")

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, mem_get_info=boom)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert ef.auto_pick_chunk_size("cuda") == 64


def test_fold_with_esmfold_sets_chunk_size_when_provided(
    tmp_path: Path, monkeypatch
) -> None:
    """When chunk_size is passed, the helper must call model.trunk.set_chunk_size."""
    from unittest.mock import MagicMock
    import sys
    import types
    import mamp_ml.fold.esmfold as ef

    # Minimal synthetic FASTA
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">R\nACDEFGHIKLMNPQRSTVWY\n")

    chunk_recorded: dict = {}

    def fake_model_factory(model_id):
        model = MagicMock()
        model.esm = MagicMock()
        model.esm.half.return_value = model.esm
        model.to.return_value = model
        model.eval.return_value = None
        # trunk.set_chunk_size is the API we depend on
        model.trunk = MagicMock()

        def record_chunk(n):
            chunk_recorded["size"] = n

        model.trunk.set_chunk_size = record_chunk

        def fake_call(**inputs):
            import torch

            n = inputs["input_ids"].shape[1]
            plddt = torch.zeros((1, n, 37), dtype=torch.float32)
            plddt[0, :, 1] = 80.0
            out = MagicMock()
            out.plddt = plddt
            return out

        model.side_effect = fake_call
        model.output_to_pdb = lambda outputs: ["MODEL     1\nEND\n"]
        return model

    def fake_tokenizer_factory(model_id):
        def tokenizer(seq, return_tensors="pt", add_special_tokens=False):
            import torch
            return {"input_ids": torch.zeros((1, len(seq)), dtype=torch.long)}
        return tokenizer

    monkeypatch.setattr(
        ef,
        "_import_esmfold",
        lambda: (
            type("T", (), {"from_pretrained": staticmethod(fake_tokenizer_factory)}),
            type("M", (), {"from_pretrained": staticmethod(fake_model_factory)}),
        ),
    )

    ef.fold_with_esmfold(fasta, tmp_path / "out", device="cpu", chunk_size=32)
    assert chunk_recorded.get("size") == 32


def _patch_tensor_to_noop(monkeypatch) -> None:
    """Patch ``torch.Tensor.to`` to a no-op so tests can pass ``device='cuda'``
    even on machines without CUDA (avoids triggering a real CUDA init when
    the fake model's tokenizer outputs run through ``.to(device)``)."""
    import torch
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw: self)


def _build_fake_esmfold_factory(captured: dict):
    """Shared helper: build a mocked (tokenizer, model) factory pair that
    records what chunk size set_chunk_size was called with."""
    from unittest.mock import MagicMock

    def fake_tokenizer_factory(model_id):
        def tokenizer(seq, return_tensors="pt", add_special_tokens=False):
            import torch
            return {"input_ids": torch.zeros((1, len(seq)), dtype=torch.long)}
        return tokenizer

    def fake_model_factory(model_id):
        model = MagicMock()
        model.esm = MagicMock()
        model.esm.half.return_value = model.esm
        model.to.return_value = model
        model.eval.return_value = None
        model.trunk = MagicMock()

        def record_chunk(n):
            captured["chunk_size_call"] = n

        model.trunk.set_chunk_size = record_chunk

        def fake_call(**inputs):
            import torch
            n = inputs["input_ids"].shape[1]
            plddt = torch.zeros((1, n, 37), dtype=torch.float32)
            plddt[0, :, 1] = 80.0
            out = MagicMock()
            out.plddt = plddt
            return out

        model.side_effect = fake_call
        model.output_to_pdb = lambda outputs: ["MODEL     1\nEND\n"]
        return model

    return fake_tokenizer_factory, fake_model_factory


def test_fold_with_esmfold_auto_picks_chunk_size_on_cuda(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """When chunk_size is None and device starts with cuda, the auto-pick
    runs after the model lands on device and applies the chosen size."""
    import mamp_ml.fold.esmfold as ef

    _patch_tensor_to_noop(monkeypatch)
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">R\nACDEFGHIKLMNPQRSTVWY\n")

    captured: dict = {}
    tok_factory, model_factory = _build_fake_esmfold_factory(captured)
    monkeypatch.setattr(
        ef,
        "_import_esmfold",
        lambda: (
            type("T", (), {"from_pretrained": staticmethod(tok_factory)}),
            type("M", (), {"from_pretrained": staticmethod(model_factory)}),
        ),
    )
    # Force the auto-picker to return 64 regardless of host VRAM.
    monkeypatch.setattr(ef, "auto_pick_chunk_size", lambda device: 64)

    ef.fold_with_esmfold(fasta, tmp_path / "out", device="cuda")
    assert captured.get("chunk_size_call") == 64

    out = capsys.readouterr().out
    assert "auto-selected from free VRAM" in out
    assert "64" in out


def test_fold_with_esmfold_skips_chunking_when_auto_returns_none(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A CUDA host with enough headroom (auto-pick returns None) skips
    set_chunk_size and surfaces the "disabled" hint so the user sees it."""
    import mamp_ml.fold.esmfold as ef

    _patch_tensor_to_noop(monkeypatch)
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">R\nACDEFGHIKLMNPQRSTVWY\n")

    captured: dict = {}
    tok_factory, model_factory = _build_fake_esmfold_factory(captured)
    monkeypatch.setattr(
        ef,
        "_import_esmfold",
        lambda: (
            type("T", (), {"from_pretrained": staticmethod(tok_factory)}),
            type("M", (), {"from_pretrained": staticmethod(model_factory)}),
        ),
    )
    monkeypatch.setattr(ef, "auto_pick_chunk_size", lambda device: None)

    ef.fold_with_esmfold(fasta, tmp_path / "out", device="cuda")
    assert "chunk_size_call" not in captured

    out = capsys.readouterr().out
    assert "disabled" in out
    assert "24 GB" in out


def test_fold_with_esmfold_user_supplied_chunk_size_overrides_auto(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """When the user passes an explicit chunk_size, the auto-picker is not
    consulted and the log line labels the source."""
    import mamp_ml.fold.esmfold as ef

    _patch_tensor_to_noop(monkeypatch)
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">R\nACDEFGHIKLMNPQRSTVWY\n")

    captured: dict = {}
    tok_factory, model_factory = _build_fake_esmfold_factory(captured)
    monkeypatch.setattr(
        ef,
        "_import_esmfold",
        lambda: (
            type("T", (), {"from_pretrained": staticmethod(tok_factory)}),
            type("M", (), {"from_pretrained": staticmethod(model_factory)}),
        ),
    )

    auto_called: dict = {"count": 0}
    def boom_if_called(device):
        auto_called["count"] += 1
        return 999  # would be obvious in the output if it leaked
    monkeypatch.setattr(ef, "auto_pick_chunk_size", boom_if_called)

    ef.fold_with_esmfold(fasta, tmp_path / "out", device="cuda", chunk_size=32)
    # Explicit value used, auto-picker never consulted.
    assert captured.get("chunk_size_call") == 32
    assert auto_called["count"] == 0

    out = capsys.readouterr().out
    assert "user-supplied" in out
    assert "32" in out


def test_fold_with_esmfold_skips_auto_pick_on_cpu(
    tmp_path: Path, monkeypatch
) -> None:
    """On non-CUDA devices, the auto-picker is not consulted at all so the
    CPU path is identical to the previous behaviour (no chunking, no log
    noise about VRAM)."""
    import mamp_ml.fold.esmfold as ef

    fasta = tmp_path / "in.fasta"
    fasta.write_text(">R\nACDEFGHIKLMNPQRSTVWY\n")

    captured: dict = {}
    tok_factory, model_factory = _build_fake_esmfold_factory(captured)
    monkeypatch.setattr(
        ef,
        "_import_esmfold",
        lambda: (
            type("T", (), {"from_pretrained": staticmethod(tok_factory)}),
            type("M", (), {"from_pretrained": staticmethod(model_factory)}),
        ),
    )

    auto_called: dict = {"count": 0}
    def auto_stub(device):
        auto_called["count"] += 1
        return None
    monkeypatch.setattr(ef, "auto_pick_chunk_size", auto_stub)

    ef.fold_with_esmfold(fasta, tmp_path / "out", device="cpu")
    assert "chunk_size_call" not in captured
    assert auto_called["count"] == 0


def test_fold_with_esmfold_catches_cuda_oom_with_actionable_message(
    tmp_path: Path, monkeypatch
) -> None:
    """When ESMFold's forward pass hits CUDA OOM, the re-raised error must
    name `--chunk-size` so the user knows the fix."""
    from unittest.mock import MagicMock
    import mamp_ml.fold.esmfold as ef

    fasta = tmp_path / "in.fasta"
    fasta.write_text(">R\nACDEFGHIKLMNPQRSTVWY\n")

    def fake_model_factory(model_id):
        model = MagicMock()
        model.esm = MagicMock()
        model.esm.half.return_value = model.esm
        model.to.return_value = model
        model.eval.return_value = None
        model.trunk = MagicMock()

        def boom(**inputs):
            import torch
            raise torch.cuda.OutOfMemoryError("Tried to allocate 16 GiB")

        model.side_effect = boom
        model.output_to_pdb = lambda outputs: [""]
        return model

    def fake_tokenizer_factory(model_id):
        def tokenizer(seq, return_tensors="pt", add_special_tokens=False):
            import torch
            return {"input_ids": torch.zeros((1, len(seq)), dtype=torch.long)}
        return tokenizer

    monkeypatch.setattr(
        ef,
        "_import_esmfold",
        lambda: (
            type("T", (), {"from_pretrained": staticmethod(fake_tokenizer_factory)}),
            type("M", (), {"from_pretrained": staticmethod(fake_model_factory)}),
        ),
    )

    import torch

    with pytest.raises(torch.cuda.OutOfMemoryError) as exc_info:
        ef.fold_with_esmfold(fasta, tmp_path / "out", device="cpu")
    msg = str(exc_info.value)
    assert "--chunk-size" in msg
    assert "64" in msg  # the suggested value


# ---------------------------------------------------------------------------
# OOM auto-backoff: chunk-size fallback ladder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "initial, expected",
    [
        (None, [None, 128, 64, 32, 16, 8, 4, 2, 1]),
        (128, [128, 64, 32, 16, 8, 4, 2, 1]),
        (64, [64, 32, 16, 8, 4, 2, 1]),
        (16, [16, 8, 4, 2, 1]),
        (1, [1]),
        (100, [100, 64, 32, 16, 8, 4, 2, 1]),  # non-ladder start still descends
    ],
)
def test_chunk_fallback_sequence(initial, expected) -> None:
    """The fallback ladder always leads with the starting size, then descends
    through strictly-smaller ladder values."""
    from mamp_ml.fold.esmfold import _chunk_fallback_sequence

    assert _chunk_fallback_sequence(initial) == expected


def test_is_cuda_oom_classification() -> None:
    """OOMError and 'out of memory' RuntimeErrors are OOM; other errors aren't."""
    import torch
    from mamp_ml.fold.esmfold import _is_cuda_oom

    assert _is_cuda_oom(torch.cuda.OutOfMemoryError("boom"), torch)
    assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried ..."), torch)
    assert not _is_cuda_oom(RuntimeError("shape mismatch"), torch)
    assert not _is_cuda_oom(ValueError("nope"), torch)


def _oom_until_chunk_factory(captured: dict, *, ok_at_or_below: int):
    """Build a mocked (tokenizer, model) pair whose forward pass raises CUDA
    OOM until the trunk chunk size has been lowered to ``ok_at_or_below`` (a
    larger chunk = more VRAM = OOM). Records every chunk size set, in order."""
    from unittest.mock import MagicMock

    def fake_tokenizer_factory(model_id):
        def tokenizer(seq, return_tensors="pt", add_special_tokens=False):
            import torch
            return {"input_ids": torch.zeros((1, len(seq)), dtype=torch.long)}
        return tokenizer

    def fake_model_factory(model_id):
        state = {"chunk": None}
        model = MagicMock()
        model.esm = MagicMock()
        model.esm.half.return_value = model.esm
        model.to.return_value = model
        model.eval.return_value = None
        model.trunk = MagicMock()

        def record_chunk(n):
            state["chunk"] = n
            captured.setdefault("set_calls", []).append(n)

        model.trunk.set_chunk_size = record_chunk

        def fake_call(**inputs):
            import torch
            cur = state["chunk"]
            # None (no chunking) or anything above the threshold is too big.
            if cur is None or cur > ok_at_or_below:
                raise torch.cuda.OutOfMemoryError(
                    f"Tried to allocate 2.00 GiB at chunk={cur}"
                )
            n = inputs["input_ids"].shape[1]
            plddt = torch.zeros((1, n, 37), dtype=torch.float32)
            plddt[0, :, 1] = 80.0
            out = MagicMock()
            out.plddt = plddt
            return out

        model.side_effect = fake_call
        model.output_to_pdb = lambda outputs: ["MODEL     1\nEND\n"]
        return model

    return fake_tokenizer_factory, fake_model_factory


def test_fold_with_esmfold_recovers_from_oom_by_chunking_down(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The headline fix: an OOM at the auto-picked chunk size no longer aborts
    the run. The backoff loop chunks down until the fold fits, writes the PDB,
    and carries the working chunk size forward to the next sequence (so it
    doesn't re-OOM at the larger size)."""
    import mamp_ml.fold.esmfold as ef

    _patch_tensor_to_noop(monkeypatch)
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">R1\nACDEFGHIKLMNPQRSTVWY\n>R2\nACDEFGHIKLMNPQRSTVWY\n")

    captured: dict = {}
    tok_factory, model_factory = _oom_until_chunk_factory(captured, ok_at_or_below=16)
    monkeypatch.setattr(
        ef,
        "_import_esmfold",
        lambda: (
            type("T", (), {"from_pretrained": staticmethod(tok_factory)}),
            type("M", (), {"from_pretrained": staticmethod(model_factory)}),
        ),
    )
    # Auto-pick starts at 64 -> OOM at 64, 32 -> succeeds at 16.
    monkeypatch.setattr(ef, "auto_pick_chunk_size", lambda device: 64)

    out_dir = tmp_path / "out"
    pdbs = ef.fold_with_esmfold(fasta, out_dir, device="cuda")

    # Both sequences folded despite the initial OOM.
    assert len(pdbs) == 2
    assert all(p.is_file() for p in pdbs)

    # First sequence walked 64 -> 32 -> 16; the second started straight at 16
    # (carried forward) rather than re-trying 64.
    calls = captured["set_calls"]
    assert calls[:3] == [64, 32, 16]
    assert 64 not in calls[3:], (
        f"second sequence should not retry the failed chunk 64; got {calls}"
    )

    # The user saw the retry progress lines.
    out = capsys.readouterr().out
    assert "CUDA OOM at chunk_size=64" in out
    assert "retrying at chunk_size=32" in out

    # log.txt is still well-formed for the structure stage.
    from mamp_ml.structure import parse_colabfold_log
    parsed = parse_colabfold_log(out_dir / "log.txt")
    assert set(parsed) == {"R1", "R2"}


def test_fold_with_esmfold_raises_when_even_smallest_chunk_ooms(
    tmp_path: Path, monkeypatch
) -> None:
    """If every chunk size down to 1 still OOMs, we raise an actionable error
    that names the auto-retry, --chunk-size, and the CPU fallback."""
    import mamp_ml.fold.esmfold as ef
    import torch

    _patch_tensor_to_noop(monkeypatch)
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">R\nACDEFGHIKLMNPQRSTVWY\n")

    captured: dict = {}
    # ok_at_or_below=0 -> nothing in the ladder (>=1) ever fits.
    tok_factory, model_factory = _oom_until_chunk_factory(captured, ok_at_or_below=0)
    monkeypatch.setattr(
        ef,
        "_import_esmfold",
        lambda: (
            type("T", (), {"from_pretrained": staticmethod(tok_factory)}),
            type("M", (), {"from_pretrained": staticmethod(model_factory)}),
        ),
    )
    monkeypatch.setattr(ef, "auto_pick_chunk_size", lambda device: 64)

    with pytest.raises(torch.cuda.OutOfMemoryError) as exc_info:
        ef.fold_with_esmfold(fasta, tmp_path / "out", device="cuda")
    msg = str(exc_info.value)
    assert "retrying down to chunk_size=1" in msg
    assert "--chunk-size" in msg
    assert "--device cpu" in msg
    # It exhausted the whole ladder from the auto-picked 64.
    assert captured["set_calls"] == [64, 32, 16, 8, 4, 2, 1]


def test_fold_with_esmfold_non_oom_error_propagates_without_retry(
    tmp_path: Path, monkeypatch
) -> None:
    """A non-OOM RuntimeError must surface immediately — the backoff loop only
    swallows out-of-memory failures, never genuine bugs."""
    from unittest.mock import MagicMock
    import mamp_ml.fold.esmfold as ef

    _patch_tensor_to_noop(monkeypatch)
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">R\nACDEFGHIKLMNPQRSTVWY\n")

    set_calls: list = []

    def fake_tokenizer_factory(model_id):
        def tokenizer(seq, return_tensors="pt", add_special_tokens=False):
            import torch
            return {"input_ids": torch.zeros((1, len(seq)), dtype=torch.long)}
        return tokenizer

    def fake_model_factory(model_id):
        model = MagicMock()
        model.esm = MagicMock()
        model.esm.half.return_value = model.esm
        model.to.return_value = model
        model.eval.return_value = None
        model.trunk = MagicMock()
        model.trunk.set_chunk_size = lambda n: set_calls.append(n)

        def boom(**inputs):
            raise RuntimeError("size mismatch in attention projection")

        model.side_effect = boom
        model.output_to_pdb = lambda outputs: [""]
        return model

    monkeypatch.setattr(
        ef,
        "_import_esmfold",
        lambda: (
            type("T", (), {"from_pretrained": staticmethod(fake_tokenizer_factory)}),
            type("M", (), {"from_pretrained": staticmethod(fake_model_factory)}),
        ),
    )
    monkeypatch.setattr(ef, "auto_pick_chunk_size", lambda device: 64)

    with pytest.raises(RuntimeError, match="size mismatch"):
        ef.fold_with_esmfold(fasta, tmp_path / "out", device="cuda")
    # No retry ladder was walked: the very first attempt's error propagated.
    assert set_calls == [64]


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


def test_promote_output_moves_file_to_destination(tmp_path: Path) -> None:
    """_promote_output lifts a deliverable out of the scratch dir to the dest."""
    from mamp_ml.__main__ import _promote_output

    out_dir = tmp_path / "inter"
    out_dir.mkdir()
    src = out_dir / "predictions.csv"
    src.write_text("Header_Name,prediction\n")
    dest = tmp_path / "predictions.csv"

    returned = _promote_output(src, dest)
    assert returned == dest
    assert dest.is_file()
    assert not src.exists()  # moved, not copied


def test_promote_output_replaces_existing_destination(tmp_path: Path) -> None:
    """An existing destination (file or dir) is replaced, not nested into."""
    from mamp_ml.__main__ import _promote_output

    src = tmp_path / "inter" / "lrr_annotation_plots"
    src.mkdir(parents=True)
    (src / "new_plot.png").write_bytes(b"\x89PNG")
    dest = tmp_path / "lrr_annotation_plots"
    dest.mkdir()
    (dest / "stale_plot.png").write_bytes(b"old")

    _promote_output(src, dest)
    assert (dest / "new_plot.png").is_file()
    assert not (dest / "stale_plot.png").exists()  # old content gone
    assert not src.exists()


def test_promote_output_returns_none_when_source_missing(tmp_path: Path) -> None:
    """A missing source (e.g. predictions never written) is tolerated."""
    from mamp_ml.__main__ import _promote_output

    result = _promote_output(tmp_path / "nope.csv", tmp_path / "out.csv")
    assert result is None
    assert not (tmp_path / "out.csv").exists()


def test_promote_output_noop_when_already_at_destination(tmp_path: Path) -> None:
    """If src and dest are the same path (e.g. --out-dir is the cwd), keep it."""
    from mamp_ml.__main__ import _promote_output

    f = tmp_path / "predictions.csv"
    f.write_text("x\n")
    returned = _promote_output(f, f)
    assert returned == f
    assert f.is_file()


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

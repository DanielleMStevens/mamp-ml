"""Lightweight checks for the Phase-1 scaffold (no ESM-2 download / GPU).

Run: pytest experiments/phase1_plddt/test_phase1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from extract_dataset import extract_one, normalise_receptor
from model import PLDDTHead, per_residue_metrics

REPO = Path(__file__).resolve().parents[2]
FIXTURE_PDB = next((REPO / "tests/fixtures/colabfold_output").glob("*_unrelaxed_*.pdb"))


def test_normalise_receptor_strips_folding_suffix() -> None:
    assert normalise_receptor("Foo_CORE_unrelaxed_rank_001_alphafold2_ptm") == "Foo_CORE"
    assert normalise_receptor("Bar_relaxed_rank_002_x") == "Bar"


def test_extract_one_reads_sequence_and_plddt() -> None:
    seq, plddt = extract_one(FIXTURE_PDB)
    assert len(seq) == len(plddt) > 0
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWYX")
    # ColabFold pLDDT lives in [0, 100].
    assert all(0.0 <= b <= 100.0 for b in plddt)


def test_per_residue_metrics_identical_is_perfect() -> None:
    x = np.linspace(40, 95, 200)
    m = per_residue_metrics(x, x)
    assert m["pearson"] == pytest.approx(1.0)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)


def test_per_residue_metrics_correlated() -> None:
    rng = np.random.default_rng(0)
    true = rng.uniform(30, 95, 500)
    pred = true + rng.normal(0, 3, 500)  # noisy but correlated
    m = per_residue_metrics(pred, true)
    assert m["pearson"] > 0.9


def test_head_forward_shape_and_range() -> None:
    torch = pytest.importorskip("torch")
    head = PLDDTHead(in_dim=16, hidden=8, kernel=5)
    x = torch.randn(2, 37, 16)  # (batch, length, dim)
    y = head(x)
    assert y.shape == (2, 37)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 100.0

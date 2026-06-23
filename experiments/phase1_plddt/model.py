"""Phase 1 — folding-free per-residue pLDDT head.

A small 1-D CNN that maps per-residue ESM-2 embeddings to a per-residue pLDDT
(0–100). Trained against pLDDT read from existing ColabFold/ESMFold structures,
so a future run can produce the B-factor weighting *without* folding. Pure
torch + numpy/scipy; importable for testing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # allow importing the metrics without torch
    torch = None
    nn = object  # type: ignore


class PLDDTHead(nn.Module):  # type: ignore[misc]
    """1-D CNN over the sequence: ``(B, L, D) embeddings -> (B, L) pLDDT``.

    Two conv layers (kernel ``k``) give each residue a local window of context —
    pLDDT varies smoothly along the chain, so local context matters more than a
    purely per-residue MLP. A final sigmoid keeps the output in ``[0, 100]``.
    """

    def __init__(self, in_dim: int, hidden: int = 128, kernel: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, 1),
        )

    def forward(self, x):  # x: (B, L, D)
        h = x.transpose(1, 2)            # (B, D, L)
        y = self.net(h).squeeze(1)       # (B, L)
        return torch.sigmoid(y) * 100.0


def per_residue_metrics(pred, true) -> dict:
    """Pearson/Spearman/MAE between two 1-D arrays of per-residue pLDDT.

    Pass the residues concatenated across the evaluation proteins. Spearman is
    the headline number — what matters downstream is whether the *shape* of the
    pLDDT profile (which the B-factor bandpass keys on) is preserved.
    """
    from scipy.stats import pearsonr, spearmanr

    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    out: dict = {"n": int(pred.size)}
    if pred.size < 3 or pred.std() == 0 or true.std() == 0:
        out.update(pearson=float("nan"), spearman=float("nan"),
                   mae=float(np.abs(pred - true).mean()) if pred.size else float("nan"))
        return out
    out["pearson"] = float(pearsonr(pred, true)[0])
    out["spearman"] = float(spearmanr(pred, true)[0])
    out["mae"] = float(np.abs(pred - true).mean())
    return out

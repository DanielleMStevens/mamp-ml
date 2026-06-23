#!/usr/bin/env python
"""Compare two mamp-ml runs to judge whether an alternative structure backend
preserves the pipeline's downstream behaviour.

The downstream ESM-2 model + the B-factor weighting are FIXED, so a ColabFold
alternative is only acceptable if it reproduces, vs the reference run:

  1. final predictions       (predictions.csv)        <- the metric that matters
  2. LRR domain boundaries   (lrr_annotation_results.txt)
  3. per-residue B-factor    (bfactor_winding_lrr_segments.csv)

Point this at two completed run directories (reference first, candidate
second). Each dir is searched recursively for the three files, so you can pass
an output folder, an intermediate_files/<token> working dir (use ``--keep
all``), or a parent of both.

    python experiments/compare_runs.py REF_DIR CAND_DIR [--report out.md]

Pure pandas/numpy — no mamp-ml import — so it runs anywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _find(root: Path, name: str) -> "Optional[Path]":
    """First file named ``name`` anywhere under ``root`` (shallowest wins)."""
    hits = sorted(root.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


# --- predictions ----------------------------------------------------------
def _class_column(df: pd.DataFrame) -> "Optional[str]":
    for c in ("prediction", "predicted_class", "class", "Prediction", "predicted_label"):
        if c in df.columns:
            return c
    # else: the categorical column with the fewest distinct string values
    obj = [c for c in df.columns if df[c].dtype == object]
    return min(obj, key=lambda c: df[c].nunique()) if obj else None


def compare_predictions(ref: Path, cand: Path) -> dict:
    a, b = pd.read_csv(ref), pd.read_csv(cand)
    n = min(len(a), len(b))
    out: dict = {"n_ref": len(a), "n_cand": len(b), "n_compared": n}
    if n == 0:
        return out
    a, b = a.iloc[:n].reset_index(drop=True), b.iloc[:n].reset_index(drop=True)
    cc = _class_column(a)
    if cc and cc in b.columns:
        out["class_agreement"] = float((a[cc].astype(str) == b[cc].astype(str)).mean())
        out["class_column"] = cc
    # per-probability MAE on shared numeric columns
    num = [c for c in a.columns if c in b.columns
           and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    if num:
        maes = {c: float(np.abs(a[c].to_numpy() - b[c].to_numpy()).mean()) for c in num}
        out["prob_mae"] = maes
        out["prob_mae_max"] = max(maes.values())
    return out


# --- LRR boundaries -------------------------------------------------------
def _lrr_spans(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    g = df.groupby("PDB_Filename").agg(
        start=("Start_Position", "min"),
        end=("End_Position", "max"),
        n_regions=("Region_Number", "count"),
    )
    return g


def compare_lrr(ref: Path, cand: Path) -> dict:
    a, b = _lrr_spans(ref), _lrr_spans(cand)
    common = a.index.intersection(b.index)
    out = {"n_ref": len(a), "n_cand": len(b), "n_common": len(common)}
    if len(common) == 0:
        return out
    a, b = a.loc[common], b.loc[common]
    out["region_count_match"] = float((a["n_regions"] == b["n_regions"]).mean())
    out["mean_abs_dstart"] = float(np.abs(a["start"] - b["start"]).mean())
    out["mean_abs_dend"] = float(np.abs(a["end"] - b["end"]).mean())
    return out


# --- B-factor -------------------------------------------------------------
def compare_bfactor(ref: Path, cand: Path) -> dict:
    a, b = pd.read_csv(ref), pd.read_csv(cand)
    key, ridx, val = "Protein Key", "Residue Index", "Filtered B-Factor"
    common = sorted(set(a[key]).intersection(b[key]))
    out = {"n_ref": a[key].nunique(), "n_cand": b[key].nunique(), "n_common": len(common)}
    if not common:
        return out
    corrs, dcounts = [], []
    for k in common:
        ak = a[a[key] == k][[ridx, val]].rename(columns={val: "ref"})
        bk = b[b[key] == k][[ridx, val]].rename(columns={val: "cand"})
        dcounts.append(abs(len(ak) - len(bk)))
        m = ak.merge(bk, on=ridx)
        if len(m) >= 3 and m["ref"].std() > 0 and m["cand"].std() > 0:
            corrs.append(float(np.corrcoef(m["ref"], m["cand"])[0, 1]))
    out["mean_bfactor_corr"] = float(np.mean(corrs)) if corrs else None
    out["mean_abs_drowcount"] = float(np.mean(dcounts)) if dcounts else None
    return out


# --- driver ---------------------------------------------------------------
_FILES = {
    "predictions": ("predictions.csv", compare_predictions),
    "lrr_boundaries": ("lrr_annotation_results.txt", compare_lrr),
    "bfactor": ("bfactor_winding_lrr_segments.csv", compare_bfactor),
}


def compare(ref_dir: Path, cand_dir: Path) -> dict:
    report: dict = {}
    for label, (fname, fn) in _FILES.items():
        rp, cp = _find(ref_dir, fname), _find(cand_dir, fname)
        if rp is None or cp is None:
            report[label] = {"skipped": f"missing {fname} in "
                             f"{'ref' if rp is None else 'cand'} dir"}
            continue
        try:
            report[label] = fn(rp, cp)
        except Exception as exc:  # robustness over a research comparator
            report[label] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


def _render(ref_dir: Path, cand_dir: Path, report: dict) -> str:
    lines = ["# structure-backend comparison", "",
             f"- reference: `{ref_dir}`", f"- candidate: `{cand_dir}`", ""]
    for label, r in report.items():
        lines.append(f"## {label}")
        for k, v in r.items():
            if isinstance(v, dict):
                lines.append(f"- {k}:")
                for kk, vv in v.items():
                    lines.append(f"    - {kk}: {vv:.4g}" if isinstance(vv, float) else f"    - {kk}: {vv}")
            else:
                lines.append(f"- {k}: {v:.4g}" if isinstance(v, float) else f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref_dir", type=Path, help="reference run dir (e.g. ColabFold)")
    p.add_argument("cand_dir", type=Path, help="candidate run dir (e.g. ESMFold)")
    p.add_argument("--report", type=Path, default=None, help="also write a markdown report here")
    args = p.parse_args(argv)
    report = compare(args.ref_dir, args.cand_dir)
    md = _render(args.ref_dir, args.cand_dir, report)
    print(md)
    if args.report:
        args.report.write_text(md, encoding="utf-8")
        print(f"\n(wrote {args.report})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

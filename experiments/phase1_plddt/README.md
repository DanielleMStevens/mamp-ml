# Phase 1 — folding-free per-residue pLDDT head

Goal: produce the per-residue **pLDDT** (the required B-factor weighting signal)
from **ESM-2 embeddings**, with **no folding** — so the weighting no longer
depends on ColabFold/ESMFold.

## Why this matters even more given ESMFold's length ceiling

ESMFold (≤~1024 aa positional cap) and ColabFold (truncated to 1300 aa to avoid
OOM) both have **length/memory limits** that cut off the C-terminal end of long
LRR receptors — so for receptors whose LRR span runs past the cap, the *folded*
structure is wrong or incomplete. Folding cost/memory scales ~O(L²) in the 3-D
structure module, which is the wall you hit.

A pLDDT head on ESM-2 embeddings **sidesteps that wall**: ESM-2 is a sequence
encoder (no 3-D structure module), runs far cheaper, handles longer inputs, and
can be **chunked with overlap** for very long receptors. So it can cover
receptors that no folder fits in memory.

> ⚠️ Scope: this gives **pLDDT only**, not LRR boundaries. Today's B-factor CSV
> needs both pLDDT *and* the LRR breakpoints (still structure-derived). Phase 1
> validates the easy/low-risk half; boundaries are Phase 2 (DeepTMHMM ectodomain
> + ESM-2 repeat head). A learned pLDDT becomes a drop-in *component* once a
> boundary source exists.

## Run

```
# 1. Build the teacher set from structures you've already folded
#    (pLDDT is already in the PDB B-factor column).
python extract_dataset.py /path/to/receptor_only -o dataset.jsonl

# 2. Embed with ESM-2 + train + evaluate on held-out receptors.
python train_head.py dataset.jsonl --esm-model esm2_t33_650M_UR50D \
    --cache-dir emb_cache --epochs 40 --out plddt_head.pt --report report.md
```

- `--esm-model esm2_t6_8M_UR50D` = the *cheap* option matching the embeddings the
  pipeline already computes; `esm2_t33_650M_UR50D` (default) = stronger signal.
  Comparing the two is part of the experiment.
- For a rigorous estimate, cluster receptors by sequence identity (mmseqs/cd-hit)
  and pass `--groups groups.tsv` (`receptor<TAB>cluster`). Random splits overstate
  in-family accuracy and hide the out-of-family failure mode — the case you care
  about for novel receptors.

## Metrics

Held-out per-residue **Spearman** is the headline: the B-factor bandpass keys on
the *shape* of the pLDDT profile, so rank-correlation matters more than absolute
MAE. Precedent (SETH on ProtT5, ~1.2k proteins, ρ≈0.72 disorder / ρ≈0.67 vs AF2
pLDDT) suggests ρ≈0.6–0.8 in-family is achievable.

## Next checks after a good head

1. **Closed-loop:** replace a held-out receptor's PDB B-factors with the
   predicted pLDDT, re-run the real B-factor stage, and compare the resulting
   `bfactor_winding_lrr_segments.csv` (and downstream `predictions.csv`) to the
   ColabFold reference with `../compare_runs.py`. That tests what actually
   matters — does the FIXED model behave the same?
2. **Long receptors:** verify chunked ESM-2 embedding reproduces the single-pass
   profile (no seam artifacts) on receptors beyond the folders' length cap.

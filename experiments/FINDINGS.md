# Findings: ML approaches to replace/accelerate ColabFold

Synthesis of three parallel research investigations (folding-free confidence,
sequence-based boundary detection, distillation/surrogate models). Sources are
linked inline. **Constraints:** downstream ESM-2 model fixed; per-residue
B-factor (pLDDT) weighting required; only the **Cα trace + per-residue pLDDT**
are consumed; ESM-2 embeddings are already computed (so heads on them are ~free).

## The key split

The folding step does **two jobs of very different difficulty**:

- **(b) per-residue pLDDT weighting → EASY to replace.** pLDDT/disorder is
  demonstrably recoverable from protein-LM embeddings, and a faster *folder*
  (ESMFold) emits real pLDDT directly.
- **(a) LRR repeat boundaries from Cα geometry → HARD to replace folding-free.**
  The SOTA for LRR boundaries is exactly our method: an *unsupervised geometric*
  analysis of a folded Cα solenoid. Sequence-only repeat callers fail at the
  divergent N-/C-terminal "capping" repeats — the very residues we crop on.

## Recommended staged plan (low → high risk)

### Phase 0 — swap ColabFold → ESMFold (no ML, no training) ★ do first
- ESMFold is single-sequence (no MSA), MIT, and emits **real per-residue pLDDT
  in the PDB B-factor + Cα**, so our winding-number boundaries **and** B-factor
  weighting work unchanged. ~6–60× faster than AF2. https://github.com/facebookresearch/esm
- It is **already integrated** (`--structure esmfold`). So Phase 0 is purely a
  benchmark: ColabFold vs ESMFold on the same receptors via `compare_runs.py`.
- For receptors beyond ESMFold's ~1024-aa memory ceiling, **OmegaFold** folds up
  to 4096 aa and also emits pLDDT in the B-factor. https://github.com/HeliXonProtein/OmegaFold
- **This is the highest-leverage, lowest-risk change** — keeps a real structure,
  so neither downstream signal regresses.

### Phase 1 — folding-free per-residue pLDDT head (low risk, drops folding for the weighting)
- Train a small CNN/MLP head on **frozen ESM-2 embeddings → per-residue pLDDT**,
  using our existing ColabFold PDBs as labels. Proven recipe:
  - **SETH** — 2-layer CNN on ProtT5 → per-residue disorder, ρ≈0.72, trained on
    ~1.2k proteins; correlates with AF2 pLDDT at ρ≈0.67. https://github.com/Rostlab/SETH
  - **NetSurfP-3.0** — head on ESM-1b → SS/RSA/disorder, ~10k proteins. https://github.com/Eryk96/NetSurfP-3.0
  - **AFDistill** — distills AF confidence → per-residue track, CC-BY. https://github.com/IBM/AFDistill
- **Zero-train alternative:** **ProstT5** translates sequence→Foldseek 3Di tokens
  (~3,600× faster than AF2, MIT); per-residue 3Di-probability entropy is an order
  signal and the long low-entropy periodic run also localizes the LRR solenoid.
  https://huggingface.co/Rostlab/ProstT5
- **Avoid:** `pLDDT-Predictor` (ESM-2→pLDDT) — authors **withdrew the weights for
  overfitting**; the idea is sound but don't use the artifact. https://arxiv.org/abs/2410.21283
- **Validate** with subfamily-/identity-held-out splits (never random); the risk
  is out-of-family generalization to novel receptors.

### Phase 2 — folding-free boundaries (higher risk; gate behind a fallback)
- **Ectodomain span (what we crop):** bound it with **DeepTMHMM** (signal-peptide
  cleavage → TM-helix start) — fast, sequence-only, robust, GFF3 output. The LRR
  ectodomain sits between the SP and the single TM helix. https://dtu.biolib.com/DeepTMHMM
- **Internal repeat boundaries:** sequence-only callers (**DeepLRR** F1≈0.76;
  LRRpredictor) are usable but weak at divergent/terminal repeats. A per-residue
  **ESM-2 boundary head** (CRF, à la PEFT-SP signal-peptide labeling) trained on
  our winding-number labels is an open niche — plausible but unproven.
  https://github.com/zhenyaliu77/DeepLRR
- **Always gate** folding-free boundaries behind an out-of-family / low-confidence
  detector that falls back to ESMFold, because even the geometric method degrades
  on divergent plant NLRs.

## How each option maps to the fixed contract

| Option | Produces Cα? | Produces pLDDT? | Boundaries? | Risk | Effort |
|---|---|---|---|---|---|
| ESMFold (Phase 0) | yes (real) | yes (real) | via geom_lrr | very low | none (integrated) |
| OmegaFold (long receptors) | yes | yes | via geom_lrr | low | new backend |
| Single-seq ColabFold | yes | yes | via geom_lrr | low | one flag |
| ESM-2 pLDDT head (Phase 1) | no | yes (learned) | still needs a span | medium | train small head |
| DeepTMHMM + ESM-2 boundary head (Phase 2) | no | (from Phase 1) | learned | high | ML + fallback |

## First experiment to run (on the cluster)

```
mamp-ml predict input.xlsx --device cuda            --keep all --output-name ref_colabfold
mamp-ml predict input.xlsx --structure esmfold --device cuda --keep all --output-name cand_esmfold
python experiments/compare_runs.py ref_colabfold cand_esmfold --report esmfold_vs_colabfold.md
```

If predictions agree closely, ESMFold is the answer for speed *today* with zero
risk to the fixed model — and Phases 1–2 become optional optimizations.

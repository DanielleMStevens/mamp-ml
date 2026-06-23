# Structure-backend experiments

Goal: **make structural modeling faster / cheaper than ColabFold** without
changing the downstream model.

## Hard constraints (these shape everything)

- The downstream **ESM-2 prediction model + its weights are FIXED.**
- The **B-factor weighting is required** (not optional / not uniform).

So the structure step's *only* job is to feed the model the same three things
it gets today, and any alternative is acceptable **iff** it reproduces them:

| What the model consumes | Produced from | File |
|---|---|---|
| LRR-domain sequences (crop) | Cα trace → geom_lrr winding → breakpoints | `lrr_annotation_results.txt` |
| Per-residue B-factor weighting | per-residue pLDDT at Cα → bandpass | `bfactor_winding_lrr_segments.csv` |
| Final call | the two above → ESM-2 + FiLM | `predictions.csv` |

**Key fact:** the pipeline never uses side chains, all-atom detail, or
MSA-quality refinement — only the **Cα trace** and **per-residue pLDDT**. That
is a low bar, and it's what makes alternatives viable.

## Candidate alternatives to ColabFold

Each must emit a Cα trace + per-residue pLDDT (in the PDB B-factor column) so
the existing `geom_lrr` + B-factor stages parse it unchanged — *or* directly
emit the two derived inputs above.

| # | Approach | Why it could be faster | Effort | Notes |
|---|---|---|---|---|
| 1 | **ESMFold** (already integrated, `--structure esmfold`) | single-sequence, no MSA server | none — just benchmark | first thing to measure |
| 2 | **Single-sequence ColabFold** (`--msa-mode single_sequence`) | MSA search is ColabFold's dominant cost | tiny (extra_args) | keeps AF2 weights |
| 3 | **Ectodomain-only folding** (fold ~first 600–800 aa, not 1300) | AF2 cost scales steeply with length | small | orthogonal; needs a fast TM/signal cut |
| 4 | **OmegaFold** | single-sequence fast folder | new backend | emits pLDDT-like confidence |
| 5 | **Boltz-1/2** (already in the cluster env) | open AF3-class | new backend | mmCIF + confidence; MSA-based |
| 6 | **Distilled ESM-2 heads** → boundaries + pLDDT (folding-free) | reuses the ESM-2 pass we already run | ML project | trained on existing ColabFold outputs; biggest upside |

## How we judge an alternative — `compare_runs.py`

Produce two runs on the **same input** (ColabFold = reference, candidate =
e.g. ESMFold), each with `--keep all` so the intermediates survive:

```
mamp-ml predict input.xlsx --device cuda --keep all --output-name ref_colabfold
mamp-ml predict input.xlsx --structure esmfold --device cuda --keep all --output-name cand_esmfold
```

Then compare their downstream effect:

```
python experiments/compare_runs.py ref_colabfold cand_esmfold --report compare.md
```

It reports, vs the reference:
- **Predictions** — % rows with the same class + per-probability MAE (the metric that actually matters).
- **LRR boundaries** — region-count agreement + mean |Δstart| / |Δend|.
- **B-factor** — per-receptor correlation of the filtered B-factor + row-count agreement.

A candidate is "good enough" when prediction agreement is high; the boundary /
B-factor numbers explain *why* it does or doesn't agree.

## Suggested order of experiments

1. **Baseline:** ColabFold vs **ESMFold** (no new code) — establishes how much
   prediction drift the existing fast path already causes. If small → ESMFold
   is the answer for screens, today.
2. **Single-sequence ColabFold** — keeps AF2 weights; isolates the MSA's value.
3. **Ectodomain-only truncation** — speed lever that stacks with any backend.
4. If a folding-free path is wanted: build the **distilled heads** (extract a
   `(sequence → pLDDT, breakpoints)` dataset from existing structures, train
   small heads, plug their output into the same `lrr_annotation_results.txt` /
   `bfactor_*.csv` contract).

# Folding-free LRR annotation — research-backed design

Goal (revised per your ask): **no structural modeling at all.** Two per-residue
outputs from frozen ESM-2, feeding the FIXED downstream contract:
- **pLDDT** → the required B-factor weighting.
- **LRR boundaries** → the LRR-domain crop + the bandpass region.

Synthesis of three investigations (architectures; pLDDT-as-feature; distillation
+ periodicity + robustness). Citations inline.

## The crux finding (this changes the plan)

Your idea was "feed the boundary head a combination of embeddings **and predicted
pLDDT**." The literature is clear that **predicted pLDDT as an *input* channel is
information-theoretically redundant** with the ESM-2 embeddings it was computed
from (data-processing inequality): it can only act as a finite-data optimization
shortcut that *vanishes as data/model grow*, and it adds two failure traps
(train-on-true/serve-on-predicted covariate shift; label leakage). The clean
protein precedent (EMBER2) found such redundant features "not statistically
significant for the final model." https://arxiv.org/abs/2512.21315 · https://pubmed.ncbi.nlm.nih.gov/35609601/

**So pLDDT belongs as an *auxiliary output head*, not an input** — a multi-task
shared trunk, where the pLDDT task *regularizes* the boundary task (a safe
gradient signal, not a laundered input). Precedent: NetSurfP-3.0 (SS+RSA+disorder
joint), PEER, PatchProt. If you want extra *input* signal, use **pattern**
channels (secondary structure, ProstT5 3Di tokens) — a boundary is where the
structural *pattern changes*, which a confidence scalar captures only indirectly.

> Your intuition (LRR boundaries coincide with pLDDT dips / disorder) is correct
> — the fix is to *predict pLDDT jointly* rather than *consume a predicted pLDDT*.

## The teacher signal is already a phase — distill it as one

geom-lrr (the current method, Xu/Cerbu/Tralie/Krasileva, PLOS Comp Biol 2024,
`pip install geom-lrr`) flattens the Cα solenoid and computes a **winding number
W(t)** that rises ~linearly with slope m≈residues/repeat inside the LRR and slope
0 at the termini; **repeat boundaries are integer crossings of W(t)**. That is a
per-residue **phase** target. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012526

So the highest-leverage design (independently recommended by two agents):
**predict a continuous per-residue phase (the sequence-space analog of W(t))**,
and read boundaries off integer/zero crossings — instead of brittle one-hot
"boundary vs not" labels that discard the quasi-periodicity. Soft/phase targets
beat hard targets for localization across every cited domain (landmark detection,
distance-transform segmentation, the winding method itself).

## Recommended architecture

```
frozen ESM-2 (650M)  ──►  shared trunk (NetSurfP-3.0 style:
   per-residue (L×D)        2× dilated 1-D CNN  +  1–2 BiLSTM)
        │  + aux input channel: embedding self-similarity / autocorrelation
        │    (anchors phase to internal repetition; ESM-2 induction heads
        │     already encode it — pLM-Repeat exploits exactly this)
        ▼
   ┌─────────────┬───────────────────────────────────────────┐
   │ pLDDT head  │ boundary head (primary)                    │
   │ (auxiliary) │  • target: per-residue cyclic phase sin/cos │
   │  per-res    │    (W(t) analog) + Gaussian soft "unit-start"│
   │  regression │    heatmap; decode by integer-crossing /     │
   │             │    peak-pick                                 │
   │             │  • CRF / monotonic repeat-index → ONE        │
   │             │    contiguous, ordered LRR run (PEFT-SP style)│
   │             │  • loss: focal + Dice/Tversky (boundaries     │
   │             │    are rare positives)                       │
   └─────────────┴───────────────────────────────────────────┘
```

Reference recipes: NetSurfP-3.0 (trunk + per-residue heads, ESM embeddings,
~600× faster than MSA) https://github.com/Eryk96/NetSurfP-3.0 ; PEFT-SP (frozen
ESM-2 + linear-chain CRF for a boundary, ±3-residue accuracy)
https://github.com/shuaizengMU/PEFT-SP ; pLM-Repeat (embedding self-similarity
for repeat units) https://github.com/KYQiu21/plmrepeat .

## Honest risk assessment

- **Interior repeats: likely to transfer well** — LRR periodicity is strongly
  encoded in ESM-2 embeddings; per-residue structural-label distillation routinely
  works (SS3 ~80%, RSA ρ~0.79, AFDistill pLDDT r=0.76).
- **Divergent N-/C-terminal "capping" repeats + out-of-family receptors (<~30%
  identity, ESM-2 twilight zone): will be unreliable** — and this is *also* where
  the teacher is weakest (AF2 produces confident-but-wrong β-solenoids; geom-lrr's
  own discrepancy rises 0.127→0.373 on divergent NLRs). Plan for "accurate core,
  unreliable ends/novel families," not uniform accuracy. Treat teacher labels as
  noisy (RSA may be a better-calibrated auxiliary target than raw pLDDT).
- ⚠️ Avoid `pLDDT-Predictor` (arXiv 2410.21283) — **withdrawn by authors for
  overfitting**; sound idea, do not use the artifact.

## Evaluation + abstention (non-negotiable)

- **Splits:** sequence-identity-clustered CV (GraphPart / MMseqs2), never random
  (~0.2 optimism gap). Report boundary error (median |Δ| residues, fraction within
  ±1/±2, whole-repeat-offset rate) **stratified by test-to-train identity and
  terminal-vs-interior repeats**; hold out entire sub-families (NLR vs LRR-RLK vs
  LRR-RLP) for a true out-of-family test. https://academic.oup.com/nargab/article/5/4/lqad088/7318077
- **Abstain + fall back to ColabFold+geom-lrr** when uncertain, via three tiers:
  (1) input OOD gate — max train-set identity; widen abstention below ~30%;
  (2) a 5-member deep ensemble (per-residue disagreement) + split-conformal
  per-breakpoint residue intervals;
  (3) output self-check — reconstruct the winding from predicted boundaries and
  reject if they aren't ~equally spaced at one dominant period (free, mirrors
  geom-lrr's own residual escalation).

## Build order

1. **Teacher extraction:** extend `extract_dataset.py` to also emit the per-residue
   **phase** target (reconstructed from geom-lrr breakpoints: linear ramp slope m
   inside the LRR, integer at each boundary, flat at termini) + the boundary
   positions. (pLDDT already extracted.)
2. **Multi-task model:** shared trunk + pLDDT head + phase/boundary head; add the
   self-similarity input channel; focal+Dice + CRF.
3. **Train + clustered-CV eval**, stratified report, ensemble + conformal.
4. **Closed-loop:** feed predicted boundaries + pLDDT through the real B-factor
   stage and compare predictions to the ColabFold reference via `../compare_runs.py`.

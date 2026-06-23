# legacy/

Archived files that are **no longer used by the package or its tests**, parked
here (rather than deleted) so the history/reference is preserved. Nothing in
this folder ships in the installed wheel or is imported at runtime.

| File | Why it's here | Replaced by |
|---|---|---|
| `extract_lrr_sequences.py` | Old LRR-sequence extraction module (`LRRSequenceExtractor`). Functionally unused — only an import test referenced it. | `mamp_ml.lrr_annotation.geom_lrr` (the geometric LRR annotator) + `mamp_ml.structure` |
| `prepare_input_data.sh` | "Stage 1" of the legacy shell pipeline (xlsx → receptor FASTA). Kept for muscle memory; the e2e test copied but never ran it. | `mamp-ml prepare-fasta` (single stage) or `mamp-ml prepare` / `mamp-ml predict` (full pipeline) |

If you're sure these are gone for good, this whole folder can be deleted.

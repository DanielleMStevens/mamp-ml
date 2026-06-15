# mamp-ml

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielleMStevens/mamp-ml/blob/version2/mamp_ml_colab.ipynb)

Deep-learning predictor for plant receptor–ligand (MAMP) immunogenicity, built on the [ESM-2](https://github.com/facebookresearch/esm) protein language model with structure-aware features from [AlphaFold2 / ColabFold](https://github.com/sokrypton/ColabFold). Achieves 74% accuracy on a held-out test set; supports high-throughput screening of LRR receptor–ligand combinations.

> **Workflow at a glance**
> ```bash
> mamp-ml predict your_input.xlsx
> ```
> That single command runs everything: receptor FASTA generation → ColabFold (with clear gating) → LRR annotation → B-factor analysis → chemical-feature CSV → ESM-2 inference → `predictions.csv`.

---

## Quickstart on Google Colab

The fastest way to get predictions: click the **Open in Colab** badge above and run the notebook. It installs `mamp-ml`, walks you through uploading your spreadsheet, runs ColabFold on a free T4 GPU, runs ESM-2 inference, and offers `predictions.csv` for download. ~15 minutes end-to-end for typical inputs.

---

## Local install

Recommended for users with 10+ receptors who want repeated runs.

### 1. Install the Python package

```bash
# from PyPI (once published)
pip install mamp-ml

# or directly from this repo (version2 branch)
pip install git+https://github.com/DanielleMStevens/mamp-ml.git@version2

# or editable from a clone
git clone --branch version2 https://github.com/DanielleMStevens/mamp-ml.git
cd mamp-ml
pip install -e .
```

The pretrained checkpoint (`mamp_ml_weights.pth`, ~33 MB) is bundled inside the wheel — no separate download.

### 2. Install a structure-prediction tool

`mamp-ml` needs receptor structures to run its LRR-feature analysis. Pick one of two tools — both produce ColabFold-compatible output so the downstream pipeline is unchanged.

#### Option A — ColabFold (default, higher-accuracy folds)

```bash
# Linux + NVIDIA GPU (recommended)
bash scripts/install_colabbatch_linux.sh
```

Or run everything (Python package + ColabFold) in one command via:

```bash
bash install_software.sh
```

#### Option B — ESMFold (in-process, no separate env)

```bash
pip install mamp-ml[esmfold]
```

That's it — ESMFold runs inside the same Python process as `mamp-ml`, no extra conda env or weights download dance. The HuggingFace `facebook/esmfold_v1` model (~7 GB) downloads automatically on first use.

#### Comparison

| | ColabFold (`--structure colabfold`) | ESMFold (`--structure esmfold`) |
|---|---|---|
| Accuracy | Higher (MSA-conditioned, production-tested) | Slightly lower in low-confidence regions |
| Install cost | ~5 GB conda env + ~3.6 GB AF2 params + a separate install script | `pip install mamp-ml[esmfold]` |
| Wall-clock per receptor | ~5 min on RTX 3070+ | ~2 min on RTX 3070+ |
| Max sequence length | unlimited | 1024 AAs (longer truncates from N-terminus → preserves LRR ectodomain) |
| Requires CUDA | yes (for tractable wall-clock) | yes (for tractable wall-clock); also works on Apple Silicon via MPS |

No local GPU at all? Use the [Google Colab notebook](https://colab.research.google.com/github/DanielleMStevens/mamp-ml/blob/version2/mamp_ml_colab.ipynb) — it provisions a free T4 and runs the whole pipeline end-to-end with either tool.

<details>
<summary>macOS install (for package development, not for predictions)</summary>

A macOS install path exists for contributors who need to test the preparation pipeline locally — `bash scripts/install_colabbatch_mac.sh` — but actual prediction wall-clock on Apple CPU is impractical. Run inference on a CUDA host or Colab.

</details>

### 3. Predict

A CUDA-capable GPU is required for reasonable wall-clock — ColabFold and ESM-2 are both heavy.

```bash
mamp-ml predict your_input.xlsx --device cuda

# use ESMFold instead of ColabFold:
mamp-ml predict your_input.xlsx --structure esmfold --device cuda

# use a custom-trained model checkpoint instead of the bundled one:
mamp-ml predict your_input.xlsx --weights /path/to/my_finetune.pth

# keep every intermediate file (default keeps only predictions.csv + LRR plots):
mamp-ml predict your_input.xlsx --keep all
```

By default the run leaves just the user-facing outputs on disk:

| File | What it is |
|---|---|
| `intermediate_files/predictions.csv` | per-row immunogenicity-class probabilities |
| `intermediate_files/lrr_annotation_plots/` | per-receptor LRR regression plots (PNG) |

Pass `--keep all` to retain everything in `intermediate_files/` (FASTAs, CSVs, intermediate text files, the copied ColabFold/ESMFold PDBs) — useful when you want to rerun prediction without re-folding.

---

## Input format

A single Excel file (`.xlsx`, `Sheet1`) with these columns — `example_data.xlsx` in this repo is the canonical reference:

| Column | Description |
|---|---|
| `plant_species` | Species the receptor comes from (e.g. `Solanum habrochates`) |
| `receptor` | Receptor family / short name (e.g. `CORE`) |
| `locus_id` | Unique locus identifier (e.g. `scaffold11`) |
| `receptor_sequence` | Full-length receptor amino-acid sequence (one-letter code) |
| `ligand_sequence` | Ligand (epitope) amino-acid sequence to predict immunogenicity for |

Each row is one receptor–ligand pair to score. Multiple rows can share a receptor; the structure is folded once and re-used.

---

## How `mamp-ml predict` works

```
your_input.xlsx
      │
      ▼  ① receptor FASTA
intermediate_files/receptor_full_length.fasta
      │
      ▼  ② ColabFold (external — gated cleanly if not yet run)
intermediate_files/receptor_only/{log.txt, *.pdb}
      │
      ▼  ③ structure-stage (best-model selection + LRR annotation)
intermediate_files/{alphafold_scores.txt, lrr_annotation_results.txt,
                   pdb_for_lrr_annotator/*.pdb}
      │
      ▼  ④ LRR-domain FASTA + ⑤ B-factor bandpass analysis
intermediate_files/{lrr_domain_sequences.fasta,
                   bfactor_winding_lrr_segments.csv}
      │
      ▼  ⑥ test-data assembly + ⑦ chemical features
intermediate_files/{test_data.csv, ready_test_data.csv}
      │
      ▼  ⑧ ESM-2 inference (bundled mamp_ml_weights.pth)
intermediate_files/predictions.csv      ← what you want
```

`mamp-ml predict` does all eight steps in one shot. If ColabFold hasn't been run yet for your input, it stops cleanly after step ② and prints the exact `colabfold_batch` command for you to run before re-invoking it.

---

## Power-user CLI

Each pipeline stage is exposed as its own subcommand for debugging or partial re-runs:

```bash
mamp-ml prepare-fasta INPUT.xlsx out.fasta
mamp-ml fold receptor.fasta out_dir --structure esmfold      # standalone fold
mamp-ml structure-stage COLABFOLD_DIR scores.txt PDB_DIR lrr.txt
mamp-ml lrr-domain-fasta lrr.txt receptor.fasta out.fasta
mamp-ml bfactor PDB_DIR CACHE_DIR out.csv
mamp-ml assemble-test-data INPUT.xlsx lrr_fasta out.csv
mamp-ml chemical-features in.csv out.csv

# Or run the full prep pipeline (no model inference) in one shot:
mamp-ml prepare INPUT.xlsx
```

`mamp-ml --help` lists everything.

---

## Computational requirements

A **CUDA-capable GPU is required**. ColabFold needs CUDA for tractable folding wall-clock, and ESM-2 inference is dominated by attention over ~1000-residue sequences. We've tested on RTX 3070+, RTX 4090, A100, and the free T4 in Google Colab.

| Step | Resource |
|---|---|
| ColabFold folding | NVIDIA GPU, ~5 min/receptor on RTX 3070+ |
| ESM-2 inference | NVIDIA GPU, ~30s for ~100 receptor-ligand pairs |
| Disk | ~4 GB for ColabFold + AF2 params; bundled MAMP-ml weights ~33 MB |
| RAM | 16 GB minimum |

For users without a local GPU, the [Google Colab notebook](https://colab.research.google.com/github/DanielleMStevens/mamp-ml/blob/version2/mamp_ml_colab.ipynb) gives you a free T4 — the recommended path.

---

## Citation

> Stevens *et al.* 2025. **Mamp-ml: a deep-learning approach to epitope immunogenicity in plants.** *BioRxiv.* DOI: *(forthcoming)*

Pipeline + model design details: see the companion repo [mamp-prediction-ml](https://github.com/DanielleMStevens/mamp_prediction_ml).

---

## Authors

| | |
|---|---|
| **Danielle M. Stevens** [![ORCID](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0001-5630-137X) | Dept. of Plant & Microbial Biology, UC Berkeley |
| **David Yang** | Center for Computational Biology, UC Berkeley |
| **Tatiana Liang** | Dept. of Plant & Microbial Biology, UC Berkeley |
| **Tianrun Li** [![ORCID](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-8589-4634) | Dept. of Plant Pathology, UC Davis |
| **Brandon Vega** | Dept. of Plant & Microbial Biology, UC Berkeley |
| **Gitta Coaker** [![ORCID](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0003-0899-2449) | Dept. of Plant Pathology, UC Davis |
| **Ksenia Krasileva** [![ORCID](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-1679-0700) | Dept. of Plant & Microbial Biology + Computational Biology, UC Berkeley |

---

## Contact + contributions

Issues / questions: [github.com/DanielleMStevens/mamp-ml/issues](https://github.com/DanielleMStevens/mamp-ml/issues), or **Danielle Stevens** ([@dani_m_stevens](https://bsky.app/profile/danimstevens.bsky.social), dmstev@berkeley.edu).

Have a new dataset or want to extend the model to other LRR-PRR families? Please reach out — we're actively working to expand coverage.

---

## License

MIT (see `LICENSE.txt`).

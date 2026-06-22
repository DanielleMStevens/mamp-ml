# mamp-ml

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielleMStevens/mamp-ml/blob/version2/mamp_ml_colab.ipynb)



This repository contains the code for mamp-ml, a deep learning approach to epitope immunogenicity in plants. If you plan to run on a small number of receptor–epitope combinations (less than 10 receptors), we recommend you use Google Colab. If you plan to run on 100–1000s of receptor–epitope combinations, we recommend you install locally and have access to a CUDA-capable GPU (RTX 3070 or better; A5000+ for larger jobs). MSAs for the receptor structures can be pulled either from the ColabFold MSA server (default) or locally via [localcolabfold](https://github.com/YoshitakaMo/localcolabfold).

## Authors
* __Danielle M. Stevens__ <a itemprop="sameAs" content="https://orcid.org/0000-0001-5630-137X" href="https://orcid.org/0000-0001-5630-137X" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a>   </br>
_Dept. of Plant & Microbial Biology, University of California, Berkeley_

* __David Yang__ </br>
_Center for Computational Biology, University of California, Berkeley_

* __Tatiana Liang__ </br>
_Dept. of Plant & Microbial Biology, University of California, Berkeley_

* __Tianrun Li__ <a itemprop="sameAs" content="https://orcid.org/0000-0002-8589-4634" href="https://orcid.org/0000-0002-8589-4634" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a> </br>
_Dept. of Plant Pathology, University of California, Davis_

* __Brandon Vega__ </br>
_Dept. of Plant & Microbial Biology, University of California, Berkeley_

* __Gitta Coaker__ <a itemprop="sameAs" content="https://orcid.org/0000-0003-0899-2449" href="https://orcid.org/0000-0003-0899-2449" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a> </br>
_Dept. of Plant Pathology, University of California, Davis_

* __Ksenia Krasileva__ <a itemprop="sameAs" content="https://orcid.org/0000-0002-1679-0700" href="https://orcid.org/0000-0002-1679-0700" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a> </br>
_Dept. of Plant & Microbial Biology, University of California, Berkeley_, </br>
_Center for Computational Biology, University of California, Berkeley_


## Abstract

>Eukaryotes detect biomolecules through surface-localized receptors, a central signaling response hub. A subset of receptors survey for pathogens, induce immunity, and restrict pathogen growth. Comparative genomics of both hosts and pathogens has unveiled vast sequence variation in receptors and potential ligands, creating an experimental bottleneck. We have developed mamp-ml, a machine learning framework for predicting plant receptor-ligand interactions. We leveraged existing functional data from over two decades of foundational research, together with the large protein language model ESM-2, to build a pipeline and model that predicts immunogenic outcomes using a combination of receptor-ligand features. Our model achieves 74% prediction accuracy on a held-out test set, even when an experimental structure is lacking. Our approach enables high-throughput screening of LRR receptor-ligand combinations and provides a computational framework for engineering plant immune systems.

## General installation and running instructions:

To install mamp-ml and (optionally) ColabFold for receptor structure prediction, run:
```
bash install_software.sh
```

This installs the Python package (mamp-ml + pretrained weights bundled) and ColabFold + its AlphaFold2 parameters. Alternatively, install just the Python package via pip:
```
pip install git+https://github.com/DanielleMStevens/mamp-ml.git@version2
```

> **Note:** If you see `WARNING: There was an error checking the latest version of pip`, you can ignore it. That is pip's own self-update check failing to reach PyPI — common on cluster compute nodes with restricted internet — not a problem with the install, which has already completed. To suppress it, prefix the command with `PIP_DISABLE_PIP_VERSION_CHECK=1`.

To **reinstall / upgrade** to the latest version (e.g. to pick up a fix pushed to the branch), force a clean reinstall — because the branch version often doesn't change between pushes, a plain `pip install` will report "already satisfied" and skip the update:
```
pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/DanielleMStevens/mamp-ml.git@version2
```
You can confirm which version is installed with `mamp-ml --version`.

The sample spreadsheet ships inside the package, so you can grab a local copy to inspect the expected input format (or to smoke-test the install) without cloning the repo:
```
mamp-ml example                          # writes example_data.xlsx into the current directory
mamp-ml predict example_data.xlsx --device cuda
```
Or run the whole pipeline directly on the bundled sample in one step:
```
mamp-ml predict --example --device cuda
```

Please prepare an excel file in the following format (see example_data.xlsx as an example):
```
plant_species | receptor | locus_id | receptor_sequence | ligand_sequence
```

Once your excel file is prepared, the full pipeline runs in a single command:
```
mamp-ml predict input_data.xlsx --device cuda
```

The first invocation generates the receptor FASTA, then folds the receptors: if a `colabfold_batch` install is found on the system it is run automatically, otherwise the command prints the exact `colabfold_batch` invocation for you to run and exits. Once ColabFold has produced the structures, the rest of the pipeline (LRR annotation, B-factor analysis, chemical features, ESM-2 inference) runs end-to-end.

Alternatively, ESMFold can fold the receptors in-process without a separate ColabFold install:
```
pip install mamp-ml[esmfold]
mamp-ml predict input_data.xlsx --structure esmfold --device cuda
```

A successful run always bundles its outputs — `predictions.csv` (per row: the predicted class and the per-class probabilities, labeled `Immunogenic` / `Non-Immunogenic` / `Weakly Immunogenic` rather than 0/1/2), `lrr_annotation_plots/` (per-receptor LRR regression plots), `structures/` (the folded receptor structures — see reuse below), and `mamp-ml-run.log` (see below) — into a labeled folder in the directory you ran the command from. By default the folder is uniquely named so repeated runs don't overwrite each other, e.g. `output_Cabernet_filtered_RLKs_2026-06-15_20-30-00_a4f9/predictions.csv`; pass `--output-name myrun` to name it (`myrun/predictions.csv`). The `--keep` flag controls only the leftover working directory: the default removes it (the structures + log are already promoted to the output folder), while `--keep all` keeps it (useful for debugging). Pass `--weights /path/to/checkpoint.pth` to predict against a custom-trained model instead of the bundled one.

**Running many jobs at once (HPC / SLURM).** Each run uses a *unique* working directory (`intermediate_files/<input>_<timestamp-or-SLURM-job>_<random>/`) and a matching unique output folder, so you can launch multiple runs — e.g. a SLURM array job — from the same directory and they will never clobber each other or pick up each other's folds. This needs no configuration and works identically off-cluster (laptop, workstation, cloud); when SLURM is present the folder name embeds the job id so it maps back to `sacct`. Pin `--out-dir` / `--output-name` if you want explicit names.

**Reusing structures (e.g. sweeping weights).** Folding is the expensive step, so each run saves its structures to `output_.../structures/`. To predict the same receptors again with a different model — without re-folding — point `--structures` at that folder; the structural-modeling stage is skipped entirely:
```
mamp-ml predict data.xlsx --weights weightsA.pth
mamp-ml predict data.xlsx --structures output_data_..._a4f9/structures --weights weightsB.pth
mamp-ml predict data.xlsx --structures output_data_..._a4f9/structures --weights weightsC.pth
```
The exact `--structures` command for a finished run is printed in its completion summary.

**Progress + run log.** The terminal stays compact: the long folding step shows a single per-receptor progress bar instead of ColabFold/ESMFold's verbose output. That full backend output — together with the exact command, the installed `mamp-ml` version, and every pipeline step — is written to `mamp-ml-run.log`. It is created in the working directory from the very start of the run (so it survives an early failure like an OOM-killed ColabFold, exit status `-9`) and copied into the final output folder on success. **When reporting an error, attach this file**: it records the version that actually ran (handy for spotting a stale install) and the complete backend log.

### Model weights cache (and HPC quotas)

`mamp-ml` downloads the model weights (ESM-2, and ESMFold if you use it) through the HuggingFace cache. By **default it caches into a `model_cache/` folder next to the mamp-ml install** — i.e. on whatever filesystem you installed onto — rather than the usual `~/.cache/huggingface`. This keeps the (multi-GB) weights off a small-quota HOME on shared systems without you having to configure anything.

To put the cache somewhere else (a shared read-only mirror, a larger volume, etc.), either pass `--cache-dir` or export `HF_HOME`:
```
mamp-ml predict input_data.xlsx --device cuda --cache-dir /path/with/room
# or, once for the whole session / in ~/.bashrc:
export HF_HOME=/path/with/room
```
An explicit `HF_HOME` always wins over the default. If a run ever hits `OSError: ... Disk quota exceeded`, `mamp-ml predict` catches it and prints exactly this guidance.

## Computational requirements:

To run this package locally, we recommend having a CUDA-capable NVIDIA GPU and at least 16 GB RAM and 16 GB VRAM. The main step that is slow and memory-intensive is running the structure prediction (ColabFold/AlphaFold2 or ESMFold). While we were able to run predictions on a 1080Ti, we found considerable runtime improvements using RTX A5000 and A100 cards. For users without a local GPU, the [Google Colab notebook](https://colab.research.google.com/github/DanielleMStevens/mamp-ml/blob/version2/mamp_ml_colab.ipynb) provisions a free T4 and runs the whole pipeline end-to-end.


__If you use this tool, please cite the following paper:__ </br>
Stevens et al. 2025. Mamp-ml: a deep learning approach to epitope immunogenicity in plants. _BioRxiv._ </br>
DOI:


Details on building this pipeline and model can be found in another GitHub Repo: [mamp-prediction-ml](https://github.com/DanielleMStevens/mamp_prediction_ml).


License
----
Code is freely available under the MIT license


Have data to contribute?
----
We are always looking to improve mamp-ml to improve prediction accuracy and expand to other LRR-PRR receptors and their protein ligands. Please feel free to contact us if you have recently published a dataset or would like to contribute to make this tool better!


Contact
----
Please feel free to contact me directly with any questions or issues with the code
Danielle Stevens - [@dani_m_stevens](https://bsky.app/profile/danimstevens.bsky.social) - dmstev@berkeley.edu

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

Please prepare an excel file in the following format (see example_data.xlsx as an example):
```
plant_species | receptor | locus_id | receptor_sequence | ligand_sequence
```

Once your excel file is prepared, the full pipeline runs in a single command:
```
mamp-ml predict input_data.xlsx --device cuda
```

The first invocation generates the receptor FASTA, then exits with the colabfold_batch command for you to run; once ColabFold has produced the structures, re-invoke the command and the rest of the pipeline (LRR annotation, B-factor analysis, chemical features, ESM-2 inference) runs end-to-end.

Alternatively, ESMFold can fold the receptors in-process without a separate ColabFold install:
```
pip install mamp-ml[esmfold]
mamp-ml predict input_data.xlsx --structure esmfold --device cuda
```

A successful run produces `intermediate_files/predictions.csv` (per-row class probabilities) and `intermediate_files/lrr_annotation_plots/` (per-receptor LRR regression plots). By default the other intermediates are cleaned up; pass `--keep all` to retain them. Pass `--weights /path/to/checkpoint.pth` to predict against a custom-trained model instead of the bundled one.

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

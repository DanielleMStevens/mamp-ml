# mamp-ml

This repository contains the code for mamp-ml, a deep learning approach to epitope immunogenicity in plants. If you plan to run on a small number of receptor-epitope combinations (less than 10 receptors), we recommend you use co-lab. If you plan to run on 100-1000s of receptor-epitope combinations, we recommend you install locally, have access to a GPU (at least A5000) and potenitally adjust the code to pull MSAs for receptor structure generation locally (future feature enhancement). To do so, please see info from localcolab fold (link here).

## Authors
'- Danielle M. Stevens
Dept. of Plant & Microbial Biology, University of California, Berkeley

'-David Yang
Center for Computational Biology, University of California, Berkeley

'-Tatiana Liang
Dept. of Plant & Microbial Biology, University of California, Berkeley

'-Tianrun Li
Dept. of Plant Pathology, University of California, Davis

'-Brandon Vega
Dept. of Plant & Microbial Biology, University of California, Berkeley

'-Gitta Coaker
Dept. of Plant Pathology, University of California, Davis

'-Ksenia Krasileva
Dept. of Plant & Microbial Biology, University of California, Berkeley

## Abstract

>Eukaryotes detect biomolecules through surface-localized receptors, a central signaling response hub. A subset of receptors survey for pathogens, induce immunity, and restrict pathogen growth. Comparative genomics of both hosts and pathogens has unveiled vast sequence variation in receptors and potential ligands, creating an experimental bottleneck. We have developed mamp-ml, a machine learning framework for predicting plant receptor-ligand interactions. We leveraged existing functional data from over two decades of foundational research, together with the large protein language model ESM-2, to build a pipeline and model that predicts immunogenic outcomes using a combination of receptor-ligand features. Our model achieves 74% prediction accuracy on a held-out test set, even when an experimental structure is lacking. Our approach enables high-throughput screening of LRR receptor-ligand combinations and provides a computational framework for engineering plant immune systems.

## General installation and running instructions:

```
To install the software needed before model prediction, AlphaFold2 via colabfold and LRR-Annotation, we will run the following command:
bash install_software.sh
```

Once your excel file with receptor and ligands sequences is prepared (see example_data.xlsx to see format), run the following command to run the pipeline.
```
bash run_prediction_pipeline.sh input_data.xlsx
```


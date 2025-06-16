# mamp-ml

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RSMHMNVxAQiz-V5PAfHaKG9V-CkKYurH?usp=sharing)



This repository contains the code for mamp-ml, a deep learning approach to epitope immunogenicity in plants. If you plan to run on a small number of receptor-epitope combinations (less than 10 receptors), we recommend you use Google Colab. If you plan to run on 100-1000s of receptor-epitope combinations, we recommend you install locally, have access to a GPU (at least A5000) and potenitally adjust the code to pull MSAs for receptor structure generation locally (future enhancement). To do so, please see info from localcolab: [link here](https://github.com/YoshitakaMo/localcolabfold).

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

To install the software needed before model prediction, AlphaFold2 via colabfold and LRR-Annotation, we will run the following command:
```
bash install_software.sh
```

Please prepare an excel file in the following format (see example_data.xlsx as an example):
```
plant_species | locus_id | receptor | ligand_sequence | receptor_sequence
```

Once your excel file with receptor and ligands sequences is prepared, follow the below pipeline:
```
# this will transform your excel sheet into a fasta file
bash mamp-ml/prepare_input_data.sh input_data.xlsx

# activate AlphaFold and run (please update your path
conda activate localfold
export PATH="/global/scratch/users/dmstev/localcolabfold/colabfold-conda/bin:$PATH" 
colabfold_batch --num-models 1 ./mamp-ml/intermediate_files/receptor_full_length.fasta \
   ./mamp-ml/intermediate_files/receptor_only/

# once AlphaFold models are complete, run to process the model structures via LRR-Annotation
# and prep data for prediction via mamp-ml
bash run_prediction_pipeline.sh input_data.xlsx

# deactivate conda environment to activate another
conda deactivate localfold
conda activate esmfold
python mamp-ml/main_train.py \
    --model esm2_bfactor_weighted \
    --eval_only_data_path /content/mamp-ml/intermediate_files/ready_test_data.csv \
    --model_checkpoint_path /content/mamp-ml/mamp_ml_weights.pth \
    --device cpu \
    --disable_wandb
```

A sucessful run will prodice a csv file with processed input data (plant species, receptor, locus_id, ligand and receptor sequence) as well as prediction and their associated softmax probabilities. 

## Computational requirements:

To run this package locally, we recommend having compute with a NIVDIA GPU available and at least 16 GB RAM and 16 GB VRAM. The main step that is slow + memory intensive is running AlphaFold. While we were able to run predictions on a 1080Ti, we found considerable runtime improvements using RTX A5000 cards. 


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

#!/bin/bash

# Exit on error
set -e

echo "Creating esmfold environment..."
conda create -n esmfold python=3.9 -y

echo "Activating esmfold environment..."
eval "$(conda shell.bash hook)"
conda activate esmfold

echo "Installing PyTorch and related packages..."
conda install -y -c pytorch-nightly -c conda-forge -c bioconda \
    pytorch \
    torchvision \
    torchaudio \
    openmm=7.7.0 \
    pdbfixer \
    einops \
    fairscale \
    omegaconf \
    hydra-core \
    pandas \
    pytest \
    numpy=1.24.3 \
    hmmer \
    kalign2 \
    wandb=0.12.21

echo "Installing pip packages..."
conda run -n esmfold pip install biopython==1.79 \
    deepspeed==0.5.9 \
    dm-tree==0.1.6 \
    ml-collections==0.1.0 \
    pytorch_lightning==2.0.9 \
    transformers \
    git+https://github.com/NVIDIA/dllogger.git

echo "Installation complete! You can now activate the environment with:"
echo "conda activate esmfold" 
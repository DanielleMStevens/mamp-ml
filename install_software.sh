conda create --name=localfold python=3.11 -y
conda activate localfold
conda install mamba 
mamba update --all
mamba install cuda -c nvidia
bash scripts/install_colabbatch_linux.sh


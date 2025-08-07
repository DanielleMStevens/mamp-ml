# install conda packages + localcolabfold
conda install -n base python=3.11 -y
conda install mamba
mamba update --all
mamba install cuda -c nvidia
wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh
bash install_colabbatch_linux.sh

# install some packages for LRR-Annotation
pip install matplotlib-inline
conda install -c conda-forge wxpython -y
pip install scikit-learn
pip install openpyxl
conda deactivate

# install packages need for esm-2
bash /content/mamp-ml/scripts/install_esmfold_env.sh
pip install torchvision
pip install transformers
pip install wandb
pip install scikit-learn
pip install matplotlib

# --------- old code based from during development ------------
#conda create --name=localfold python=3.11 -y
#conda activate localfold
#conda install mamba 
#amba update --all
#mamba install cuda -c nvidia
#bash scripts/install_colabbatch_linux.sh

#conda deactivate
#conda create --name env_esm.ym
#conda env create -f env_esm.yml
#conda env update -f env_esm.yml

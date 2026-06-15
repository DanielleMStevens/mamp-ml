#!/bin/bash
#
# install_software.sh
# -------------------
# Bootstrap script for users who want a local install of mamp-ml + ColabFold.
# Creates a fresh conda env, installs the Python package + its dependencies,
# and then installs ColabFold (which is too heavy / GPU-specific to ship as
# a pip extra).
#
# Usage:
#   bash install_software.sh
#
# After this script finishes, you can run:
#   conda activate mamp-ml
#   mamp-ml predict your_input.xlsx

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 1) Create the mamp-ml conda env from environment.yml (Python 3.10 + all the
#    package's runtime deps, then `pip install -e .[esmfold]` at the end).
conda env create -f "${REPO_ROOT}/environment.yml" || conda env update -f "${REPO_ROOT}/environment.yml"

# 2) Install ColabFold into a separate prefix env. Picks the right installer
#    for the current OS — the Linux installer pulls a CUDA build of jax,
#    the Mac (arm64) one uses CPU jax instead.
case "$(uname -s)" in
    Linux)  bash "${REPO_ROOT}/scripts/install_colabbatch_linux.sh" ;;
    Darwin) bash "${REPO_ROOT}/scripts/install_colabbatch_mac.sh" ;;
    *)      echo "Unsupported OS: $(uname -s). Install ColabFold manually."; exit 1 ;;
esac

echo
echo "Done. To use mamp-ml:"
echo "  conda activate mamp-ml"
echo "  export PATH=\"${REPO_ROOT}/localcolabfold/colabfold-conda/bin:\$PATH\""
echo "  mamp-ml predict your_input.xlsx"

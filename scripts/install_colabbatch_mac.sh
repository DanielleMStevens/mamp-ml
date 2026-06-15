#!/bin/bash -e
# =============================================================================
# install_colabbatch_mac.sh
# -----------------------------------------------------------------------------
# macOS (arm64 / Apple Silicon) sibling of install_colabbatch_linux.sh.
#
# Mirrors the structure of the Linux installer step-for-step but adapts each
# step to macOS:
#
#   - Skips the Miniforge3 install (uses the user's existing conda).
#   - Uses CPU JAX (jax / jaxlib without the [cuda12] extra), because Apple
#     Silicon Macs don't have CUDA.
#   - Drops mmseqs2 + hhsuite from the conda install — colabfold_batch's
#     default MSA mode queries the remote ColabFold MSA API server, so the
#     local MMseqs2 / HHsuite binaries are not required to fold the example
#     receptors. Add them back if you ever want to do local-MSA pairing.
#   - Uses macOS-compatible sed flags (BSD sed -i '' rather than GNU sed -i).
#   - Installs to ./localcolabfold/colabfold-conda just like the Linux script,
#     so the rest of the toolchain (e.g. PATH exports in
#     run_preparation_pipeline.sh) needs no changes.
#
# Usage:
#   conda activate localfold        # uses any conda env as a launcher
#   bash scripts/install_colabbatch_mac.sh
#
# Cost: ~5 GB of disk for the env + ~3.6 GB for the AlphaFold2 params.
# Wall-clock: ~10-15 min total (conda solve + pip + params download).
# =============================================================================

type curl >/dev/null 2>&1 || { echo "curl not installed. brew install curl." >&2 ; exit 1 ; }

CURRENTPATH="$(pwd)"
COLABFOLDDIR="${CURRENTPATH}/localcolabfold"

mkdir -p "${COLABFOLDDIR}"

# Initialise conda for this shell session.
# Uses the caller's existing conda install rather than installing a fresh
# Miniforge3 (which would only duplicate state).
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the dedicated colabfold env at $COLABFOLDDIR/colabfold-conda.
# kalign2 is pulled from bioconda; openmm + pdbfixer + git from conda-forge.
# All three are available for osx-arm64 (verified at install time).
conda create -p "$COLABFOLDDIR/colabfold-conda" \
    -c conda-forge -c bioconda \
    git python=3.10 openmm pdbfixer kalign2=2.04 -y

conda activate "$COLABFOLDDIR/colabfold-conda"

# Install ColabFold and (CPU) JAX.
# The Linux script pins jax==0.4.35 to keep things deterministic; we do the
# same so that the structures we produce can be cross-checked against the
# Linux pipeline.
"$COLABFOLDDIR/colabfold-conda/bin/pip" install --no-warn-conflicts \
    "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
"$COLABFOLDDIR/colabfold-conda/bin/pip" install "colabfold[alphafold]"
"$COLABFOLDDIR/colabfold-conda/bin/pip" install --upgrade "jax==0.4.35" "jaxlib==0.4.35"
"$COLABFOLDDIR/colabfold-conda/bin/pip" install tensorflow silence_tensorflow

# Patch colabfold source for macOS-friendly behaviour.
#
# NOTE: BSD sed (macOS) and GNU sed (Linux) handle `\n` in the replacement
# string very differently. GNU sed expands it to a newline; BSD sed leaves
# it literal, which silently corrupts any multi-line substitution. We work
# around this by:
#   - Skipping the Agg-backend patch entirely. Modern colabfold imports
#     `from matplotlib import pyplot as plt` INSIDE function bodies (not at
#     module level), so injecting Agg before it via sed leaves the new
#     statements at column 0 and breaks indentation. Instead, the user sets
#     `MPLBACKEND=Agg` via env-var if they need it; the default Mac backend
#     handles headless invocation correctly.
#   - Doing the other two patches via single-line substitutions only.
pushd "${COLABFOLDDIR}/colabfold-conda/lib/python3.10/site-packages/colabfold"
# Pin the AlphaFold2 params cache directory to our local install root so it
# doesn't pollute ~/Library/Caches and is co-located with the env.
sed -i '' -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${COLABFOLDDIR}/colabfold\"#g" download.py
# Suppress noisy tensorflow warnings during inference. Single-line sub,
# no multi-line replacement, so BSD-sed-safe.
python3 -c "
import re
p = 'batch.py'
src = open(p).read()
needle = 'from io import StringIO'
if needle in src and 'silence_tensorflow' not in src:
    src = src.replace(needle, needle + '\nfrom silence_tensorflow import silence_tensorflow\nsilence_tensorflow()')
    open(p, 'w').write(src)
"
rm -rf __pycache__
popd

# Download the AlphaFold2 model weights.
# Only the monomeric pTM params (~3.6 GB) are needed for folding single-chain
# receptors like the ones in example_data.xlsx. We deliberately skip the
# multimer download (~3.7 GB more) which would only be used for hetero-complex
# prediction — not part of this pipeline.
"$COLABFOLDDIR/colabfold-conda/bin/python3" -m colabfold.download AlphaFold2-ptm
echo "Download of alphafold2 weights finished."
echo "-----------------------------------------"
echo "Installation of ColabFold (macOS / arm64) finished."
echo "Add ${COLABFOLDDIR}/colabfold-conda/bin to your PATH to use colabfold_batch."
echo -e "i.e. for Bash/Zsh:\n\texport PATH=\"${COLABFOLDDIR}/colabfold-conda/bin:\$PATH\""
echo "For more details, please run 'colabfold_batch --help'."

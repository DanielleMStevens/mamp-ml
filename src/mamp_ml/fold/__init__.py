"""mamp_ml.fold — structure-prediction backends.

The MAMP-ml pipeline needs receptor structures to run its B-factor bandpass
LRR-repeat analysis. Two backends produce those structures:

* :mod:`mamp_ml.fold.esmfold` — opt-in, runs in-process via HuggingFace
  ``facebook/esmfold_v1``. No external env, fast on GPU (5-15 min/receptor),
  works on CPU but slowly. Limited to ~1024 AA per sequence.

* ``colabfold_batch`` — the canonical backend; lives outside this package
  because it has heavy native deps (jax CUDA build, openmm, etc.). The
  pipeline gates on its presence rather than running it inline.

The CLI's ``--backend`` flag picks which path to take. Both backends produce
identical on-disk output (``log.txt`` + ``*_unrelaxed_rank_001_*.pdb``
files) so the downstream :mod:`mamp_ml.structure` stage sees the same
schema regardless of which folded the receptor.
"""

from __future__ import annotations

__all__ = ["esmfold"]

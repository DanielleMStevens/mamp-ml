"""Helpers for working with an externally-installed ColabFold.

ColabFold itself isn't part of `mamp-ml` — it's the heavy structure-prediction
binary the user supplies (or installs via :file:`scripts/install_colabbatch_*.sh`).
On cluster hosts, ColabFold is frequently already present somewhere — either as
a sysadmin-provided ``module load`` install, a personal conda env, or a
``localcolabfold/`` clone under the user's home directory. Rather than ask the
user to remember the right ``export PATH`` incantation, this module sniffs the
host for any reachable ``colabfold_batch`` binaries and reports them.

The main entry point is :func:`find_colabfold_installs`, which returns a list
of every ``colabfold_batch`` we can locate. It's surfaced two places in the
CLI:

* ``python -m mamp_ml find-colabfold`` — explicit list, with the
  ``export PATH=...`` command needed to use each one.

* The colabfold gate in :func:`mamp_ml.__main__._run_prepare` — when the user
  runs ``mamp-ml predict`` and ColabFold output is missing, the same search
  fires automatically and the hint message highlights any local installs.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple

#: Filenames we recognise as the ColabFold entry point.
_COLABFOLD_BIN_NAME = "colabfold_batch"

# Common locations where conda lives. We check each for an ``envs/`` subtree
# and then scan ``envs/<name>/bin/colabfold_batch``.
_CONDA_ROOT_CANDIDATES: Tuple[str, ...] = (
    "~/anaconda3",
    "~/miniconda3",
    "~/miniforge3",
    "~/conda",
    "/opt/anaconda3",
    "/opt/miniconda3",
    "/opt/miniforge3",
    "/usr/local/anaconda3",
    "/usr/local/miniconda3",
)

# Directories where users typically drop a ``localcolabfold`` clone. Each
# parent gets a ``<parent>/localcolabfold/colabfold-conda/bin/colabfold_batch``
# probe (the layout produced by :file:`scripts/install_colabbatch_linux.sh`
# and :file:`scripts/install_colabbatch_mac.sh`).
_LOCALCOLABFOLD_PARENT_CANDIDATES: Tuple[str, ...] = (
    "~",
    "~/scratch",
    "~/projects",
    ".",
    "/opt",
    "/usr/local",
)


def _resolve_silently(p: Path) -> "Path | None":
    """Resolve ``p`` to an absolute path; return ``None`` on permission errors."""
    try:
        return p.resolve()
    except (OSError, RuntimeError):
        return None


def find_colabfold_installs() -> List[Tuple[Path, str]]:
    """Locate every reachable ``colabfold_batch`` on this machine.

    Search order (deduplicated by resolved path):

    1. The user's ``$PATH`` (via :func:`shutil.which`).
    2. ``$CONDA_PREFIX`` and the env tree it implies, if set.
    3. Standard conda root candidates (``~/anaconda3``,
       ``~/miniconda3``, ``/opt/...``, etc.) — each ``envs/*/bin/colabfold_batch``
       is probed.
    4. ``localcolabfold/colabfold-conda/bin/colabfold_batch`` under typical
       parent directories (``~``, cwd, ``/opt``, ``/usr/local``, etc.).

    Returns
    -------
    list of (pathlib.Path, str)
        One tuple per distinct install: ``(absolute_path_to_binary,
        short_human_description)``. Empty if no installs are found.

    Notes
    -----
    * Symlinks are followed (so an install reachable via both ``$PATH`` and
      ``$CONDA_PREFIX`` is reported once, at its target).
    * Directories we can't access (permission errors on a shared host) are
      silently skipped — we never raise.
    * The search is bounded: no recursive directory walks beyond the
      candidates listed above, so a worst-case run completes in well under
      a second even on a slow filesystem.
    """
    found: List[Tuple[Path, str]] = []
    seen: set = set()

    def _register(candidate: Path, description: str) -> None:
        if not candidate.is_file():
            return
        resolved = _resolve_silently(candidate)
        if resolved is None or resolved in seen:
            return
        seen.add(resolved)
        found.append((resolved, description))

    # 1. $PATH ---------------------------------------------------------
    on_path = shutil.which(_COLABFOLD_BIN_NAME)
    if on_path:
        _register(Path(on_path), "on $PATH")

    # 2. $CONDA_PREFIX implied env tree -------------------------------
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        # If we're in a child env, walk back to the conda root's envs/.
        envs_root = prefix.parent
        if envs_root.is_dir() and envs_root.name == "envs":
            for env_dir in _list_dir_silent(envs_root):
                _register(env_dir / "bin" / _COLABFOLD_BIN_NAME, f"conda env: {env_dir.name}")
        # Also check the current env itself (the user may be running mamp-ml
        # from within the same env where colabfold_batch is installed).
        _register(prefix / "bin" / _COLABFOLD_BIN_NAME, f"current conda env: {prefix.name}")

    # 3. Common conda roots -------------------------------------------
    for candidate in _CONDA_ROOT_CANDIDATES:
        envs_dir = Path(candidate).expanduser() / "envs"
        if not envs_dir.is_dir():
            continue
        for env_dir in _list_dir_silent(envs_dir):
            _register(
                env_dir / "bin" / _COLABFOLD_BIN_NAME,
                f"conda env: {env_dir.name} (under {candidate})",
            )

    # 4. localcolabfold installs --------------------------------------
    for parent_candidate in _LOCALCOLABFOLD_PARENT_CANDIDATES:
        parent = Path(parent_candidate).expanduser()
        if not parent.is_dir():
            continue
        # Look for *exact* `localcolabfold` directory under parent.
        lcf = parent / "localcolabfold" / "colabfold-conda" / "bin" / _COLABFOLD_BIN_NAME
        _register(lcf, f"localcolabfold install at {parent / 'localcolabfold'}")

    return found


def _list_dir_silent(p: Path) -> List[Path]:
    """``iterdir`` that swallows permission errors."""
    try:
        return list(p.iterdir())
    except (OSError, RuntimeError):
        return []


def format_activation_hint(install_path: Path) -> str:
    """Return the ``export PATH=...`` line a user would run to activate
    ``install_path``.

    Parameters
    ----------
    install_path
        Absolute path to a ``colabfold_batch`` binary (as returned by
        :func:`find_colabfold_installs`).

    Returns
    -------
    str
        Single-line shell command, ready to be copy-pasted.
    """
    bin_dir = install_path.parent
    return f'export PATH="{bin_dir}:$PATH"'

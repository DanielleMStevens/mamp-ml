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
  fires automatically; a discovered install is run via
  :func:`run_colabfold_batch` (no manual re-invocation), and only when nothing
  is found does the gate fall back to a copy-paste hint.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple, Union

PathLike = Union[str, Path]

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


def build_colabfold_command(
    binary: PathLike,
    fasta_path: PathLike,
    output_dir: PathLike,
    *,
    num_models: int = 1,
    num_recycle: int = 1,
    extra_args: "Sequence[str] | None" = None,
) -> List[str]:
    """Assemble the ``colabfold_batch`` argv for a discovered install.

    Factored out from :func:`run_colabfold_batch` so the exact command can be
    asserted in tests and reused in user-facing hints. The binary is referenced
    by its absolute path; positional ``fasta_path`` and ``output_dir`` come
    last, matching ColabFold's ``colabfold_batch <input> <output>`` contract.
    """
    cmd = [
        str(binary),
        "--num-models",
        str(num_models),
        "--num-recycle",
        str(num_recycle),
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend([str(fasta_path), str(output_dir)])
    return cmd


def run_colabfold_batch(
    binary: PathLike,
    fasta_path: PathLike,
    output_dir: PathLike,
    *,
    num_models: int = 1,
    num_recycle: int = 1,
    extra_args: "Sequence[str] | None" = None,
) -> int:
    """Run a discovered ``colabfold_batch`` binary on ``fasta_path``.

    The binary is invoked by its **absolute path**, so it works regardless of
    the caller's active environment: localcolabfold and conda installs ship a
    ``colabfold_batch`` whose shebang points at their own interpreter, so no
    ``export PATH`` / ``conda activate`` is required. ColabFold's own output is
    streamed straight to the terminal (not captured) so the user sees its
    progress live.

    Parameters
    ----------
    binary
        Absolute path to a ``colabfold_batch`` binary, as returned by
        :func:`find_colabfold_installs`.
    fasta_path
        Input receptor FASTA.
    output_dir
        Directory ColabFold writes its PDBs + ``log.txt`` into.
    num_models, num_recycle
        Passed through as ``--num-models`` / ``--num-recycle``. Defaults match
        the pipeline's documented invocation (1 model, 1 recycle).
    extra_args
        Optional extra CLI arguments inserted before the positional
        input/output (e.g. ``["--amber"]``).

    Returns
    -------
    int
        The subprocess exit code (``0`` on success). If the binary cannot be
        executed at all (missing / not executable), returns ``127`` after
        printing an error — mirroring a shell "command not found".
    """
    cmd = build_colabfold_command(
        binary,
        fasta_path,
        output_dir,
        num_models=num_models,
        num_recycle=num_recycle,
        extra_args=extra_args,
    )
    try:
        completed = subprocess.run(cmd, check=False)
    except OSError as exc:
        print(f"Failed to execute {binary}: {exc}")
        return 127
    return completed.returncode


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

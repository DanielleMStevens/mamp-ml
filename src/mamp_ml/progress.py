"""Eye-friendly, timed progress reporting for the mamp-ml CLI pipeline.

Pure standard-library (no ``rich`` / ``tqdm`` dependency). The output is
designed to degrade gracefully:

* Glyphs are plain Unicode (``▶ ✓ ✗ ↳ →``), which render on any modern
  terminal and stay readable even where they don't.
* ANSI colour is emitted **only** when writing to a real TTY that hasn't opted
  out via ``NO_COLOR`` (or forced it on via ``FORCE_COLOR``), so piping the run
  to a file or a CI log stays clean and greppable.

Each step prints a header line with a ``[i/N]`` tag and a *rough* time estimate
up front, then a ``✓ <elapsed> · <summary>`` line when it finishes — so the
user always knows where they are, roughly how long the current step should
take, and how long it actually took.
"""

from __future__ import annotations

import datetime
import os
import platform
import re
import sys
import time
from typing import List, Optional, TextIO, Tuple

__all__ = [
    "PipelineProgress",
    "Phase",
    "RunLogger",
    "FoldProgressBar",
    "format_duration",
]

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_RED = "\033[31m"

_RULE_WIDTH = 62

#: Matches any ANSI SGR escape (colour/style) so the plain-text run log stays
#: greppable even when the terminal copy is coloured.
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Return ``text`` with any ANSI colour/style escapes removed."""
    return _ANSI_RE.sub("", text)


def format_duration(seconds: float) -> str:
    """Render an elapsed time compactly: ``0.4s`` / ``14.2s`` / ``6m 12s`` / ``1h 03m``."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes}m {secs:02d}s"
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes:02d}m"


class RunLogger:
    """A plain-text transcript of one CLI run, written to a file in the output.

    The terminal stays concise (numbered step lines + a fold progress bar), but
    everything — the exact command, the package version, every step line, and
    the *full* ColabFold / ESMFold subprocess output — is appended here so a
    failed run (e.g. an OOM-killed ColabFold, exit status ``-9``) can be
    diagnosed from a single self-contained file.

    The file is written eagerly (flushed after every line) so a hard crash or
    an OOM kill still leaves a usable log on disk. All writes are best-effort:
    if the log file becomes unwritable mid-run we swallow the error rather than
    take down the pipeline over a diagnostics side-channel.

    Parameters
    ----------
    path
        Where to write the log (e.g. ``intermediate_files/mamp-ml-run.log``).
    command
        The reconstructed command line (``mamp-ml predict ... --device cuda``).
    version
        The installed ``mamp-ml`` version string, recorded so a stale install
        is obvious from the log alone.
    context
        Optional extra ``label -> value`` pairs to record in the header (e.g.
        input file, device, structure backend).
    """

    def __init__(
        self,
        path: "object",
        *,
        command: str,
        version: str,
        context: "Optional[List[Tuple[str, str]]]" = None,
    ) -> None:
        from pathlib import Path

        self.path = Path(path)
        self._fh: "Optional[TextIO]" = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", encoding="utf-8")
        except OSError:
            self._fh = None
        self._write_header(command, version, context or [])

    def _write_header(
        self, command: str, version: str, context: "List[Tuple[str, str]]"
    ) -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows: List[Tuple[str, str]] = [
            ("date", now),
            ("version", f"mamp-ml {version}"),
            ("command", command),
            ("python", platform.python_version()),
            ("platform", platform.platform()),
        ]
        rows.extend(context)
        width = max(len(label) for label, _ in rows)
        lines = ["=" * _RULE_WIDTH, "mamp-ml run log", "=" * _RULE_WIDTH]
        for label, value in rows:
            lines.append(f"{label.ljust(width)} : {value}")
        lines.append("-" * _RULE_WIDTH)
        self.write("\n".join(lines) + "\n")

    def write(self, text: str) -> None:
        """Append ``text`` verbatim to the log (best-effort, flushed)."""
        if self._fh is None:
            return
        try:
            self._fh.write(text)
            self._fh.flush()
        except (OSError, ValueError):
            # Disk full / quota / already-closed handle: never let logging
            # failures escape and break the actual pipeline.
            self._fh = None

    def write_line(self, text: str) -> None:
        """Append a single line (a trailing newline is ensured)."""
        self.write(text.rstrip("\n") + "\n")

    def close(self) -> None:
        """Flush and close the underlying file handle (idempotent)."""
        if self._fh is None:
            return
        try:
            self._fh.flush()
            self._fh.close()
        except (OSError, ValueError):
            pass
        finally:
            self._fh = None


class FoldProgressBar:
    """In-place progress bar for the long structure-prediction step.

    Replaces the verbose ColabFold / ESMFold per-recycle output on the terminal
    with a single advancing bar (``[23/58] ▸ folding Vitis_vinifera_… (1617 aa)``).
    The full backend output still goes to the :class:`RunLogger`; this is purely
    the terminal-facing summary.

    Two rendering modes, chosen from whether ``stream`` is a real terminal:

    * **TTY** — a true in-place bar redrawn on the same line via ``\\r``.
    * **Non-TTY** (a SLURM ``.out`` file, a pipe, CI) — one short line *per
      completed receptor*, so the captured stdout stays readable (no thousands
      of carriage returns) and greppable.
    """

    _BAR_WIDTH = 24

    def __init__(
        self,
        total: int,
        *,
        stream: Optional[TextIO] = None,
        color: Optional[bool] = None,
        label: str = "folding",
    ) -> None:
        self.total = max(0, int(total))
        self._explicit_stream = stream
        self.color = _supports_color(self._stream) if color is None else color
        try:
            self._tty = bool(self._stream.isatty())
        except Exception:
            self._tty = False
        self._verb = label
        self._done = 0
        self._rendered = False

    @property
    def _stream(self) -> TextIO:
        return self._explicit_stream if self._explicit_stream is not None else sys.stdout

    def _dim(self, text: str) -> str:
        return f"{_DIM}{text}{_RESET}" if self.color else text

    def update(self, done: int, *, total: "Optional[int]" = None, label: str = "") -> None:
        """Advance the bar to ``done`` completed items (of ``total``)."""
        if total is not None and total > 0:
            self.total = total
        self._done = max(self._done, int(done))
        n, tot = self._done, self.total
        suffix = f" {label}" if label else ""
        if self._tty:
            filled = 0 if tot <= 0 else int(self._BAR_WIDTH * min(n, tot) / tot)
            bar = "█" * filled + "·" * (self._BAR_WIDTH - filled)
            count = f"{n}/{tot}" if tot else f"{n}"
            text = f"  {self._verb} [{bar}] {count}{suffix}"
            # Pad to clear any longer previous line, then return to col 0.
            self._stream.write("\r" + text.ljust(72)[:120])
            self._stream.flush()
            self._rendered = True
        else:
            count = f"{n}/{tot}" if tot else f"{n}"
            self._stream.write(f"  {self._verb} [{count}]{suffix}\n")
            self._stream.flush()

    def finish(self, summary: str = "") -> None:
        """Close out the bar (newline on a TTY) and optionally print a summary."""
        if self._tty and self._rendered:
            self._stream.write("\n")
            self._stream.flush()
        if summary:
            self._stream.write(f"  {self._dim(summary)}\n")
            self._stream.flush()


def _supports_color(stream: TextIO) -> bool:
    """Whether ANSI colour should be written to ``stream``.

    Honours the de-facto ``NO_COLOR`` / ``FORCE_COLOR`` conventions, then falls
    back to "is this an interactive TTY?".
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class Phase:
    """Handle for one in-progress step; finish it with :meth:`done` / :meth:`fail`."""

    def __init__(self, parent: "PipelineProgress", start: float) -> None:
        self._parent = parent
        self._start = start
        self.finished = False

    def info(self, text: str) -> None:
        """Emit an indented sub-line under the step header (e.g. a tool path)."""
        self._parent._emit(self._parent._dim(f"    ↳ {text}"))

    def done(self, summary: Optional[str] = None, *, target: "object | None" = None) -> None:
        """Mark the step succeeded, printing ``✓ <elapsed> · <summary>``."""
        self.finished = True
        elapsed = format_duration(time.monotonic() - self._start)
        line = (
            f"  {self._parent._green('✓')} {self._parent._dim(elapsed)} "
            f"· {summary or 'done'}"
        )
        self._parent._emit(line)
        if target is not None:
            self._parent._emit(self._parent._dim(f"     → {target}"))

    def fail(self, summary: Optional[str] = None) -> None:
        """Mark the step failed, printing ``✗ <elapsed> · <summary>``."""
        self.finished = True
        elapsed = format_duration(time.monotonic() - self._start)
        self._parent._emit(
            f"  {self._parent._red('✗')} {self._parent._dim(elapsed)} "
            f"· {summary or 'failed'}"
        )


class PipelineProgress:
    """Sequential step reporter for a fixed-size pipeline.

    Parameters
    ----------
    total_steps
        How many *numbered* steps the run has (used for the ``[i/N]`` tag).
    stream
        Where to write. Defaults to the live ``sys.stdout`` at emit time, so it
        composes with pytest's ``capsys`` and stdout redirection.
    color
        Force colour on/off. ``None`` (default) auto-detects from the stream.
    """

    def __init__(
        self,
        total_steps: int,
        *,
        stream: Optional[TextIO] = None,
        color: Optional[bool] = None,
        logger: "Optional[RunLogger]" = None,
    ) -> None:
        self.total = total_steps
        self._explicit_stream = stream
        self.color = _supports_color(self._stream) if color is None else color
        self._n = 0
        self._t0 = time.monotonic()
        self.logger = logger

    def attach_logger(self, logger: "Optional[RunLogger]") -> None:
        """Attach a :class:`RunLogger` after construction.

        ``predict`` builds the reporter (and prints its banner) before the
        output directory that holds the log file exists, so the logger is
        wired in once :func:`_run_prepare` has created that directory.
        """
        self.logger = logger

    # -- stream + styling -------------------------------------------------
    @property
    def _stream(self) -> TextIO:
        return self._explicit_stream if self._explicit_stream is not None else sys.stdout

    def _wrap(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self.color else text

    def _bold(self, text: str) -> str:
        return self._wrap(text, _BOLD)

    def _dim(self, text: str) -> str:
        return self._wrap(text, _DIM)

    def _green(self, text: str) -> str:
        return self._wrap(text, _GREEN)

    def _cyan(self, text: str) -> str:
        return self._wrap(text, _CYAN)

    def _red(self, text: str) -> str:
        return self._wrap(text, _RED)

    def _emit(self, line: str = "") -> None:
        print(line, file=self._stream)
        if self.logger is not None:
            self.logger.write_line(_strip_ansi(line))

    def note(self, text: str) -> None:
        """Emit an un-styled line to both the terminal and the run log.

        Used for the free-standing status/hint lines that aren't a step header
        or summary (e.g. the truncation notice, the "running the discovered
        colabfold_batch" line, or a failure hint) so they land in the log too.
        """
        self._emit(text)

    # -- public API -------------------------------------------------------
    def banner(self, title: str, subtitle: Optional[str] = None) -> None:
        """Print the run header. Estimates are flagged as rough up front."""
        self._emit()
        self._emit(self._bold(title))
        if subtitle:
            self._emit(self._dim(subtitle))
        self._emit(
            self._dim("Step time estimates are rough and scale with input size + hardware.")
        )
        self._emit(self._dim("─" * _RULE_WIDTH))

    def start(self, label: str, *, estimate: str, numbered: bool = True) -> Phase:
        """Print a step header and return a :class:`Phase` to finish.

        ``numbered=True`` consumes the next ``[i/N]`` tag; ``False`` is for
        interstitial phases (folding, inference) that aren't one of the N
        counted preparation steps.
        """
        self._emit()
        if numbered:
            self._n += 1
            tag = self._bold(f"[{self._n}/{self.total}] ")
        else:
            tag = ""
        self._emit(
            f"{self._cyan('▶')} {tag}{label}  {self._dim('· est. ' + estimate)}"
        )
        return Phase(self, time.monotonic())

    def complete(
        self,
        message: str,
        *,
        outputs: "Optional[List[Tuple[str, object]]]" = None,
    ) -> None:
        """Print the closing summary with the total wall-clock time."""
        total = format_duration(time.monotonic() - self._t0)
        self._emit()
        self._emit(self._dim("─" * _RULE_WIDTH))
        self._emit(
            f"{self._green('✓')} {self._bold(message)}  {self._dim('· total ' + total)}"
        )
        if outputs:
            width = max(len(label) for label, _ in outputs)
            for label, value in outputs:
                self._emit(f"  {label.ljust(width)}  {value}")

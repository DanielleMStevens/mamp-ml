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

import os
import sys
import time
from typing import List, Optional, TextIO, Tuple

__all__ = ["PipelineProgress", "Phase", "format_duration"]

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_RED = "\033[31m"

_RULE_WIDTH = 62


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
    ) -> None:
        self.total = total_steps
        self._explicit_stream = stream
        self.color = _supports_color(self._stream) if color is None else color
        self._n = 0
        self._t0 = time.monotonic()

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

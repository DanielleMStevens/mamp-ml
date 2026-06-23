"""Eye-friendly, timed progress reporting for the mamp-ml CLI pipeline.

Pure standard-library (no ``rich`` / ``tqdm`` dependency). The output is
designed to degrade gracefully:

* Glyphs are plain Unicode (``▶ ✓ ✗ ↳ →``), which render on any modern
  terminal and stay readable even where they don't.
* ANSI colour is emitted **only** when writing to a real TTY that hasn't opted
  out via ``NO_COLOR`` (or forced it on via ``FORCE_COLOR``), so piping the run
  to a file or a CI log stays clean and greppable.

On a TTY the whole pipeline shares a single sticky progress bar pinned to the
bottom (``[████░░░░] 4/8 · B-factor winding analysis``); each finished stage
prints a one-line ``✓`` confirmation above it, and the long folding/inference
stages update the bar's trailing detail in place. Off a TTY it degrades to
plain ``▶``/``✓`` lines. See :class:`PipelineProgress`.
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
    """Handle for one in-progress stage; finish it with :meth:`done` / :meth:`fail`."""

    def __init__(self, parent: "PipelineProgress", start: float, label: str) -> None:
        self._parent = parent
        self._start = start
        self.label = label
        self.finished = False

    def info(self, text: str) -> None:
        """Emit an indented sub-line above the bar (e.g. a discovered tool path)."""
        self._parent._line(self._parent._dim(f"    ↳ {text}"))

    def detail(self, text: str) -> None:
        """Update the live sub-progress shown inside the bar for this stage.

        Used by the long folding stage to show ``folding 12/58 · <name>`` while
        the overall bar stays on the same stage.
        """
        self._parent.set_detail(text)

    def done(self, summary: Optional[str] = None, *, target: "object | None" = None) -> None:
        """Mark the stage succeeded: print a ``✓`` line and advance the bar."""
        self.finished = True
        self._parent._stage_finished(
            self.label, summary or "done", target=target,
            seconds=time.monotonic() - self._start, ok=True,
        )

    def fail(self, summary: Optional[str] = None) -> None:
        """Mark the stage failed: print a ``✗`` line and stop the bar."""
        self.finished = True
        self._parent._stage_finished(
            self.label, summary or "failed", target=None,
            seconds=time.monotonic() - self._start, ok=False,
        )


class PipelineProgress:
    """Whole-pipeline progress reporter: one sticky bar + a ``✓`` line per stage.

    On a TTY this keeps a single in-place progress bar pinned to the bottom of
    the terminal — ``[████░░░░] 4/8 · B-factor winding analysis`` — and prints a
    one-line ``✓`` confirmation for each finished stage *above* it. The long
    folding/inference stages update the bar's trailing detail in place (e.g.
    ``folding 12/58 · Solanum… (1042 aa)``). Off a TTY (a SLURM ``.out`` file, a
    pipe, CI) it degrades to plain ``▶``/``✓`` lines with no carriage returns.

    Parameters
    ----------
    total_steps
        Total number of stages in the run (the bar denominator). e.g. 8 for
        ``predict`` (FASTA, fold, LRR annotation, LRR-domain FASTA, B-factor,
        test-data, chemical features, inference); 7 for standalone ``prepare``.
    stream
        Where to write. Defaults to the live ``sys.stdout`` at emit time.
    color
        Force colour on/off. ``None`` (default) auto-detects from the stream.
    logger
        Optional :class:`RunLogger`; every line (and stage event) is mirrored
        there, ANSI-stripped, so the run log stays a complete transcript.
    """

    _BAR_WIDTH = 24

    def __init__(
        self,
        total_steps: int,
        *,
        stream: Optional[TextIO] = None,
        color: Optional[bool] = None,
        logger: "Optional[RunLogger]" = None,
    ) -> None:
        self.total = max(0, int(total_steps))
        self._explicit_stream = stream
        self.color = _supports_color(self._stream) if color is None else color
        try:
            self._tty = bool(self._stream.isatty())
        except Exception:
            self._tty = False
        self._t0 = time.monotonic()
        self.logger = logger
        self._completed = 0      # fully-finished stages
        self._active = False     # a stage is currently running
        self._running = False    # between the first start() and finish()
        self._bar_shown = False  # a bar line is currently drawn on the TTY
        self._current_label = ""
        self._detail = ""
        self._timings: List[Tuple[str, float, bool]] = []  # (label, seconds, ok)

    def attach_logger(self, logger: "Optional[RunLogger]") -> None:
        """Attach a :class:`RunLogger` after construction (predict prints its
        banner before the output dir that holds the log file exists)."""
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

    # -- bar rendering ----------------------------------------------------
    def _bar_text(self) -> str:
        tot = self.total
        # Fill reflects *completed* stages (honest); the count shows the stage
        # currently being worked on (completed + 1 while a stage is active).
        filled = int(self._BAR_WIDTH * (self._completed / tot)) if tot else 0
        filled = max(0, min(self._BAR_WIDTH, filled))
        bar = "█" * filled + "·" * (self._BAR_WIDTH - filled)
        shown = min(self._completed + (1 if self._active else 0), tot) if tot else self._completed
        count = f"{shown}/{tot}" if tot else f"{shown}"
        detail = f" · {self._detail}" if self._detail else ""
        return f"  [{bar}] {count} · {self._current_label}{detail}"

    def _render_bar(self) -> None:
        if not self._tty or self.total <= 0:
            return
        import shutil

        cols = shutil.get_terminal_size((80, 24)).columns
        text = self._bar_text()
        if len(text) > cols - 1:
            text = text[: cols - 1]
        self._stream.write("\r" + text + "\033[K")
        self._stream.flush()
        self._bar_shown = True

    # -- output primitives ------------------------------------------------
    def _log(self, text: str) -> None:
        if self.logger is not None:
            self.logger.write_line(_strip_ansi(text))

    def _term_line(self, text: str) -> None:
        """Print a full line to the terminal, keeping the sticky bar below it."""
        if self._tty and self._bar_shown:
            # Clear the bar line, print the message, then redraw the bar.
            self._stream.write("\r\033[K" + text + "\n")
            if self._running:
                self._render_bar()
            else:
                self._bar_shown = False
            self._stream.flush()
        else:
            print(text, file=self._stream)

    def _line(self, text: str = "") -> None:
        """Emit a line to both the run log and the terminal (above the bar)."""
        self._log(text)
        self._term_line(text)

    def note(self, text: str) -> None:
        """Emit a free-standing status/hint line to the terminal and run log."""
        self._line(text)

    # also expose the old name used in a couple of call sites / tests
    def _emit(self, line: str = "") -> None:
        self._line(line)

    # -- public API -------------------------------------------------------
    def banner(self, title: str, subtitle: Optional[str] = None) -> None:
        """Print the run header (before any stage / bar)."""
        self._line()
        self._line(self._bold(title))
        if subtitle:
            self._line(self._dim(subtitle))
        self._line(self._dim("─" * _RULE_WIDTH))

    def start(self, label: str, *, estimate: "Optional[str]" = None, numbered: bool = True) -> Phase:
        """Begin a stage: set it as current and (re)draw the bar.

        ``estimate`` / ``numbered`` are accepted for backwards compatibility but
        no longer shown — every stage advances the single combined bar.
        """
        self._current_label = label
        self._detail = ""
        self._active = True
        self._running = True
        self._log("▶ " + label + (f"  (est. {estimate})" if estimate else ""))
        if self._tty:
            self._render_bar()
        else:
            print(f"{self._cyan('▶')} {label}", file=self._stream)
        return Phase(self, time.monotonic(), label)

    def set_detail(self, text: str) -> None:
        """Update the live sub-progress inside the bar (folding receptor, …)."""
        self._detail = text
        if self._tty:
            self._render_bar()
        else:
            line = f"    {text}"
            print(line, file=self._stream)
            self._log(line)

    def _stage_finished(
        self,
        label: str,
        summary: str,
        *,
        target: "object | None",
        seconds: float,
        ok: bool,
    ) -> None:
        elapsed = format_duration(seconds)
        self._timings.append((label, seconds, ok))
        glyph = self._green("✓") if ok else self._red("✗")
        plain_glyph = "✓" if ok else "✗"
        self._log(f"  {plain_glyph} {label} · {summary} · {elapsed}")
        if target is not None:
            self._log(f"     → {target}")
        if ok:
            self._completed += 1
        self._active = False
        self._detail = ""
        if not ok:
            # A failed stage ends the run; stop the bar so error hints print
            # cleanly underneath.
            self._running = False
        # The per-stage runtime is shown (dimmed) on the terminal too.
        self._term_line(
            f"  {glyph} {label} {self._dim('· ' + summary + ' · ' + elapsed)}"
        )
        if target is not None:
            self._term_line(self._dim(f"     → {target}"))

    def finish(self) -> None:
        """Clear the sticky bar (call before the closing summary / on exit)."""
        if self._tty and self._bar_shown:
            self._stream.write("\r\033[K")
            self._stream.flush()
        self._bar_shown = False
        self._running = False

    def complete(
        self,
        message: str,
        *,
        outputs: "Optional[List[Tuple[str, object]]]" = None,
    ) -> None:
        """Clear the bar and print the closing summary with total wall time."""
        self.finish()
        total = format_duration(time.monotonic() - self._t0)
        self._line()
        self._line(self._dim("─" * _RULE_WIDTH))
        self._line(
            f"{self._green('✓')} {self._bold(message)}  {self._dim('· total ' + total)}"
        )
        if outputs:
            width = max(len(label) for label, _ in outputs)
            for label, value in outputs:
                self._line(f"  {label.ljust(width)}  {value}")
        self._write_timings(total)

    def _write_timings(self, total: str) -> None:
        """Append a per-step runtime breakdown to the run log (log-only)."""
        if not self._timings or self.logger is None:
            return
        width = max(len(label) for label, _, _ in self._timings)
        width = max(width, len("total"))
        self._log("")
        self._log("Step timings")
        for label, seconds, ok in self._timings:
            mark = "" if ok else "  (failed)"
            self._log(f"  {label.ljust(width)}  {format_duration(seconds)}{mark}")
        self._log(f"  {'total'.ljust(width)}  {total}")

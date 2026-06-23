"""Tests for the run-log + fold progress-bar plumbing (mamp_ml.progress).

Covers:
* :class:`RunLogger` — header content (command + version), line appending,
  and best-effort survival when the handle goes away.
* :class:`FoldProgressBar` — non-TTY (one line per completed receptor) vs.
  TTY (in-place ``\\r``) rendering.
* :data:`COLABFOLD_QUERY_RE` — parsing ColabFold's per-receptor progress line.
* :class:`PipelineProgress` teeing emitted lines into an attached logger,
  with ANSI stripped.
"""

from __future__ import annotations

import io

from mamp_ml.fold.colabfold import COLABFOLD_QUERY_RE
from mamp_ml.progress import FoldProgressBar, PipelineProgress, RunLogger


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------


def test_runlogger_header_records_command_and_version(tmp_path) -> None:
    path = tmp_path / "run.log"
    logger = RunLogger(
        path,
        command="mamp-ml predict input.xlsx --device cuda",
        version="0.2.4",
        context=[("structure", "colabfold")],
    )
    logger.write_line("a backend line")
    logger.close()

    text = path.read_text(encoding="utf-8")
    assert "mamp-ml run log" in text
    # Labels are padded for alignment, so match label + value loosely.
    assert "command" in text
    assert "mamp-ml predict input.xlsx --device cuda" in text
    assert "mamp-ml 0.2.4" in text
    assert "colabfold" in text
    assert "a backend line" in text


def test_runlogger_write_is_durable_each_line(tmp_path) -> None:
    """Lines are flushed immediately, so a crash mid-run still leaves them on
    disk (we read without closing)."""
    path = tmp_path / "run.log"
    logger = RunLogger(path, command="mamp-ml prepare x.xlsx", version="0.2.4")
    logger.write_line("first")
    logger.write_line("second")
    # No close() — simulate reading the log of a still-running / crashed process.
    text = path.read_text(encoding="utf-8")
    assert "first" in text and "second" in text
    logger.close()


def test_runlogger_close_is_idempotent_and_swallows_post_close_writes(tmp_path) -> None:
    path = tmp_path / "run.log"
    logger = RunLogger(path, command="mamp-ml prepare x.xlsx", version="0.2.4")
    logger.close()
    logger.close()  # no raise
    logger.write_line("after close")  # no raise; silently dropped


# ---------------------------------------------------------------------------
# COLABFOLD_QUERY_RE
# ---------------------------------------------------------------------------


def test_colabfold_query_regex_parses_progress_line() -> None:
    line = (
        "2026-06-19 02:17:37,427 Query 58/58: "
        "Vitis_vinifera_g224180.t01_VCORE (length 1617)"
    )
    m = COLABFOLD_QUERY_RE.search(line)
    assert m is not None
    assert m.group("i") == "58"
    assert m.group("n") == "58"
    assert m.group("name") == "Vitis_vinifera_g224180.t01_VCORE"
    assert m.group("length") == "1617"


def test_colabfold_query_regex_ignores_other_lines() -> None:
    assert COLABFOLD_QUERY_RE.search("2026-06-19 02:10:09 Sleeping for 5s.") is None
    assert COLABFOLD_QUERY_RE.search("Padding length to 1343") is None


# ---------------------------------------------------------------------------
# FoldProgressBar
# ---------------------------------------------------------------------------


def test_fold_bar_non_tty_emits_one_line_per_update() -> None:
    buf = io.StringIO()  # StringIO.isatty() is False -> non-TTY mode
    bar = FoldProgressBar(58, stream=buf, color=False)
    bar.update(1, total=58, label="recA (900 aa)")
    bar.update(2, total=58, label="recB (1200 aa)")
    bar.finish()
    out = buf.getvalue()
    assert "folding [1/58] recA (900 aa)" in out
    assert "folding [2/58] recB (1200 aa)" in out
    # Non-TTY mode must not emit carriage returns (keeps SLURM .out readable).
    assert "\r" not in out


def test_fold_bar_tty_renders_in_place_with_carriage_return() -> None:
    class _TTY(io.StringIO):
        def isatty(self) -> bool:  # noqa: D401 - trivial
            return True

    buf = _TTY()
    bar = FoldProgressBar(2, stream=buf, color=False)
    bar.update(1, total=2, label="recA")
    bar.finish()
    out = buf.getvalue()
    assert "\r" in out  # in-place redraw
    assert "1/2" in out
    assert out.endswith("\n")  # finish() terminates the line


# ---------------------------------------------------------------------------
# PipelineProgress -> RunLogger teeing
# ---------------------------------------------------------------------------


def test_complete_writes_per_step_timings_to_log(tmp_path) -> None:
    log_path = tmp_path / "run.log"
    logger = RunLogger(log_path, command="mamp-ml predict x.xlsx", version="0.2.13")
    term = io.StringIO()
    p = PipelineProgress(2, stream=term, color=False, logger=logger)
    p.start("First").done("a")
    p.start("Second").done("b")
    p.complete("Done", outputs=[("Output", "out/")])
    logger.close()

    text = log_path.read_text(encoding="utf-8")
    # Each ✓ line carries the step's runtime, and a consolidated block lists them.
    assert "✓ First · a · " in text
    assert "Step timings" in text
    assert "First" in text and "Second" in text
    assert "total" in text


def test_progress_tees_lines_into_logger_without_ansi(tmp_path) -> None:
    log_path = tmp_path / "run.log"
    logger = RunLogger(log_path, command="mamp-ml predict x.xlsx", version="0.2.4")
    term = io.StringIO()
    p = PipelineProgress(3, stream=term, color=True, logger=logger)
    p.start("First", estimate="<5s").done("did a thing")
    p.note("a free-standing note")
    logger.close()

    log_text = log_path.read_text(encoding="utf-8")
    # The styled terminal copy carries ANSI; the logged copy is stripped.
    assert "\033[" in term.getvalue()
    assert "\033[" not in log_text
    assert "First" in log_text
    assert "did a thing" in log_text
    assert "a free-standing note" in log_text

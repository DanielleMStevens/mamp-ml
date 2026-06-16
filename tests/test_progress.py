"""Tests for the CLI progress reporter (mamp_ml.progress).

These cover the pure formatting + the rendered line shapes (with colour forced
off for deterministic assertions) and the colour-gating policy.
"""

from __future__ import annotations

import io

import pytest

from mamp_ml.progress import PipelineProgress, format_duration


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0.0, "0.0s"),
        (0.42, "0.4s"),
        (14.25, "14.2s"),
        (59.9, "59.9s"),
        (60, "1m 00s"),
        (75, "1m 15s"),
        (3599, "59m 59s"),
        (3600, "1h 00m"),
        (7320, "2h 02m"),
        (-5, "0.0s"),  # clamped
    ],
)
def test_format_duration(seconds, expected) -> None:
    assert format_duration(seconds) == expected


# ---------------------------------------------------------------------------
# rendered line shapes (colour off)
# ---------------------------------------------------------------------------


def _render(fn) -> str:
    buf = io.StringIO()
    p = PipelineProgress(3, stream=buf, color=False)
    fn(p)
    return buf.getvalue()


def test_banner_includes_title_subtitle_and_estimate_disclaimer() -> None:
    out = _render(lambda p: p.banner("mamp-ml predict", "input: x.xlsx"))
    assert "mamp-ml predict" in out
    assert "input: x.xlsx" in out
    assert "estimates are rough" in out
    assert "─" in out  # rule


def test_numbered_steps_increment_tag_and_finish_with_check() -> None:
    def script(p: PipelineProgress) -> None:
        s1 = p.start("First", estimate="<5s")
        s1.done("did a thing", target="out/a.txt")
        s2 = p.start("Second", estimate="~1m")
        s2.done("did another")

    out = _render(script)
    assert "[1/3] First" in out
    assert "[2/3] Second" in out
    assert "est. <5s" in out
    assert "✓" in out
    assert "· did a thing" in out
    assert "→ out/a.txt" in out


def test_unnumbered_phase_does_not_consume_a_tag() -> None:
    def script(p: PipelineProgress) -> None:
        p.start("Fold receptors", estimate="~5m", numbered=False).done("done")
        p.start("First numbered", estimate="<5s").done("ok")

    out = _render(script)
    # The interstitial phase has no [i/N] tag; the first numbered step is still [1/3].
    assert "Fold receptors" in out
    assert "[1/3] First numbered" in out
    assert "[0/3]" not in out


def test_phase_fail_marks_with_cross() -> None:
    out = _render(lambda p: p.start("Risky", estimate="?").fail("boom"))
    assert "✗" in out
    assert "· boom" in out


def test_complete_prints_total_and_aligned_outputs() -> None:
    def script(p: PipelineProgress) -> None:
        p.complete(
            "Prediction complete",
            outputs=[("Predictions", "out/p.csv"), ("LRR annotation plots", "out/plots/")],
        )

    out = _render(script)
    assert "Prediction complete" in out
    assert "· total" in out
    assert "Predictions" in out
    assert "out/p.csv" in out
    assert "LRR annotation plots" in out
    # Shorter label is padded to align with the longer one.
    assert "Predictions         " in out  # ljust to len("LRR annotation plots")


# ---------------------------------------------------------------------------
# colour gating
# ---------------------------------------------------------------------------


def test_color_off_emits_no_ansi() -> None:
    buf = io.StringIO()
    p = PipelineProgress(1, stream=buf, color=False)
    p.start("x", estimate="<5s").done("y")
    assert "\033[" not in buf.getvalue()


def test_color_on_emits_ansi() -> None:
    buf = io.StringIO()
    p = PipelineProgress(1, stream=buf, color=True)
    p.start("x", estimate="<5s").done("y")
    assert "\033[" in buf.getvalue()


def test_color_autodetect_off_for_non_tty(monkeypatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    buf = io.StringIO()  # StringIO has no isatty()->True
    p = PipelineProgress(1, stream=buf)  # color=None -> autodetect
    assert p.color is False


def test_force_color_env_overrides_non_tty(monkeypatch) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    buf = io.StringIO()
    p = PipelineProgress(1, stream=buf)
    assert p.color is True


def test_no_color_env_wins(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")  # NO_COLOR takes precedence
    buf = io.StringIO()
    p = PipelineProgress(1, stream=buf)
    assert p.color is False

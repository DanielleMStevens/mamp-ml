"""Tests for the actionable disk-quota / out-of-space handling in predict.

Regression: on an HPC node with a full HOME quota, `mamp-ml predict` crashed
with a 20-line traceback ending in ``OSError: [Errno 122] Disk quota exceeded``
when HuggingFace tried to cache the ESM-2 weights under ~/.cache/huggingface.
predict now catches that and prints a clear "point HF_HOME at scratch" message.
"""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from mamp_ml.__main__ import _is_disk_full_error


# ---------------------------------------------------------------------------
# _is_disk_full_error
# ---------------------------------------------------------------------------


def test_is_disk_full_error_by_errno() -> None:
    assert _is_disk_full_error(OSError(errno.ENOSPC, "No space left on device"))
    assert _is_disk_full_error(OSError(errno.EDQUOT, "Disk quota exceeded"))


def test_is_disk_full_error_by_message_fallback() -> None:
    # No clean errno, but the message gives it away.
    assert _is_disk_full_error(OSError("Disk quota exceeded: /home/.cache/..."))
    assert _is_disk_full_error(OSError("No space left on device"))


def test_is_disk_full_error_rejects_unrelated_oserrors() -> None:
    assert not _is_disk_full_error(OSError(errno.ENOENT, "No such file or directory"))
    assert not _is_disk_full_error(OSError("connection reset by peer"))


# ---------------------------------------------------------------------------
# predict surfaces an actionable message instead of a traceback
# ---------------------------------------------------------------------------


def test_predict_reports_disk_quota_actionably(
    tmp_path: Path, example_xlsx: Path, monkeypatch, capsys
) -> None:
    """When the inference step hits a quota error, predict returns rc 5 and
    prints HF_HOME guidance rather than letting the OSError escape."""
    import mamp_ml.__main__ as cli

    out_dir = tmp_path / "inter"
    out_dir.mkdir()
    # _run_predict checks for ready_test_data.csv after prepare; create it.
    (out_dir / "ready_test_data.csv").write_text("a,b\n1,2\n")
    # A weights file that exists so weight resolution passes.
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"\x00")

    # Stub prepare so we go straight to the inference step.
    monkeypatch.setattr(cli, "_run_prepare", lambda args, *, progress=None: 0)

    # Make the inference call raise an over-quota OSError, like HF caching does.
    from mamp_ml import train

    def boom(train_args):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(train, "main", boom)

    rc = cli.main(
        [
            "predict",
            str(example_xlsx),
            "--out-dir",
            str(out_dir),
            "--weights",
            str(weights),
            "--device",
            "cpu",
        ]
    )
    assert rc == 5
    out = capsys.readouterr().out
    assert "HF_HOME" in out
    assert "out of space" in out or "quota" in out.lower()


def test_predict_reraises_unrelated_oserror(
    tmp_path: Path, example_xlsx: Path, monkeypatch
) -> None:
    """A non-quota OSError from inference must NOT be swallowed as a disk issue."""
    import mamp_ml.__main__ as cli

    out_dir = tmp_path / "inter"
    out_dir.mkdir()
    (out_dir / "ready_test_data.csv").write_text("a,b\n1,2\n")
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"\x00")

    monkeypatch.setattr(cli, "_run_prepare", lambda args, *, progress=None: 0)

    from mamp_ml import train

    def boom(train_args):
        raise OSError(errno.ENOENT, "No such file or directory")

    monkeypatch.setattr(train, "main", boom)

    with pytest.raises(OSError, match="No such file"):
        cli.main(
            [
                "predict",
                str(example_xlsx),
                "--out-dir",
                str(out_dir),
                "--weights",
                str(weights),
                "--device",
                "cpu",
            ]
        )

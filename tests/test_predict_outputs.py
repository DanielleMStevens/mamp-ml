"""Where `mamp-ml predict` puts its outputs.

Default (`--keep default`): the two deliverables — predictions.csv and
lrr_annotation_plots/ — are promoted to the directory the user invoked from,
and the intermediate_files/ scratch dir is removed, so outputs aren't buried.
`--keep all`: everything stays in intermediate_files/ for debugging / reuse.

The heavy stages are stubbed; these tests are purely about output placement.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _stub_pipeline(monkeypatch, *, with_plots=True):
    """Stub _run_prepare (builds the scratch dir) and train.main (writes
    predictions.csv into the cwd, which predict has chdir'd to out_dir)."""
    import mamp_ml.__main__ as cli
    from mamp_ml import train

    def fake_prepare(args, *, progress=None):
        od = Path(args.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        (od / "ready_test_data.csv").write_text("a,b\n1,2\n")
        (od / "test_data.csv").write_text("intermediate\n")  # should not survive default
        (od / "receptor_full_length.fasta").write_text(">r\nACDE\n")
        if with_plots:
            plots = od / "lrr_annotation_plots"
            plots.mkdir()
            (plots / "receptor_plot.png").write_bytes(b"\x89PNG")
        return 0

    def fake_train_main(train_args):
        # The model writes predictions.csv to cwd; predict has chdir'd into out_dir.
        Path("predictions.csv").write_text("Header_Name,prediction\nR1,0.91\n")

    monkeypatch.setattr(cli, "_run_prepare", fake_prepare)
    monkeypatch.setattr(train, "main", fake_train_main)


def _weights(tmp_path: Path) -> Path:
    w = tmp_path / "weights.pth"
    w.write_bytes(b"\x00")
    return w


def test_predict_default_bundles_outputs_in_named_folder(
    tmp_path, example_xlsx, monkeypatch, capsys
) -> None:
    from mamp_ml.__main__ import main as cli_main

    monkeypatch.chdir(tmp_path)  # invocation dir = tmp_path
    _stub_pipeline(monkeypatch)

    rc = cli_main(
        [
            "predict",
            str(example_xlsx),
            "--device",
            "cpu",
            "--weights",
            str(_weights(tmp_path)),
            "--output-name",
            "myrun",
        ]
    )
    assert rc == 0

    # Deliverables land in a labeled folder, predictions.csv keeps its name...
    assert (tmp_path / "myrun" / "predictions.csv").is_file()
    assert (tmp_path / "myrun" / "lrr_annotation_plots" / "receptor_plot.png").is_file()
    # ...and the intermediate_files working dir is gone entirely.
    assert not (tmp_path / "intermediate_files").exists()

    out = capsys.readouterr().out
    assert "Output folder" in out


def test_predict_default_output_folder_is_unique_timestamped(
    tmp_path, example_xlsx, monkeypatch
) -> None:
    """With no --output-name, outputs go into a unique `output_<timestamp>/`."""
    import re

    from mamp_ml.__main__ import main as cli_main

    monkeypatch.chdir(tmp_path)
    _stub_pipeline(monkeypatch)

    rc = cli_main(
        ["predict", str(example_xlsx), "--device", "cpu", "--weights", str(_weights(tmp_path))]
    )
    assert rc == 0

    folders = [p for p in tmp_path.glob("output_*") if p.is_dir()]
    assert len(folders) == 1, f"expected one timestamped output folder, got {folders}"
    assert re.match(r"output_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", folders[0].name)
    assert (folders[0] / "predictions.csv").is_file()
    assert (folders[0] / "lrr_annotation_plots").is_dir()
    assert not (tmp_path / "intermediate_files").exists()


def test_resolve_output_dirname_uses_flag() -> None:
    import types

    from mamp_ml.__main__ import _resolve_output_dirname

    assert _resolve_output_dirname(types.SimpleNamespace(output_name="run1")) == "run1"


def test_resolve_output_dirname_default_is_timestamped() -> None:
    import re
    import types

    from mamp_ml.__main__ import _resolve_output_dirname

    name = _resolve_output_dirname(types.SimpleNamespace(output_name=None))
    assert re.match(r"output_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", name)


def test_predict_subparser_accepts_output_name(example_xlsx) -> None:
    from mamp_ml.__main__ import _build_parser

    parser = _build_parser()
    parsed = parser.parse_args(["predict", str(example_xlsx), "--output-name", "foo"])
    assert parsed.output_name == "foo"
    assert parser.parse_args(["predict", str(example_xlsx)]).output_name is None


def test_predict_keep_all_makes_output_folder_and_keeps_intermediates(
    tmp_path, example_xlsx, monkeypatch
) -> None:
    """--keep all still produces the labeled output folder (deliverables moved
    there); it only additionally retains the intermediate_files/ working dir."""
    from mamp_ml.__main__ import main as cli_main

    monkeypatch.chdir(tmp_path)
    _stub_pipeline(monkeypatch)

    rc = cli_main(
        [
            "predict",
            str(example_xlsx),
            "--device",
            "cpu",
            "--weights",
            str(_weights(tmp_path)),
            "--output-name",
            "myrun",
            "--keep",
            "all",
        ]
    )
    assert rc == 0

    # Deliverables are in the labeled output folder, as in default mode.
    assert (tmp_path / "myrun" / "predictions.csv").is_file()
    assert (tmp_path / "myrun" / "lrr_annotation_plots" / "receptor_plot.png").is_file()

    # intermediate_files/ is retained, holding the non-deliverable intermediates...
    inter = tmp_path / "intermediate_files"
    assert inter.is_dir()
    assert (inter / "test_data.csv").is_file()
    # ...but the deliverables were MOVED out of it, not left behind.
    assert not (inter / "predictions.csv").exists()
    assert not (inter / "lrr_annotation_plots").exists()


def test_predict_default_handles_custom_out_dir(
    tmp_path, example_xlsx, monkeypatch
) -> None:
    """A custom --out-dir is still removed after its deliverables are promoted."""
    from mamp_ml.__main__ import main as cli_main

    monkeypatch.chdir(tmp_path)
    _stub_pipeline(monkeypatch)
    scratch = tmp_path / "my_scratch"

    rc = cli_main(
        [
            "predict",
            str(example_xlsx),
            "--device",
            "cpu",
            "--weights",
            str(_weights(tmp_path)),
            "--out-dir",
            str(scratch),
            "--output-name",
            "run",
        ]
    )
    assert rc == 0
    assert (tmp_path / "run" / "predictions.csv").is_file()
    assert not scratch.exists()

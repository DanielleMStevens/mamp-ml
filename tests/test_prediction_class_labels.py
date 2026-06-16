"""Tests for human-readable class labels in predictions.csv.

The model emits interaction classes as 0/1/2 internally; the prediction output
maps them to their names (Immunogenic / Non-Immunogenic / Weakly Immunogenic).
These tests pin the mapping and guard that it stays in sync with the canonical
label encoding used at training time.
"""

from __future__ import annotations

import pytest

# The model module imports torch + transformers at import time.
pytest.importorskip("torch")
pytest.importorskip("transformers")

from mamp_ml.models.esm_positon_weighted import CLASS_NAMES, class_name_for  # noqa: E402


def test_class_names_order_is_the_label_contract() -> None:
    """Index position is the label: 0/1/2 -> the three classes, in order."""
    assert CLASS_NAMES == ["Immunogenic", "Non-Immunogenic", "Weakly Immunogenic"]


def test_class_name_for_maps_each_label() -> None:
    assert class_name_for(0) == "Immunogenic"
    assert class_name_for(1) == "Non-Immunogenic"
    assert class_name_for(2) == "Weakly Immunogenic"


def test_class_name_for_accepts_numpy_and_str_ints() -> None:
    import numpy as np

    assert class_name_for(np.int64(2)) == "Weakly Immunogenic"
    assert class_name_for("1") == "Non-Immunogenic"


def test_class_name_for_passes_through_unknown_labels() -> None:
    # Out-of-range or non-numeric -> stringified, never raises.
    assert class_name_for(7) == "7"
    assert class_name_for("garbage") == "garbage"
    assert class_name_for(None) == "None"


def test_class_names_match_training_label_encoding() -> None:
    """CLASS_NAMES must agree with the dataset's category_to_index, which is
    what mapped the input categories to 0/1/2 during training. If these drift,
    predictions.csv would mislabel the classes."""
    from mamp_ml.datasets.seq_with_receptor_dataset import category_to_index

    for index, name in enumerate(CLASS_NAMES):
        assert category_to_index[name] == index, (
            f"{name} encodes to {category_to_index[name]} at training time but "
            f"CLASS_NAMES puts it at {index}"
        )
    assert len(CLASS_NAMES) == len(category_to_index)

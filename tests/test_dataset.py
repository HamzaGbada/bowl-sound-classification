"""Tests for src/dataset.py — Dataset, splits, class weights, collate."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from src.config import TARGET_CLASSES, NUM_CLASSES, PreprocessingConfig
from src.dataset import compute_class_weights, collate_fn


# ── compute_class_weights ─────────────────────────────────────────────

class TestComputeClassWeights:
    @pytest.mark.parametrize("labels,expected_len", [
        ([0, 0, 0, 1, 1, 2], NUM_CLASSES),
        ([0, 1, 2], NUM_CLASSES),
    ])
    def test_output_length(self, labels: list[int], expected_len: int):
        ds = MagicMock()
        ds.labels = labels
        weights = compute_class_weights(ds)
        assert len(weights) == expected_len

    def test_rare_class_gets_higher_weight(self):
        """Class with fewer samples should have higher weight."""
        ds = MagicMock()
        ds.labels = [0] * 100 + [1] * 100 + [2] * 10  # class 2 is rare
        weights = compute_class_weights(ds)
        assert weights[2] > weights[0]
        assert weights[2] > weights[1]

    def test_balanced_classes_get_equal_weights(self):
        ds = MagicMock()
        ds.labels = [0] * 50 + [1] * 50 + [2] * 50
        weights = compute_class_weights(ds)
        assert abs(weights[0] - weights[1]) < 0.01
        assert abs(weights[1] - weights[2]) < 0.01

    def test_weights_sum_to_num_classes(self):
        ds = MagicMock()
        ds.labels = [0] * 80 + [1] * 70 + [2] * 10
        weights = compute_class_weights(ds)
        assert abs(weights.sum().item() - NUM_CLASSES) < 0.01

    def test_output_dtype(self):
        ds = MagicMock()
        ds.labels = [0, 1, 2]
        weights = compute_class_weights(ds)
        assert weights.dtype == torch.float32

    def test_subset_style_dataset(self):
        """Test with Subset-like structure (dataset.dataset.labels + indices)."""
        inner = MagicMock()
        inner.labels = [0, 0, 1, 1, 2]
        outer = MagicMock(spec=["dataset", "indices"])
        outer.dataset = inner
        outer.indices = [0, 1, 2, 3, 4]
        # Remove labels attr from outer so it falls through
        del outer.labels
        weights = compute_class_weights(outer)
        assert len(weights) == NUM_CLASSES


# ── collate_fn ────────────────────────────────────────────────────────

class TestCollateFn:
    def test_stacks_specs(self):
        batch = [
            (torch.randn(1, 128, 128), 0, {"file_id": "a"}),
            (torch.randn(1, 128, 128), 1, {"file_id": "b"}),
        ]
        specs, labels, metas = collate_fn(batch)
        assert specs.shape == (2, 1, 128, 128)

    def test_labels_dtype(self):
        batch = [
            (torch.randn(1, 128, 128), 0, {}),
            (torch.randn(1, 128, 128), 2, {}),
        ]
        _, labels, _ = collate_fn(batch)
        assert labels.dtype == torch.long
        assert labels.tolist() == [0, 2]

    def test_metadata_preserved(self):
        meta1 = {"file_id": "A", "start": 1.0}
        meta2 = {"file_id": "B", "start": 2.0}
        batch = [
            (torch.randn(1, 128, 128), 0, meta1),
            (torch.randn(1, 128, 128), 1, meta2),
        ]
        _, _, metas = collate_fn(batch)
        assert metas == [meta1, meta2]

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_various_batch_sizes(self, batch_size: int):
        batch = [
            (torch.randn(1, 128, 128), i % NUM_CLASSES, {"idx": i})
            for i in range(batch_size)
        ]
        specs, labels, metas = collate_fn(batch)
        assert specs.shape[0] == batch_size
        assert labels.shape[0] == batch_size
        assert len(metas) == batch_size


# ── PreprocessingConfig integration ───────────────────────────────────

class TestPreprocessingConfigIntegration:
    def test_default_config_has_all_flags_false(self):
        config = PreprocessingConfig()
        d = config.to_dict()
        assert d["use_bandpass"] is False
        assert d["use_normalise"] is False
        assert d["use_augmentation"] is False

    @pytest.mark.parametrize("input_dict", [
        {"use_bandpass": True, "use_normalise": False, "use_augmentation": False},
        {"use_bandpass": True, "use_normalise": True, "use_augmentation": True},
        {},
    ])
    def test_dict_roundtrip(self, input_dict: dict):
        """Configs from JSON should survive dict→dataclass→dict roundtrip."""
        config = PreprocessingConfig.from_dict(input_dict)
        restored = config.to_dict()
        for key in input_dict:
            assert restored[key] == input_dict[key]
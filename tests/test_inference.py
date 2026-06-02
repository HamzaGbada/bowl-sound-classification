"""Tests for app/inference.py — BowelSoundDetector and detection merging."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from app.inference import BowelSoundDetector
from src.config import TARGET_CLASSES
from tests.conftest import make_mock_model


# ── _merge_detections (pure logic, no mocking needed) ─────────────────

class TestMergeDetections:
    def test_empty_input(self):
        assert BowelSoundDetector._merge_detections([]) == []

    def test_single_detection_unchanged(self):
        det = [{"start_s": 1.0, "end_s": 2.0, "class_label": "b", "confidence": 0.9}]
        merged = BowelSoundDetector._merge_detections(det)
        assert len(merged) == 1
        assert merged[0] == det[0]

    def test_consecutive_same_class_merged(self):
        dets = [
            {"start_s": 1.0, "end_s": 2.0, "class_label": "b", "confidence": 0.8},
            {"start_s": 2.0, "end_s": 3.0, "class_label": "b", "confidence": 0.9},
        ]
        merged = BowelSoundDetector._merge_detections(dets)
        assert len(merged) == 1
        assert merged[0]["start_s"] == 1.0
        assert merged[0]["end_s"] == 3.0
        assert merged[0]["confidence"] == 0.9  # max confidence

    def test_different_classes_not_merged(self):
        dets = [
            {"start_s": 1.0, "end_s": 2.0, "class_label": "b", "confidence": 0.9},
            {"start_s": 2.0, "end_s": 3.0, "class_label": "mb", "confidence": 0.8},
        ]
        merged = BowelSoundDetector._merge_detections(dets)
        assert len(merged) == 2

    @pytest.mark.parametrize("gap,expected_count", [
        (0.0, 1),    # no gap → merge
        (0.1, 1),    # small gap → merge
        (0.19, 1),   # just below threshold → merge
        (0.3, 2),    # above threshold → separate
        (1.0, 2),    # large gap → separate
    ])
    def test_gap_threshold(self, gap: float, expected_count: int):
        dets = [
            {"start_s": 1.0, "end_s": 2.0, "class_label": "b", "confidence": 0.9},
            {"start_s": 2.0 + gap, "end_s": 3.0 + gap, "class_label": "b", "confidence": 0.8},
        ]
        merged = BowelSoundDetector._merge_detections(dets, max_gap=0.2)
        assert len(merged) == expected_count

    def test_confidence_takes_maximum(self):
        dets = [
            {"start_s": 1.0, "end_s": 2.0, "class_label": "b", "confidence": 0.7},
            {"start_s": 2.0, "end_s": 3.0, "class_label": "b", "confidence": 0.95},
            {"start_s": 3.0, "end_s": 4.0, "class_label": "b", "confidence": 0.8},
        ]
        merged = BowelSoundDetector._merge_detections(dets)
        assert len(merged) == 1
        assert merged[0]["confidence"] == 0.95

    def test_mixed_merge_and_separate(self):
        dets = [
            {"start_s": 0.0, "end_s": 1.0, "class_label": "b", "confidence": 0.9},
            {"start_s": 1.0, "end_s": 2.0, "class_label": "b", "confidence": 0.8},  # merge with prev
            {"start_s": 5.0, "end_s": 6.0, "class_label": "mb", "confidence": 0.7},  # new (diff class + gap)
            {"start_s": 6.0, "end_s": 7.0, "class_label": "mb", "confidence": 0.85}, # merge with prev
        ]
        merged = BowelSoundDetector._merge_detections(dets)
        assert len(merged) == 2
        assert merged[0]["class_label"] == "b"
        assert merged[0]["end_s"] == 2.0
        assert merged[1]["class_label"] == "mb"
        assert merged[1]["end_s"] == 7.0

    def test_does_not_mutate_input(self):
        dets = [
            {"start_s": 1.0, "end_s": 2.0, "class_label": "b", "confidence": 0.8},
            {"start_s": 2.0, "end_s": 3.0, "class_label": "b", "confidence": 0.9},
        ]
        original_end = dets[0]["end_s"]
        BowelSoundDetector._merge_detections(dets)
        assert dets[0]["end_s"] == original_end  # original not modified


# ── BowelSoundDetector initialisation ─────────────────────────────────

class TestDetectorInit:
    @patch("app.inference.get_model")
    @patch("app.inference.torch.load", return_value={})
    def test_loads_config_and_model(
        self, mock_load: MagicMock, mock_get_model: MagicMock, best_model_config: Path
    ):
        mock_model = make_mock_model()
        mock_get_model.return_value = mock_model

        detector = BowelSoundDetector(config_path=best_model_config)

        assert detector.config["model"] == "resnet_cnn"
        assert detector.config["macro_f1"] == 0.9104
        mock_get_model.assert_called_once_with("resnet_cnn")

    @patch("app.inference.get_model")
    @patch("app.inference.torch.load", return_value={})
    def test_augmentation_forced_false(
        self, mock_load: MagicMock, mock_get_model: MagicMock, best_model_config: Path
    ):
        mock_get_model.return_value = make_mock_model()
        detector = BowelSoundDetector(config_path=best_model_config)
        assert detector.pp_config.use_augmentation is False

    @patch("app.inference.get_model")
    @patch("app.inference.torch.load", return_value={})
    def test_preserves_bandpass_setting(
        self, mock_load: MagicMock, mock_get_model: MagicMock, best_model_config: Path
    ):
        mock_get_model.return_value = make_mock_model()
        detector = BowelSoundDetector(config_path=best_model_config)
        assert detector.pp_config.use_bandpass is True


# ── detect() ──────────────────────────────────────────────────────────

class TestDetect:
    @patch("app.inference.get_model")
    @patch("app.inference.torch.load", return_value={})
    @patch("app.inference.librosa.load")
    @patch("app.inference.apply_preprocessing_pipeline")
    def test_returns_dataframe(
        self,
        mock_pp: MagicMock,
        mock_librosa: MagicMock,
        mock_torch_load: MagicMock,
        mock_get_model: MagicMock,
        best_model_config: Path,
    ):
        # Setup: 2-second audio
        audio = np.random.randn(int(22050 * 2)).astype(np.float32)
        mock_librosa.return_value = (audio, 22050)
        mock_pp.return_value = audio
        mock_get_model.return_value = make_mock_model()

        detector = BowelSoundDetector(config_path=best_model_config)
        result = detector.detect("fake.wav", window_s=1.0, hop_s=0.5)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["start_s", "end_s", "class_label", "confidence"]

    @patch("app.inference.get_model")
    @patch("app.inference.torch.load", return_value={})
    @patch("app.inference.librosa.load")
    @patch("app.inference.apply_preprocessing_pipeline")
    def test_empty_result_on_low_confidence(
        self,
        mock_pp: MagicMock,
        mock_librosa: MagicMock,
        mock_torch_load: MagicMock,
        mock_get_model: MagicMock,
        best_model_config: Path,
    ):
        audio = np.random.randn(int(22050 * 2)).astype(np.float32)
        mock_librosa.return_value = (audio, 22050)
        mock_pp.return_value = audio
        # Low-confidence logits → all below threshold
        low_conf = torch.tensor([[0.1, 0.1, 0.1]])
        mock_get_model.return_value = make_mock_model(output_logits=low_conf)

        detector = BowelSoundDetector(config_path=best_model_config)
        result = detector.detect("fake.wav", confidence_threshold=0.9)

        assert len(result) == 0
        assert list(result.columns) == ["start_s", "end_s", "class_label", "confidence"]
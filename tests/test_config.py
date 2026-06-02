"""Tests for src/config.py — constants, paths, and configuration dataclasses."""

from __future__ import annotations

import pytest

from src.config import (
    SR,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    SPEC_SIZE,
    SEED,
    NUM_CLASSES,
    TARGET_CLASSES,
    CLASS_TO_IDX,
    LABEL_MAP,
    PROJECT_ROOT,
    DATA_DIR,
    EXPERIMENTS_DIR,
    RESULTS_DIR,
    PreprocessingConfig,
    TrainingConfig,
)

# ── Constants ─────────────────────────────────────────────────────────


class TestConstants:
    def test_audio_constants(self):
        assert SR == 22050
        assert N_MELS == 128
        assert N_FFT == 1024
        assert HOP_LENGTH == 512
        assert SPEC_SIZE == 128

    def test_classification_constants(self):
        assert TARGET_CLASSES == ["b", "mb", "h"]
        assert NUM_CLASSES == len(TARGET_CLASSES)
        assert NUM_CLASSES == 3

    def test_class_to_idx_matches_target_classes(self):
        for i, cls in enumerate(TARGET_CLASSES):
            assert CLASS_TO_IDX[cls] == i

    def test_label_map_normalisation(self):
        assert LABEL_MAP["sb"] == "b"
        assert LABEL_MAP["sbs"] == "b"

    def test_seed(self):
        assert SEED == 42

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_paths_are_absolute(self):
        assert PROJECT_ROOT.is_absolute()
        assert DATA_DIR.is_absolute()
        assert EXPERIMENTS_DIR.is_absolute()
        assert RESULTS_DIR.is_absolute()


# ── PreprocessingConfig ───────────────────────────────────────────────


class TestPreprocessingConfig:
    def test_defaults(self, default_pp_config: PreprocessingConfig):
        assert default_pp_config.use_bandpass is False
        assert default_pp_config.use_normalise is False
        assert default_pp_config.use_augmentation is False
        assert default_pp_config.low_hz == 21.5
        assert default_pp_config.high_hz == 409.1

    @pytest.mark.parametrize(
        "field,value",
        [
            ("use_bandpass", True),
            ("use_normalise", True),
            ("use_augmentation", True),
            ("low_hz", 50.0),
            ("high_hz", 500.0),
        ],
    )
    def test_custom_fields(self, field: str, value):
        config = PreprocessingConfig(**{field: value})
        assert getattr(config, field) == value

    @pytest.mark.parametrize(
        "input_dict,expected_bandpass",
        [
            ({"use_bandpass": True}, True),
            ({"use_bandpass": False, "use_normalise": True}, False),
            ({}, False),
        ],
    )
    def test_from_dict(self, input_dict: dict, expected_bandpass: bool):
        config = PreprocessingConfig.from_dict(input_dict)
        assert config.use_bandpass is expected_bandpass

    def test_from_dict_ignores_unknown_keys(self):
        config = PreprocessingConfig.from_dict(
            {
                "use_bandpass": True,
                "unknown_future_key": 42,
                "another_key": "hello",
            }
        )
        assert config.use_bandpass is True
        assert not hasattr(config, "unknown_future_key")

    def test_to_dict_roundtrip(self, bandpass_pp_config: PreprocessingConfig):
        d = bandpass_pp_config.to_dict()
        restored = PreprocessingConfig.from_dict(d)
        assert restored == bandpass_pp_config

    def test_equality(self):
        a = PreprocessingConfig(use_bandpass=True)
        b = PreprocessingConfig(use_bandpass=True)
        c = PreprocessingConfig(use_bandpass=False)
        assert a == b
        assert a != c

    @pytest.mark.parametrize(
        "json_str",
        [
            '{"use_bandpass": true, "use_normalise": false, "use_augmentation": false}',
            '{"use_bandpass": true}',
            "{}",
        ],
    )
    def test_from_dict_with_json_strings(self, json_str: str):
        """Backward compat: configs from shell scripts arrive as JSON strings."""
        import json

        d = json.loads(json_str)
        config = PreprocessingConfig.from_dict(d)
        assert isinstance(config, PreprocessingConfig)


# ── TrainingConfig ────────────────────────────────────────────────────


class TestTrainingConfig:
    def test_defaults(self, default_training_config: TrainingConfig):
        assert default_training_config.model == "resnet_cnn"
        assert default_training_config.epochs == 30
        assert default_training_config.batch_size == 16
        assert default_training_config.lr == 1e-3
        assert default_training_config.weight_decay == 1e-4
        assert default_training_config.patience == 7
        assert default_training_config.preprocessing is None

    def test_with_preprocessing(self):
        pp = PreprocessingConfig(use_bandpass=True)
        tc = TrainingConfig(model="crnn", preprocessing=pp)
        assert tc.model == "crnn"
        assert tc.preprocessing.use_bandpass is True

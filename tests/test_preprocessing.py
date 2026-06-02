"""Tests for src/preprocessing.py — bandpass, normalisation, augmentation, pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import SR
from src.preprocessing import (
    bandpass_filter,
    rms_normalise,
    augment_clip,
    apply_preprocessing_pipeline,
)

# ── bandpass_filter ───────────────────────────────────────────────────


class TestBandpassFilter:
    def test_preserves_in_band_signal(self, sine_wave: np.ndarray):
        """200 Hz sine is within 21.5–409.1 Hz — should be preserved."""
        filtered = bandpass_filter(sine_wave, SR, 21.5, 409.1)
        # Allow some edge attenuation, but signal should not be zeroed
        assert np.max(np.abs(filtered)) > 0.1 * np.max(np.abs(sine_wave))

    def test_attenuates_out_of_band_signal(self):
        """5000 Hz sine is outside 21.5–409.1 Hz — should be attenuated."""
        t = np.linspace(0, 0.5, int(SR * 0.5), endpoint=False)
        high_freq = (0.3 * np.sin(2 * np.pi * 5000 * t)).astype(np.float32)
        filtered = bandpass_filter(high_freq, SR, 21.5, 409.1)
        assert np.max(np.abs(filtered)) < 0.05 * np.max(np.abs(high_freq))

    def test_output_dtype(self, sine_wave: np.ndarray):
        filtered = bandpass_filter(sine_wave, SR, 21.5, 409.1)
        assert filtered.dtype == np.float32

    def test_output_length_matches_input(self, sine_wave: np.ndarray):
        filtered = bandpass_filter(sine_wave, SR, 21.5, 409.1)
        assert len(filtered) == len(sine_wave)

    @pytest.mark.parametrize(
        "low,high",
        [
            (21.5, 409.1),
            (50.0, 200.0),
            (100.0, 1000.0),
        ],
    )
    def test_various_frequency_ranges(
        self, sine_wave: np.ndarray, low: float, high: float
    ):
        filtered = bandpass_filter(sine_wave, SR, low, high)
        assert filtered.shape == sine_wave.shape
        assert filtered.dtype == np.float32


# ── rms_normalise ─────────────────────────────────────────────────────


class TestRmsNormalise:
    def test_output_rms_matches_target(self, sine_wave: np.ndarray):
        normalised = rms_normalise(sine_wave, target_rms=0.1)
        actual_rms = np.sqrt(np.mean(normalised**2))
        assert abs(actual_rms - 0.1) < 0.01

    def test_silence_returns_unchanged(self, silence: np.ndarray):
        result = rms_normalise(silence)
        np.testing.assert_array_equal(result, silence)

    def test_clips_to_valid_range(self):
        """Very loud input should be clipped to [-1, 1]."""
        loud = np.ones(1000, dtype=np.float32) * 10.0
        normalised = rms_normalise(loud, target_rms=0.1)
        assert np.max(normalised) <= 1.0
        assert np.min(normalised) >= -1.0

    def test_output_dtype(self, sine_wave: np.ndarray):
        normalised = rms_normalise(sine_wave)
        assert normalised.dtype == np.float32

    @pytest.mark.parametrize("target_rms", [0.01, 0.05, 0.1, 0.2, 0.5])
    def test_various_target_rms(self, sine_wave: np.ndarray, target_rms: float):
        normalised = rms_normalise(sine_wave, target_rms=target_rms)
        actual_rms = np.sqrt(np.mean(normalised**2))
        assert abs(actual_rms - target_rms) < 0.02


# ── augment_clip ──────────────────────────────────────────────────────


class TestAugmentClip:
    def test_returns_three_variants_by_default(self, sine_wave: np.ndarray):
        variants = augment_clip(sine_wave, SR)
        assert len(variants) == 3

    @pytest.mark.parametrize(
        "methods,expected_count",
        [
            (("pitch",), 1),
            (("stretch",), 1),
            (("noise",), 1),
            (("pitch", "stretch"), 2),
            (("pitch", "stretch", "noise"), 3),
        ],
    )
    def test_method_selection(
        self, sine_wave: np.ndarray, methods: tuple, expected_count: int
    ):
        variants = augment_clip(sine_wave, SR, methods=methods)
        assert len(variants) == expected_count

    def test_variants_differ_from_original(self, sine_wave: np.ndarray):
        rng = np.random.default_rng(42)
        variants = augment_clip(sine_wave, SR, rng=rng)
        for v in variants:
            assert not np.array_equal(v, sine_wave)

    def test_stretch_preserves_length(self, sine_wave: np.ndarray):
        rng = np.random.default_rng(42)
        variants = augment_clip(sine_wave, SR, methods=("stretch",), rng=rng)
        assert len(variants[0]) == len(sine_wave)

    def test_noise_preserves_length(self, sine_wave: np.ndarray):
        rng = np.random.default_rng(42)
        variants = augment_clip(sine_wave, SR, methods=("noise",), rng=rng)
        assert len(variants[0]) == len(sine_wave)

    def test_deterministic_with_seed(self, sine_wave: np.ndarray):
        v1 = augment_clip(
            sine_wave, SR, methods=("noise",), rng=np.random.default_rng(99)
        )
        v2 = augment_clip(
            sine_wave, SR, methods=("noise",), rng=np.random.default_rng(99)
        )
        np.testing.assert_array_equal(v1[0], v2[0])

    def test_output_dtype(self, sine_wave: np.ndarray):
        variants = augment_clip(sine_wave, SR)
        for v in variants:
            assert v.dtype == np.float32


# ── apply_preprocessing_pipeline ──────────────────────────────────────


class TestApplyPreprocessingPipeline:
    def test_no_preprocessing(self, sine_wave: np.ndarray):
        config = {"use_bandpass": False, "use_normalise": False}
        result = apply_preprocessing_pipeline(sine_wave, SR, config)
        np.testing.assert_array_equal(result, sine_wave)

    def test_bandpass_only(self, sine_wave: np.ndarray):
        config = {"use_bandpass": True, "use_normalise": False}
        result = apply_preprocessing_pipeline(sine_wave, SR, config)
        assert not np.array_equal(result, sine_wave)
        assert result.dtype == np.float32

    def test_normalise_only(self, sine_wave: np.ndarray):
        config = {"use_bandpass": False, "use_normalise": True}
        result = apply_preprocessing_pipeline(sine_wave, SR, config)
        rms = np.sqrt(np.mean(result**2))
        assert abs(rms - 0.1) < 0.01

    @pytest.mark.parametrize(
        "bp,norm",
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_all_flag_combinations(self, sine_wave: np.ndarray, bp: bool, norm: bool):
        config = {"use_bandpass": bp, "use_normalise": norm}
        result = apply_preprocessing_pipeline(sine_wave, SR, config)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sine_wave)

"""Tests for src/audio.py — spectrogram, parsing, seed utilities."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from src.audio import (
    set_seed, parse_labels, load_eda_summary, compute_mel_spectrogram,
)
from src.config import SR, SPEC_SIZE


# ── set_seed ──────────────────────────────────────────────────────────

class TestSetSeed:
    def test_deterministic_numpy(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_torch(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_different_seeds_produce_different_results(self, seed: int):
        set_seed(seed)
        a = np.random.rand(100)
        set_seed(seed + 1)
        b = np.random.rand(100)
        assert not np.array_equal(a, b)


# ── parse_labels ──────────────────────────────────────────────────────

class TestParseLabels:
    def test_basic_parsing(self, label_file: Path):
        events = parse_labels(label_file, "test_file")
        # Should parse b, mb, h, sb→b, sbs→b = 5 target events
        assert len(events) == 5

    def test_label_normalisation(self, label_file: Path):
        events = parse_labels(label_file, "test_file")
        labels = [e["label"] for e in events]
        assert "sb" not in labels   # mapped to b
        assert "sbs" not in labels  # mapped to b
        assert labels.count("b") == 3  # original b + sb + sbs

    def test_excludes_non_target_classes(self, label_file: Path):
        events = parse_labels(label_file, "test_file")
        labels = [e["label"] for e in events]
        assert "v" not in labels
        assert "n" not in labels

    def test_skips_malformed_lines(self, label_file: Path):
        events = parse_labels(label_file, "test_file")
        # "bad line" should be skipped silently
        assert all("start" in e for e in events)

    def test_file_id_propagated(self, label_file: Path):
        events = parse_labels(label_file, "my_file")
        assert all(e["file_id"] == "my_file" for e in events)

    @pytest.mark.parametrize("label,expected", [
        ("b", "b"),
        ("mb", "mb"),
        ("h", "h"),
        ("sb", "b"),
        ("sbs", "b"),
    ])
    def test_individual_label_mapping(self, tmp_path: Path, label: str, expected: str):
        f = tmp_path / "test.txt"
        f.write_text(f"1.0\t2.0\t{label}\n")
        events = parse_labels(f, "test")
        assert len(events) == 1
        assert events[0]["label"] == expected

    def test_event_timestamps(self, label_file: Path):
        events = parse_labels(label_file, "test")
        first = events[0]
        assert first["start"] == 1.0
        assert first["end"] == 1.1


# ── load_eda_summary ─────────────────────────────────────────────────

class TestLoadEdaSummary:
    def test_loads_json(self, eda_summary_file: Path):
        summary = load_eda_summary(eda_summary_file.parent)
        assert "class_counts" in summary
        assert "dominant_freq_band_hz" in summary
        assert summary["recommended_segment_duration_s"] == 0.5

    def test_frequency_band_values(self, eda_summary_file: Path):
        summary = load_eda_summary(eda_summary_file.parent)
        low, high = summary["dominant_freq_band_hz"]
        assert low == 21.5
        assert high == 409.1


# ── compute_mel_spectrogram ──────────────────────────────────────────

class TestComputeMelSpectrogram:
    def test_output_shape(self, sine_wave: np.ndarray):
        spec = compute_mel_spectrogram(sine_wave)
        assert spec.shape == (1, SPEC_SIZE, SPEC_SIZE)

    def test_output_dtype(self, sine_wave: np.ndarray):
        spec = compute_mel_spectrogram(sine_wave)
        assert spec.dtype == torch.float32

    @pytest.mark.parametrize("duration", [0.1, 0.5, 1.0, 2.0])
    def test_different_lengths_produce_same_shape(self, duration: float):
        """The resize step guarantees fixed output shape regardless of input length."""
        n_samples = int(SR * duration)
        signal = np.random.randn(n_samples).astype(np.float32)
        spec = compute_mel_spectrogram(signal)
        assert spec.shape == (1, SPEC_SIZE, SPEC_SIZE)

    def test_deterministic(self, sine_wave: np.ndarray):
        spec1 = compute_mel_spectrogram(sine_wave)
        spec2 = compute_mel_spectrogram(sine_wave)
        assert torch.equal(spec1, spec2)

    def test_silence_produces_finite_values(self, silence: np.ndarray):
        spec = compute_mel_spectrogram(silence)
        assert torch.isfinite(spec).all()

    def test_impulse_vs_sine_differ(self, sine_wave: np.ndarray, impulse_signal: np.ndarray):
        spec_sine = compute_mel_spectrogram(sine_wave)
        spec_impulse = compute_mel_spectrogram(impulse_signal)
        assert not torch.equal(spec_sine, spec_impulse)
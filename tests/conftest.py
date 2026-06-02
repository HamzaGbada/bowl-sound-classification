"""Shared fixtures and mock factories for the test suite."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.config import (
    SR,
    NUM_CLASSES,
    TARGET_CLASSES,
    PreprocessingConfig,
    TrainingConfig,
)

# ── Audio fixtures ────────────────────────────────────────────────────


@pytest.fixture
def sine_wave() -> np.ndarray:
    """A 0.5-second 200 Hz sine wave at SR=22050 — simulates a harmonic bowel sound."""
    duration = 0.5
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


@pytest.fixture
def impulse_signal() -> np.ndarray:
    """A 0.5-second impulse (click) — simulates a burst bowel sound."""
    n_samples = int(SR * 0.5)
    y = np.zeros(n_samples, dtype=np.float32)
    mid = n_samples // 2
    y[mid - 5 : mid + 5] = 0.8  # sharp 10-sample impulse
    return y


@pytest.fixture
def silence() -> np.ndarray:
    """A 0.5-second silent signal — edge case for normalisation."""
    return np.zeros(int(SR * 0.5), dtype=np.float32)


@pytest.fixture
def short_segment() -> np.ndarray:
    """A very short signal (100 samples) — tests padding logic."""
    return np.random.default_rng(42).normal(0, 0.1, 100).astype(np.float32)


@pytest.fixture
def random_spectrogram() -> torch.Tensor:
    """A random (1, 1, 128, 128) tensor — model input shape for testing forward pass."""
    return torch.randn(1, 1, 128, 128)


@pytest.fixture
def random_batch() -> torch.Tensor:
    """A random (4, 1, 128, 128) batch — tests batched forward pass."""
    return torch.randn(4, 1, 128, 128)


# ── Config fixtures ───────────────────────────────────────────────────


@pytest.fixture
def default_pp_config() -> PreprocessingConfig:
    """Default preprocessing config — everything disabled."""
    return PreprocessingConfig()


@pytest.fixture
def bandpass_pp_config() -> PreprocessingConfig:
    """Bandpass-only preprocessing config (the best configuration)."""
    return PreprocessingConfig(use_bandpass=True)


@pytest.fixture
def full_pp_config() -> PreprocessingConfig:
    """Full preprocessing config — all steps enabled."""
    return PreprocessingConfig(
        use_bandpass=True, use_normalise=True, use_augmentation=True
    )


@pytest.fixture
def default_training_config() -> TrainingConfig:
    """Default training config."""
    return TrainingConfig()


# ── File-system fixtures ──────────────────────────────────────────────


@pytest.fixture
def label_file(tmp_path: Path) -> Path:
    """Create a temporary label file with known events."""
    content = (
        "1.0\t1.1\tb\n"
        "2.0\t2.5\tmb\n"
        "3.0\t3.8\th\n"
        "4.0\t4.1\tsb\n"  # should be mapped to b
        "5.0\t5.05\tsbs\n"  # should be mapped to b (typo)
        "6.0\t7.0\tv\n"  # should be excluded
        "8.0\t9.0\tn\n"  # should be excluded
        "bad line\n"  # should be skipped
    )
    p = tmp_path / "test_labels.txt"
    p.write_text(content)
    return p


@pytest.fixture
def eda_summary_file(tmp_path: Path) -> Path:
    """Create a temporary EDA summary JSON."""
    summary = {
        "class_counts": {"b": 100, "mb": 90, "h": 10},
        "mean_durations": {"b": 0.12, "mb": 0.45, "h": 0.77},
        "recommended_segment_duration_s": 0.5,
        "dominant_freq_band_hz": [21.5, 409.1],
    }
    p = tmp_path / "eda_summary.json"
    p.write_text(json.dumps(summary))
    return p


@pytest.fixture
def best_model_config(tmp_path: Path) -> Path:
    """Create a temporary best_model_config.json."""
    config = {
        "model": "resnet_cnn",
        "preprocessing_config": {
            "use_bandpass": True,
            "use_normalise": False,
            "use_augmentation": False,
        },
        "macro_f1": 0.9104,
        "checkpoint": "experiments/ablation_cnn_bandpass/best_model.pt",
    }
    p = tmp_path / "best_model_config.json"
    p.write_text(json.dumps(config))
    return p


# ── Mock factories ────────────────────────────────────────────────────


def make_mock_model(output_logits: torch.Tensor | None = None) -> MagicMock:
    """Create a mock model that returns fixed logits.

    Default: high confidence for class 0 (burst).
    """
    model = MagicMock()
    if output_logits is None:
        output_logits = torch.tensor([[5.0, 0.1, -1.0]])
    model.return_value = output_logits
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=iter([torch.randn(3, 3)]))
    model.state_dict = MagicMock(return_value={})
    model.load_state_dict = MagicMock()
    return model


def make_mock_dataloader(n_batches: int = 2, batch_size: int = 4) -> list:
    """Create a list of (specs, labels, metadata) tuples simulating a DataLoader."""
    batches = []
    for _ in range(n_batches):
        specs = torch.randn(batch_size, 1, 128, 128)
        labels = torch.randint(0, NUM_CLASSES, (batch_size,))
        metas = [
            {"file_id": "test", "start": 0.0, "end": 0.5, "label_name": "b"}
        ] * batch_size
        batches.append((specs, labels, metas))
    return batches

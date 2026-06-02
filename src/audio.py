"""Shared audio I/O, spectrogram computation, and seed utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
import librosa
import numpy as np
import torch
import torch.nn.functional as F

try:
    from src.config import (
        SR,
        N_MELS,
        N_FFT,
        HOP_LENGTH,
        SPEC_SIZE,
        DATA_DIR,
        LABEL_MAP,
        TARGET_CLASSES,
        SEED,
    )
except ImportError:
    from config import (
        SR,
        N_MELS,
        N_FFT,
        HOP_LENGTH,
        SPEC_SIZE,
        DATA_DIR,
        LABEL_MAP,
        TARGET_CLASSES,
        SEED,
    )


# ── Reproducibility ───────────────────────────────────────────────────


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Audio loading ─────────────────────────────────────────────────────


def load_audio_files(data_dir: Path = DATA_DIR) -> dict[str, np.ndarray]:
    """Load and resample all audio files to the working sample rate."""
    audio = {}
    for name in ["23M74M", "AS_1"]:
        y, _ = librosa.load(str(data_dir / f"{name}.wav"), sr=SR, mono=True)
        audio[name] = y
    return audio


def parse_labels(filepath: Path, file_id: str) -> list[dict]:
    """Parse a tab-separated label file, normalise labels, filter to target classes."""
    rows: list[dict] = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2].strip()
            label = LABEL_MAP.get(label, label)
            if label in TARGET_CLASSES:
                rows.append(
                    {"start": start, "end": end, "label": label, "file_id": file_id}
                )
    return rows


def load_eda_summary(data_dir: Path = DATA_DIR) -> dict:
    """Load the EDA summary JSON produced by the EDA notebook."""
    with open(data_dir / "eda_summary.json") as f:
        return json.load(f)


# ── Spectrogram computation ──────────────────────────────────────────


def compute_mel_spectrogram(segment: np.ndarray, sr: int = SR) -> torch.Tensor:
    """Convert a raw audio segment to a log-mel spectrogram tensor (1, 128, 128).

    This is the single authoritative implementation used by both training
    and inference to guarantee consistency.
    """
    S = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    spec = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)
    spec = F.interpolate(
        spec.unsqueeze(0),
        size=(SPEC_SIZE, SPEC_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return spec

"""Single source of truth for all project constants, paths, and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# ── Audio constants ────────────────────────────────────────────────────
SR: int = 22050
N_MELS: int = 128
N_FFT: int = 1024
HOP_LENGTH: int = 512
SPEC_SIZE: int = 128  # target spectrogram height and width

# ── Classification ─────────────────────────────────────────────────────
TARGET_CLASSES: list[str] = ["b", "mb", "h"]
NUM_CLASSES: int = len(TARGET_CLASSES)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(TARGET_CLASSES)}
LABEL_MAP: dict[str, str] = {"sb": "b", "sbs": "b"}

# ── Reproducibility ───────────────────────────────────────────────────
SEED: int = 42


# ── Configuration dataclasses ─────────────────────────────────────────


@dataclass
class PreprocessingConfig:
    """Typed, validated preprocessing configuration."""

    use_bandpass: bool = False
    use_normalise: bool = False
    use_augmentation: bool = False
    low_hz: float = 21.5
    high_hz: float = 409.1

    @classmethod
    def from_dict(cls, d: dict) -> PreprocessingConfig:
        """Create from a raw dict, ignoring unknown keys."""
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""

    model: str = "resnet_cnn"
    experiment: str = "baseline_cnn"
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    preprocessing: Optional[PreprocessingConfig] = None

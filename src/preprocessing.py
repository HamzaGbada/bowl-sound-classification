"""Signal preprocessing: bandpass filtering, RMS normalisation, and augmentation."""

from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import butter, sosfilt


def bandpass_filter(
    y: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4
) -> np.ndarray:
    """4th-order Butterworth bandpass filter."""
    nyquist = sr / 2
    low = max(low_hz / nyquist, 1e-5)
    high = min(high_hz / nyquist, 0.9999)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, y).astype(np.float32)


def rms_normalise(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Scale waveform to target RMS, clip to [-1, 1]."""
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-8:
        return y
    y = y * (target_rms / rms)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def augment_clip(
    y: np.ndarray,
    sr: int,
    methods: tuple[str, ...] = ("pitch", "stretch", "noise"),
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Return a list of augmented variants of *y*."""
    if rng is None:
        rng = np.random.default_rng()
    variants: list[np.ndarray] = []

    if "pitch" in methods:
        semitones = rng.choice([-2, -1, 1, 2])
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semitones))
        variants.append(y_pitch.astype(np.float32))

    if "stretch" in methods:
        rate = rng.uniform(0.8, 1.2)
        y_stretch = librosa.effects.time_stretch(y, rate=float(rate))
        if len(y_stretch) > len(y):
            y_stretch = y_stretch[: len(y)]
        elif len(y_stretch) < len(y):
            y_stretch = np.pad(y_stretch, (0, len(y) - len(y_stretch)))
        variants.append(y_stretch.astype(np.float32))

    if "noise" in methods:
        signal_power = np.mean(y**2)
        snr_linear = 10 ** (20 / 10)
        noise_power = signal_power / snr_linear
        noise = rng.normal(0, np.sqrt(max(noise_power, 1e-10)), len(y))
        y_noisy = (y + noise).astype(np.float32)
        variants.append(y_noisy)

    return variants


def apply_preprocessing_pipeline(y: np.ndarray, sr: int, config: dict) -> np.ndarray:
    """Apply preprocessing steps based on config dict with boolean keys."""
    if config.get("use_bandpass", False):
        low_hz = config.get("low_hz", 21.5)
        high_hz = config.get("high_hz", 409.1)
        y = bandpass_filter(y, sr, low_hz, high_hz)
    if config.get("use_normalise", False):
        y = rms_normalise(y)
    return y

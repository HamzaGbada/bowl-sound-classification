"""Dataset classes for bowel sound classification.

Provides BowelSoundDataset, stratified splitting, class-weight computation,
and optional preprocessing/augmentation through PreprocessingConfig.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

try:
    from src.config import (
        SR,
        DATA_DIR,
        TARGET_CLASSES,
        CLASS_TO_IDX,
        SEED,
        PreprocessingConfig,
    )
    from src.audio import (
        load_audio_files,
        parse_labels,
        load_eda_summary,
        compute_mel_spectrogram,
    )
    from src.preprocessing import apply_preprocessing_pipeline, augment_clip
except ImportError:
    from config import (
        SR,
        DATA_DIR,
        TARGET_CLASSES,
        CLASS_TO_IDX,
        SEED,
        PreprocessingConfig,
    )
    from audio import (
        load_audio_files,
        parse_labels,
        load_eda_summary,
        compute_mel_spectrogram,
    )
    from preprocessing import apply_preprocessing_pipeline, augment_clip


# ── Main dataset ──────────────────────────────────────────────────────


class BowelSoundDataset(Dataset):
    """Bowel-sound event dataset with optional preprocessing.

    Each item is a (spectrogram, label, metadata) tuple where the spectrogram
    is a (1, 128, 128) tensor ready for the model.
    """

    def __init__(
        self,
        clip_duration: float | None = None,
        preprocessing_config: PreprocessingConfig | dict | None = None,
        split: str | None = None,
    ) -> None:
        summary = load_eda_summary()
        self.clip_duration: float = (
            clip_duration or summary["recommended_segment_duration_s"]
        )

        # Accept both dataclass and raw dict for backward compatibility
        if preprocessing_config is None:
            self.pp_config = PreprocessingConfig()
        elif isinstance(preprocessing_config, dict):
            self.pp_config = PreprocessingConfig.from_dict(preprocessing_config)
        else:
            self.pp_config = preprocessing_config

        # Merge EDA frequency band into config defaults
        freq_band = summary.get("dominant_freq_band_hz", [21.5, 409.1])
        if self.pp_config.low_hz == 21.5:
            self.pp_config.low_hz = freq_band[0]
        if self.pp_config.high_hz == 409.1:
            self.pp_config.high_hz = freq_band[1]

        self.split = split
        self.audio: dict[str, np.ndarray] = load_audio_files()

        self.events: list[dict] = []
        self.events += parse_labels(DATA_DIR / "23M74M.txt", "23M74M")
        self.events += parse_labels(DATA_DIR / "AS_1.txt", "AS_1")
        self.labels: list[int] = [CLASS_TO_IDX[e["label"]] for e in self.events]

        # Pre-extract and cache raw segments
        self._segments: list[np.ndarray] = [
            self._extract_raw_segment(i) for i in range(len(self.events))
        ]

    # ── Internal helpers ──────────────────────────────────────────────

    def _extract_raw_segment(self, idx: int) -> np.ndarray:
        """Extract a fixed-length clip centered on an event's midpoint."""
        ev = self.events[idx]
        y = self.audio[ev["file_id"]]
        midpoint = (ev["start"] + ev["end"]) / 2
        half = self.clip_duration / 2
        start_sample = int((midpoint - half) * SR)
        end_sample = int((midpoint + half) * SR)

        pad_left = max(0, -start_sample)
        pad_right = max(0, end_sample - len(y))
        start_sample = max(0, start_sample)
        end_sample = min(len(y), end_sample)

        segment = y[start_sample:end_sample]
        if pad_left > 0 or pad_right > 0:
            segment = np.pad(segment, (pad_left, pad_right))

        expected = int(self.clip_duration * SR)
        if len(segment) < expected:
            segment = np.pad(segment, (0, expected - len(segment)))
        elif len(segment) > expected:
            segment = segment[:expected]
        return segment.astype(np.float32)

    def _segment_to_spectrogram(self, segment: np.ndarray) -> torch.Tensor:
        """Apply preprocessing then convert to log-mel spectrogram."""
        segment = apply_preprocessing_pipeline(segment, SR, self.pp_config.to_dict())

        expected = int(self.clip_duration * SR)
        if len(segment) < expected:
            segment = np.pad(segment, (0, expected - len(segment)))
        elif len(segment) > expected:
            segment = segment[:expected]

        return compute_mel_spectrogram(segment)

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        segment = self._segments[idx]
        label = self.labels[idx]
        ev = self.events[idx]
        metadata = {
            "file_id": ev["file_id"],
            "start": ev["start"],
            "end": ev["end"],
            "label_name": ev["label"],
        }
        spec = self._segment_to_spectrogram(segment)
        return spec, label, metadata


# ── Split dataset with augmentation ───────────────────────────────────


class _SplitDataset(Dataset):
    """Wraps a BowelSoundDataset, exposing only selected indices + augmented items."""

    def __init__(self, full_dataset: BowelSoundDataset, indices: list[int]) -> None:
        self.full_dataset = full_dataset
        self.indices = indices
        self.n_original = len(indices)
        self.labels: list[int] = [full_dataset.labels[i] for i in indices]

        self._aug_items: list[tuple[np.ndarray, int, dict]] = []
        if full_dataset.pp_config.use_augmentation and full_dataset.split == "train":
            self._build_augmented_items()

    def _build_augmented_items(self) -> None:
        """Generate augmented copies for minority classes until balanced."""
        counts = np.bincount(self.labels, minlength=len(TARGET_CLASSES))
        max_count = int(counts.max())
        threshold = int(max_count * 0.5)
        rng = np.random.default_rng(SEED)

        for cls_idx in range(len(TARGET_CLASSES)):
            if counts[cls_idx] >= threshold:
                continue
            cls_positions = [i for i, l in enumerate(self.labels) if l == cls_idx]
            needed = max_count - counts[cls_idx]
            generated = 0
            while generated < needed:
                pos = cls_positions[rng.integers(len(cls_positions))]
                orig_idx = self.indices[pos]
                segment = self.full_dataset._segments[orig_idx]
                ev = self.full_dataset.events[orig_idx]
                variants = augment_clip(segment, SR, rng=rng)
                for v in variants:
                    if generated >= needed:
                        break
                    meta = {
                        "file_id": ev["file_id"],
                        "start": ev["start"],
                        "end": ev["end"],
                        "label_name": ev["label"],
                        "augmented": True,
                    }
                    self._aug_items.append((v, cls_idx, meta))
                    generated += 1

        self.labels += [item[1] for item in self._aug_items]

    def __len__(self) -> int:
        return self.n_original + len(self._aug_items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        if idx < self.n_original:
            return self.full_dataset[self.indices[idx]]
        aug_idx = idx - self.n_original
        segment, label, metadata = self._aug_items[aug_idx]
        spec = self.full_dataset._segment_to_spectrogram(segment)
        return spec, label, metadata


# ── Public API ────────────────────────────────────────────────────────


def get_splits(
    dataset: BowelSoundDataset | None = None,
    preprocessing_config: PreprocessingConfig | dict | None = None,
    clip_duration: float | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Deterministic stratified 70/15/15 split.

    If *preprocessing_config* is provided, creates separate dataset instances
    per split so that augmentation only applies to training.
    """
    if dataset is None and preprocessing_config is None:
        raise ValueError("Provide either dataset or preprocessing_config")

    base = dataset or BowelSoundDataset(clip_duration=clip_duration)
    labels = base.labels
    indices = list(range(len(base.events)))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=SEED,
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=SEED,
    )

    if preprocessing_config is None:
        return Subset(base, train_idx), Subset(base, val_idx), Subset(base, test_idx)

    # Normalise dict → PreprocessingConfig if needed
    if isinstance(preprocessing_config, dict):
        preprocessing_config = PreprocessingConfig.from_dict(preprocessing_config)

    splits: dict[str, Dataset] = {}
    for split_name, split_idx in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        ds = BowelSoundDataset(
            clip_duration=clip_duration,
            preprocessing_config=preprocessing_config,
            split=split_name,
        )
        splits[split_name] = _SplitDataset(ds, split_idx)

    return splits["train"], splits["val"], splits["test"]


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """Inverse-frequency class weights for weighted CrossEntropyLoss."""
    if hasattr(dataset, "labels"):
        all_labels = dataset.labels
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "labels"):
        all_labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        raise ValueError("Cannot extract labels from dataset")
    counts = np.bincount(all_labels, minlength=len(TARGET_CLASSES)).astype(float)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(TARGET_CLASSES)
    return torch.tensor(weights, dtype=torch.float32)


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Stack spectrograms and labels into batch tensors."""
    specs, labels, metas = zip(*batch)
    return torch.stack(specs), torch.tensor(labels, dtype=torch.long), list(metas)


# ── Self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = BowelSoundDataset()
    print(f"Total events: {len(ds)}")
    print(f"Clip duration: {ds.clip_duration}s")

    train_ds, val_ds, test_ds = get_splits(dataset=ds)
    print(
        f"Baseline split — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    weights = compute_class_weights(ds)
    print(f"Class weights: {dict(zip(TARGET_CLASSES, weights.tolist()))}")

    spec, label, meta = ds[0]
    print(f"Spectrogram shape: {spec.shape}, Label: {label}, Meta: {meta}")

    print("\n--- Preprocessed + Augmented ---")
    pp_config = PreprocessingConfig(
        use_bandpass=True, use_normalise=True, use_augmentation=True
    )
    train_pp, val_pp, test_pp = get_splits(preprocessing_config=pp_config)
    print(
        f"Preprocessed split — Train: {len(train_pp)}, Val: {len(val_pp)}, Test: {len(test_pp)}"
    )
    print(
        f"Train class counts: {dict(zip(TARGET_CLASSES, np.bincount(train_pp.labels, minlength=3)))}"
    )

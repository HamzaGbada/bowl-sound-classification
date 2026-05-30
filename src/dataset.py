import json
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import apply_preprocessing_pipeline, augment_clip

SR = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
LABEL_MAP = {"sb": "b", "sbs": "b"}
TARGET_CLASSES = ["b", "mb", "h"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42


def load_eda_summary():
    with open(DATA_DIR / "eda_summary.json") as f:
        return json.load(f)


def parse_labels(filepath, file_id):
    rows = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2].strip()
            label = LABEL_MAP.get(label, label)
            if label in TARGET_CLASSES:
                rows.append({
                    "start": start, "end": end, "label": label, "file_id": file_id
                })
    return rows


def load_audio_files():
    audio = {}
    for name in ["23M74M", "AS_1"]:
        y, _ = librosa.load(str(DATA_DIR / f"{name}.wav"), sr=SR, mono=True)
        audio[name] = y
    return audio


DEFAULT_PREPROCESSING_CONFIG = {
    "use_bandpass": False,
    "use_normalise": False,
    "use_augmentation": False,
}


class BowelSoundDataset(Dataset):
    def __init__(self, clip_duration=None, preprocessing_config=None, split=None):
        summary = load_eda_summary()
        self.clip_duration = clip_duration or summary["recommended_segment_duration_s"]
        self.preprocessing_config = {**DEFAULT_PREPROCESSING_CONFIG,
                                     **(preprocessing_config or {})}
        # Add EDA freq band to config for bandpass filter
        freq_band = summary.get("dominant_freq_band_hz", [21.5, 409.1])
        self.preprocessing_config.setdefault("low_hz", freq_band[0])
        self.preprocessing_config.setdefault("high_hz", freq_band[1])

        self.split = split
        self.audio = load_audio_files()
        self.events = []
        self.events += parse_labels(DATA_DIR / "23M74M.txt", "23M74M")
        self.events += parse_labels(DATA_DIR / "AS_1.txt", "AS_1")
        self.labels = [CLASS_TO_IDX[e["label"]] for e in self.events]

        # Pre-extract and cache raw segments
        self._segments = [self._extract_raw_segment(i) for i in range(len(self.events))]

    def _extract_raw_segment(self, idx):
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

    def __len__(self):
        return len(self.events)

    def _segment_to_spectrogram(self, segment):
        """Convert raw segment to log-mel spectrogram tensor (1, 128, 128)."""
        # Apply preprocessing pipeline
        segment = apply_preprocessing_pipeline(segment, SR, self.preprocessing_config)

        expected = int(self.clip_duration * SR)
        if len(segment) < expected:
            segment = np.pad(segment, (0, expected - len(segment)))
        elif len(segment) > expected:
            segment = segment[:expected]

        S = librosa.feature.melspectrogram(
            y=segment, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        spec = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)
        spec = F.interpolate(spec.unsqueeze(0), size=(128, 128), mode="bilinear",
                             align_corners=False).squeeze(0)
        return spec

    def __getitem__(self, idx):
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


def get_splits(dataset=None, preprocessing_config=None, clip_duration=None):
    """Deterministic stratified 70/15/15 split.

    If preprocessing_config is provided, creates separate dataset instances per split
    so that augmentation only applies to training.
    Otherwise falls back to Subset-based splitting of a single dataset.
    """
    if dataset is None and preprocessing_config is None:
        raise ValueError("Provide either dataset or preprocessing_config")

    # Build a base dataset for splitting indices
    base = dataset or BowelSoundDataset(clip_duration=clip_duration)
    labels = base.labels
    indices = list(range(len(base.events)))  # only original events

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=labels, random_state=SEED
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=SEED
    )

    if preprocessing_config is None:
        # Legacy path: no preprocessing, Subset-based
        return Subset(base, train_idx), Subset(base, val_idx), Subset(base, test_idx)

    # Create separate datasets per split with preprocessing
    splits = {}
    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ds = BowelSoundDataset(
            clip_duration=clip_duration,
            preprocessing_config=preprocessing_config,
            split=split_name,
        )
        # Filter to only events in this split (+ augmented items for train)
        split_ds = _SplitDataset(ds, split_idx)
        splits[split_name] = split_ds

    return splits["train"], splits["val"], splits["test"]


class _SplitDataset(Dataset):
    """Wraps a BowelSoundDataset, exposing only selected indices + augmented items."""

    def __init__(self, full_dataset, indices):
        self.full_dataset = full_dataset
        self.indices = indices
        self.n_original = len(indices)
        self.labels = [full_dataset.labels[i] for i in indices]

        # Build augmented items for training split only
        self._aug_items = []
        use_aug = full_dataset.preprocessing_config.get("use_augmentation", False)
        if use_aug and full_dataset.split == "train":
            self._build_augmented_items()

    def _build_augmented_items(self):
        """Generate augmented copies for minority classes until balanced."""
        split_labels = self.labels  # labels of original items in this split
        counts = np.bincount(split_labels, minlength=len(TARGET_CLASSES))
        max_count = int(counts.max())
        threshold = int(max_count * 0.5)
        rng = np.random.default_rng(SEED)

        for cls_idx in range(len(TARGET_CLASSES)):
            if counts[cls_idx] >= threshold:
                continue
            # Indices into self.indices for this class
            cls_positions = [i for i, l in enumerate(split_labels) if l == cls_idx]
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

        # Update labels to include augmented
        self.labels += [item[1] for item in self._aug_items]

    def __len__(self):
        return self.n_original + len(self._aug_items)

    def __getitem__(self, idx):
        if idx < self.n_original:
            return self.full_dataset[self.indices[idx]]
        else:
            aug_idx = idx - self.n_original
            segment, label, metadata = self._aug_items[aug_idx]
            spec = self.full_dataset._segment_to_spectrogram(segment)
            return spec, label, metadata


def compute_class_weights(dataset):
    """Inverse frequency class weights."""
    if hasattr(dataset, "labels"):
        all_labels = dataset.labels
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "labels"):
        # Subset
        all_labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        raise ValueError("Cannot extract labels from dataset")
    counts = np.bincount(all_labels, minlength=len(TARGET_CLASSES)).astype(float)
    counts = np.maximum(counts, 1)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(TARGET_CLASSES)
    return torch.tensor(weights, dtype=torch.float32)


def collate_fn(batch):
    specs, labels, metas = zip(*batch)
    return torch.stack(specs), torch.tensor(labels, dtype=torch.long), list(metas)


if __name__ == "__main__":
    # Test baseline mode (no preprocessing)
    ds = BowelSoundDataset()
    print(f"Total events: {len(ds)}")
    print(f"Clip duration: {ds.clip_duration}s")

    train_ds, val_ds, test_ds = get_splits(dataset=ds)
    print(f"Baseline split — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    weights = compute_class_weights(ds)
    print(f"Class weights: {dict(zip(TARGET_CLASSES, weights.tolist()))}")

    spec, label, meta = ds[0]
    print(f"Spectrogram shape: {spec.shape}, Label: {label}, Meta: {meta}")

    # Test preprocessed + augmented mode
    print("\n--- Preprocessed + Augmented ---")
    pp_config = {"use_bandpass": True, "use_normalise": True, "use_augmentation": True}
    train_pp, val_pp, test_pp = get_splits(preprocessing_config=pp_config)
    print(f"Preprocessed split — Train: {len(train_pp)}, Val: {len(val_pp)}, Test: {len(test_pp)}")
    print(f"Train class counts: {dict(zip(TARGET_CLASSES, np.bincount(train_pp.labels, minlength=3)))}")
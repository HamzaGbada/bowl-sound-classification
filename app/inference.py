"""Standalone inference engine for bowel sound detection. No Streamlit dependency."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from src.config import SR, TARGET_CLASSES, PROJECT_ROOT, PreprocessingConfig
    from src.audio import compute_mel_spectrogram
    from src.models import get_model
    from src.preprocessing import apply_preprocessing_pipeline
except ImportError:
    SRC_DIR = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(SRC_DIR))
    from config import SR, TARGET_CLASSES, PROJECT_ROOT, PreprocessingConfig
    from audio import compute_mel_spectrogram
    from models import get_model
    from preprocessing import apply_preprocessing_pipeline

import librosa


class BowelSoundDetector:
    """Sliding-window bowel sound detector using a trained classification model.

    Loads the best model configuration from a JSON file and provides a
    ``detect()`` method that scans an entire audio file for bowel sound events.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = PROJECT_ROOT / "results" / "best_model_config.json"
        else:
            config_path = Path(config_path)

        with open(config_path) as f:
            self.config: dict = json.load(f)

        # Build preprocessing config — never augment at inference
        self.pp_config = PreprocessingConfig.from_dict(self.config["preprocessing_config"])
        self.pp_config.use_augmentation = False

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(self.config["model"]).to(self.device)
        ckpt_path = PROJECT_ROOT / self.config["checkpoint"]
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def detect(
        self,
        wav_path: str | Path,
        window_s: float = 1.0,
        hop_s: float = 0.1,
        confidence_threshold: float = 0.6,
    ) -> pd.DataFrame:
        """Run sliding-window detection on a full audio file.

        Returns a DataFrame with columns: start_s, end_s, class_label, confidence.
        """
        y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        y = apply_preprocessing_pipeline(y, SR, self.pp_config.to_dict())

        total_dur = len(y) / SR
        window_samples = int(window_s * SR)
        hop_samples = int(hop_s * SR)

        raw_detections: list[dict] = []
        with torch.no_grad():
            for start_sample in range(0, len(y) - window_samples + 1, hop_samples):
                segment = y[start_sample:start_sample + window_samples]
                if len(segment) < window_samples:
                    segment = np.pad(segment, (0, window_samples - len(segment)))

                # Use shared spectrogram function, add batch dim
                spec = compute_mel_spectrogram(segment).unsqueeze(0).to(self.device)
                logits = self.model(spec)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                pred_class = int(probs.argmax())
                confidence = float(probs[pred_class])

                if confidence >= confidence_threshold:
                    start_s = start_sample / SR
                    end_s = start_s + window_s
                    raw_detections.append({
                        "start_s": round(start_s, 3),
                        "end_s": round(min(end_s, total_dur), 3),
                        "class_idx": pred_class,
                        "class_label": TARGET_CLASSES[pred_class],
                        "confidence": round(confidence, 4),
                    })

        if not raw_detections:
            return pd.DataFrame(columns=["start_s", "end_s", "class_label", "confidence"])

        merged = self._merge_detections(raw_detections, max_gap=0.2)
        return pd.DataFrame(merged)[["start_s", "end_s", "class_label", "confidence"]]

    @staticmethod
    def _merge_detections(detections: list[dict], max_gap: float = 0.2) -> list[dict]:
        """Merge consecutive detections of the same class if gap < max_gap."""
        if not detections:
            return []

        merged = [detections[0].copy()]
        for det in detections[1:]:
            prev = merged[-1]
            if (det["class_label"] == prev["class_label"]
                    and det["start_s"] - prev["end_s"] <= max_gap):
                prev["end_s"] = det["end_s"]
                prev["confidence"] = max(prev["confidence"], det["confidence"])
            else:
                merged.append(det.copy())

        return merged


if __name__ == "__main__":
    detector = BowelSoundDetector()
    if len(sys.argv) > 1:
        results = detector.detect(sys.argv[1])
        print(results.to_string())
    else:
        print(f"Model: {detector.config['model']}, F1: {detector.config['macro_f1']}")
        print("Usage: python app/inference.py <wav_file>")
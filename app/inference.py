"""Standalone inference engine for bowel sound detection. No Streamlit imports."""
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from models import get_model
from preprocessing import apply_preprocessing_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SR = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
CLASS_NAMES = ["b", "mb", "h"]


class BowelSoundDetector:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = PROJECT_ROOT / "results" / "best_model_config.json"
        else:
            config_path = Path(config_path)

        with open(config_path) as f:
            self.config = json.load(f)

        self.pp_config = self.config["preprocessing_config"]
        self.pp_config["use_augmentation"] = False  # never augment at inference

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(self.config["model"]).to(self.device)
        ckpt_path = PROJECT_ROOT / self.config["checkpoint"]
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def _audio_to_spectrogram(self, segment):
        """Convert raw audio segment to (1, 1, 128, 128) tensor."""
        S = librosa.feature.melspectrogram(
            y=segment, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        spec = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        spec = F.interpolate(spec, size=(128, 128), mode="bilinear",
                             align_corners=False)
        return spec

    def detect(self, wav_path, window_s=1.0, hop_s=0.1, confidence_threshold=0.6):
        """Run sliding-window detection on a full audio file.

        Returns a DataFrame with columns: start_s, end_s, class_label, confidence.
        """
        y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        y = apply_preprocessing_pipeline(y, SR, self.pp_config)

        total_dur = len(y) / SR
        window_samples = int(window_s * SR)
        hop_samples = int(hop_s * SR)

        # Sliding window predictions
        raw_detections = []
        with torch.no_grad():
            for start_sample in range(0, len(y) - window_samples + 1, hop_samples):
                segment = y[start_sample:start_sample + window_samples]
                if len(segment) < window_samples:
                    segment = np.pad(segment, (0, window_samples - len(segment)))

                spec = self._audio_to_spectrogram(segment).to(self.device)
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
                        "class_label": CLASS_NAMES[pred_class],
                        "confidence": round(confidence, 4),
                    })

        if not raw_detections:
            return pd.DataFrame(columns=["start_s", "end_s", "class_label", "confidence"])

        # Merge consecutive windows of the same class (gap < 0.2s)
        merged = self._merge_detections(raw_detections, max_gap=0.2)
        return pd.DataFrame(merged)[["start_s", "end_s", "class_label", "confidence"]]

    @staticmethod
    def _merge_detections(detections, max_gap=0.2):
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
    import sys as _sys
    detector = BowelSoundDetector()
    if len(_sys.argv) > 1:
        results = detector.detect(_sys.argv[1])
        print(results.to_string())
    else:
        print(f"Model: {detector.config['model']}, F1: {detector.config['macro_f1']}")
        print("Usage: python app/inference.py <wav_file>")
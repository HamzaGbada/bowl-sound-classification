"""Test-set evaluation: loads best checkpoint, computes metrics, saves results."""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix,
)

try:
    from src.config import SEED, NUM_CLASSES, EXPERIMENTS_DIR, TARGET_CLASSES, PreprocessingConfig
    from src.audio import set_seed
    from src.models import get_model
    from src.dataset import BowelSoundDataset, get_splits, collate_fn
    from src.train import build_datasets
except ImportError:
    from config import SEED, NUM_CLASSES, EXPERIMENTS_DIR, TARGET_CLASSES, PreprocessingConfig
    from audio import set_seed
    from models import get_model
    from dataset import BowelSoundDataset, get_splits, collate_fn
    from train import build_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet_cnn", "crnn", "panns"])
    parser.add_argument("--exp", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--preprocessing", type=str, default=None,
                        help="JSON string with preprocessing config")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data — test split uses preprocessing but never augmentation
    if args.preprocessing:
        pp_dict = json.loads(args.preprocessing)
        pp_dict["use_augmentation"] = False
        _, _, test_ds = build_datasets(json.dumps(pp_dict))
    else:
        _, _, test_ds = build_datasets()

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(f"Test set size: {len(test_ds)}")

    # Load model
    exp_dir = EXPERIMENTS_DIR / args.exp
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location=device,
                                     weights_only=True))
    model.eval()

    # Metrics
    f1_per_class = MulticlassF1Score(num_classes=NUM_CLASSES, average=None).to(device)
    f1_macro = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device)
    prec_per_class = MulticlassPrecision(num_classes=NUM_CLASSES, average=None).to(device)
    rec_per_class = MulticlassRecall(num_classes=NUM_CLASSES, average=None).to(device)
    cm_metric = MulticlassConfusionMatrix(num_classes=NUM_CLASSES).to(device)

    with torch.no_grad():
        for specs, labels, _ in test_loader:
            specs, labels = specs.to(device), labels.to(device)
            preds = model(specs).argmax(dim=1)
            f1_per_class.update(preds, labels)
            f1_macro.update(preds, labels)
            prec_per_class.update(preds, labels)
            rec_per_class.update(preds, labels)
            cm_metric.update(preds, labels)

    per_f1 = f1_per_class.compute().cpu().tolist()
    macro_f1 = f1_macro.compute().item()
    per_prec = prec_per_class.compute().cpu().tolist()
    per_rec = rec_per_class.compute().cpu().tolist()
    cm = cm_metric.compute().cpu().tolist()

    results = {
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {c: round(v, 4) for c, v in zip(TARGET_CLASSES, per_f1)},
        "precision": {c: round(v, 4) for c, v in zip(TARGET_CLASSES, per_prec)},
        "recall": {c: round(v, 4) for c, v in zip(TARGET_CLASSES, per_rec)},
        "confusion_matrix": cm,
    }

    with open(exp_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {exp_dir / 'test_results.json'}")
    print(json.dumps(results, indent=2))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt="g", cmap="Blues",
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {args.model}")
    plt.tight_layout()
    plt.savefig(exp_dir / "confusion_matrix.png", dpi=150)
    print(f"Confusion matrix saved to {exp_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
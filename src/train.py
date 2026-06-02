"""Training loop with early stopping, logging, and class-weighted loss."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score
from tqdm import tqdm

try:
    from src.config import (
        SEED,
        NUM_CLASSES,
        EXPERIMENTS_DIR,
        TARGET_CLASSES,
        PreprocessingConfig,
    )
    from src.audio import set_seed
    from src.models import get_model
    from src.dataset import (
        BowelSoundDataset,
        get_splits,
        compute_class_weights,
        collate_fn,
    )
except ImportError:
    from config import (
        SEED,
        NUM_CLASSES,
        EXPERIMENTS_DIR,
        TARGET_CLASSES,
        PreprocessingConfig,
    )
    from audio import set_seed
    from models import get_model
    from dataset import BowelSoundDataset, get_splits, compute_class_weights, collate_fn


# ── Training & evaluation steps ───────────────────────────────────────


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for specs, labels, _ in tqdm(loader, desc="  Train", leave=False):
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, list[float]]:
    """Evaluate on a data loader. Returns (loss, macro_f1, per_class_f1)."""
    model.eval()
    total_loss = 0.0
    n = 0
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average=None).to(device)
    f1_macro = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device)

    for specs, labels, _ in tqdm(loader, desc="  Val  ", leave=False):
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        preds = logits.argmax(dim=1)
        f1_metric.update(preds, labels)
        f1_macro.update(preds, labels)

    per_class_f1 = f1_metric.compute().cpu().tolist()
    macro_f1 = f1_macro.compute().item()
    return total_loss / n, macro_f1, per_class_f1


# ── Data loading helper ───────────────────────────────────────────────


def build_datasets(
    preprocessing_json: str | None = None,
    clip_duration: float | None = None,
) -> tuple:
    """Build train/val/test datasets from CLI preprocessing argument.

    Shared by both train.py and evaluate.py to guarantee identical splits.
    """
    if preprocessing_json:
        pp_config = PreprocessingConfig.from_dict(json.loads(preprocessing_json))
        return get_splits(preprocessing_config=pp_config, clip_duration=clip_duration)
    dataset = BowelSoundDataset(clip_duration=clip_duration)
    return get_splits(dataset=dataset)


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["resnet_cnn", "crnn", "panns"]
    )
    parser.add_argument("--exp", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--preprocessing",
        type=str,
        default=None,
        help="JSON string with preprocessing config",
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_ds, val_ds, _ = build_datasets(args.preprocessing)
    print(f"Split sizes — Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = get_model(args.model).to(device)
    print(
        f"Model: {args.model}, Params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Loss with class weights
    class_weights = compute_class_weights(train_ds).to(device)
    print(f"Class weights: {dict(zip(TARGET_CLASSES, class_weights.tolist()))}")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Experiment dir
    exp_dir = EXPERIMENTS_DIR / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Training loop with context-managed log file
    log_path = exp_dir / "train_log.csv"
    best_f1 = -1.0
    patience_counter = 0

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        header = ["epoch", "train_loss", "val_loss", "val_macro_f1"] + [
            f"val_f1_{c}" for c in TARGET_CLASSES
        ]
        writer.writerow(header)

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_macro_f1, per_class_f1 = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            row = [
                epoch,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{val_macro_f1:.4f}",
            ] + [f"{f:.4f}" for f in per_class_f1]
            writer.writerow(row)
            log_file.flush()

            f1_str = " | ".join(
                f"{c}={f:.4f}" for c, f in zip(TARGET_CLASSES, per_class_f1)
            )
            print(
                f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
                f"Val macro-F1: {val_macro_f1:.4f}"
            )
            print(f"  Per-class F1: {f1_str}")

            if val_macro_f1 > best_f1:
                best_f1 = val_macro_f1
                patience_counter = 0
                torch.save(model.state_dict(), exp_dir / "best_model.pt")
                print(f"  -> New best model saved (F1={best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(
                        f"  Early stopping at epoch {epoch} (patience={args.patience})"
                    )
                    break

    print(f"\nTraining complete. Best val macro-F1: {best_f1:.4f}")
    print(f"Checkpoint: {exp_dir / 'best_model.pt'}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()

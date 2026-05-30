"""Aggregate all test_results.json files into a single CSV."""
import json
import csv
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Map experiment folder names to metadata
EXPERIMENT_META = {
    "baseline_cnn":              ("resnet_cnn", "none"),
    "baseline_crnn":             ("crnn",       "none"),
    "baseline_panns":            ("panns",      "none"),
    "pp_cnn":                    ("resnet_cnn", "all"),
    "pp_crnn":                   ("crnn",       "all"),
    "pp_panns":                  ("panns",      "all"),
    "ablation_cnn_bandpass":     ("resnet_cnn", "bandpass"),
    "ablation_cnn_normalise":    ("resnet_cnn", "normalise"),
    "ablation_cnn_augmentation": ("resnet_cnn", "augmentation"),
    "ablation_cnn_all":          ("resnet_cnn", "all_ablation"),
}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []

    for exp_name, (model, preprocessing) in EXPERIMENT_META.items():
        results_path = EXPERIMENTS_DIR / exp_name / "test_results.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            r = json.load(f)

        pf = r["per_class_f1"]
        prec = r["precision"]
        rec = r["recall"]
        prec_macro = sum(prec.values()) / len(prec)
        rec_macro = sum(rec.values()) / len(rec)

        rows.append({
            "experiment": exp_name,
            "model": model,
            "preprocessing": preprocessing,
            "macro_f1": r["macro_f1"],
            "f1_b": pf["b"],
            "f1_mb": pf["mb"],
            "f1_h": pf["h"],
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
        })

    # Sort by macro_f1 descending
    rows.sort(key=lambda x: x["macro_f1"], reverse=True)

    # Write CSV
    csv_path = RESULTS_DIR / "all_results.csv"
    fieldnames = ["experiment", "model", "preprocessing", "macro_f1",
                  "f1_b", "f1_mb", "f1_h", "precision_macro", "recall_macro"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print table
    print(f"\n{'Experiment':<28} {'Model':<14} {'Preproc':<14} {'Macro-F1':>8} "
          f"{'F1(b)':>7} {'F1(mb)':>7} {'F1(h)':>7} {'Prec':>7} {'Rec':>7}")
    print("-" * 105)
    for r in rows:
        print(f"{r['experiment']:<28} {r['model']:<14} {r['preprocessing']:<14} "
              f"{r['macro_f1']:>8.4f} {r['f1_b']:>7.4f} {r['f1_mb']:>7.4f} "
              f"{r['f1_h']:>7.4f} {r['precision_macro']:>7.4f} {r['recall_macro']:>7.4f}")

    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
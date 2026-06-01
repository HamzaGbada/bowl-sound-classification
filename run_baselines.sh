#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "============================================"
echo " Bowel Sound Classification — Baseline Runs"
echo "============================================"

# --- M1: ResNet CNN ---
echo ""
echo ">>> Training ResNet CNN (baseline_cnn)..."
uv run python src/train.py --model resnet_cnn --exp baseline_cnn
echo ">>> Evaluating ResNet CNN..."
uv run python src/evaluate.py --model resnet_cnn --exp baseline_cnn

# --- M2: CRNN ---
echo ""
echo ">>> Training CRNN (baseline_crnn)..."
uv run python src/train.py --model crnn --exp baseline_crnn
echo ">>> Evaluating CRNN..."
uv run python src/evaluate.py --model crnn --exp baseline_crnn

# --- M3: PANNs (EfficientNet-B0) ---
echo ""
echo ">>> Training PANNs (baseline_panns)..."
uv run python src/train.py --model panns --exp baseline_panns
echo ">>> Evaluating PANNs..."
uv run python src/evaluate.py --model panns --exp baseline_panns

# --- Summary Table ---
echo ""
echo "============================================"
echo " Summary"
echo "============================================"
echo ""
uv run python -c "
import json
from pathlib import Path

models = [
    ('resnet_cnn', 'baseline_cnn'),
    ('crnn', 'baseline_crnn'),
    ('panns', 'baseline_panns'),
]
exp_dir = Path('experiments')

print(f\"{'Model':<16} | {'Macro-F1':>8} | {'F1(b)':>7} | {'F1(mb)':>7} | {'F1(h)':>7}\")
print('-' * 60)
for model_name, exp_name in models:
    rpath = exp_dir / exp_name / 'test_results.json'
    if rpath.exists():
        r = json.loads(rpath.read_text())
        pf = r['per_class_f1']
        print(f\"{model_name:<16} | {r['macro_f1']:>8.4f} | {pf['b']:>7.4f} | {pf['mb']:>7.4f} | {pf['h']:>7.4f}\")
    else:
        print(f\"{model_name:<16} | {'N/A':>8} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7}\")
"


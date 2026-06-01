#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

PP_ALL='{"use_bandpass":true,"use_normalise":true,"use_augmentation":true}'
PP_BP='{"use_bandpass":true,"use_normalise":false,"use_augmentation":false}'
PP_NORM='{"use_bandpass":false,"use_normalise":true,"use_augmentation":false}'
PP_AUG='{"use_bandpass":false,"use_normalise":false,"use_augmentation":true}'

echo "============================================"
echo " Ablation Study — Preprocessed Models"
echo "============================================"

# --- Preprocessed runs for all 3 models ---
for model_exp in "resnet_cnn pp_cnn" "crnn pp_crnn" "panns pp_panns"; do
    set -- $model_exp
    model=$1; exp=$2
    echo ""
    echo ">>> Training $model ($exp) with full preprocessing..."
    uv run python src/train.py --model "$model" --exp "$exp" --preprocessing "$PP_ALL"
    echo ">>> Evaluating $model ($exp)..."
    uv run python src/evaluate.py --model "$model" --exp "$exp" --preprocessing "$PP_ALL"
done

# --- Ablation variants for M1 (resnet_cnn) ---
echo ""
echo "============================================"
echo " Ablation Variants — ResNet CNN"
echo "============================================"

for variant_exp in "bandpass ablation_cnn_bandpass $PP_BP" \
                   "normalise ablation_cnn_normalise $PP_NORM" \
                   "augmentation ablation_cnn_augmentation $PP_AUG" \
                   "all ablation_cnn_all $PP_ALL"; do
    set -- $variant_exp
    name=$1; exp=$2; pp=$3
    echo ""
    echo ">>> Training resnet_cnn ($exp) — $name only..."
    uv run python src/train.py --model resnet_cnn --exp "$exp" --preprocessing "$pp"
    echo ">>> Evaluating resnet_cnn ($exp)..."
    uv run python src/evaluate.py --model resnet_cnn --exp "$exp" --preprocessing "$pp"
done

# --- Aggregate results ---
echo ""
echo "============================================"
echo " Aggregating Results"
echo "============================================"
uv run python src/aggregate_results.py

echo ""
echo "Done. Results saved to results/all_results.csv"
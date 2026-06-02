# Bowel Sound Classification

Automatic detection and classification of bowel sounds from abdominal audio recordings.

**Classes:** `b` (single burst), `mb` (multiple burst), `h` (harmonic)

## Quick Start

```bash
# Launch the Streamlit app
bash run_app.sh
```

Then open http://localhost:8501 and upload a `.wav` file.

## Full Pipeline

### Phase 1 — EDA
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
```

### Phase 2 — Baseline Models
```bash
bash run_baselines.sh
```

### Phase 3 — Preprocessing
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb --output 02_preprocessing.ipynb
```

### Phase 4 — Ablation Study
```bash
bash run_ablation.sh
uv run jupyter nbconvert --to notebook --execute notebooks/03_ablation_report.ipynb --output 03_ablation_report.ipynb
```

### Phase 5 — Streamlit App
```bash
bash run_app.sh
```

## Results

| Experiment | Model | Preprocessing | Macro-F1 | F1(b) | F1(mb) | F1(h) |
|-----------|-------|--------------|----------|-------|--------|-------|
| ablation_cnn_bandpass | ResNet18 | bandpass | **0.9104** | 0.9440 | 0.9302 | 0.8571 |
| baseline_panns | EfficientNet-B0 | none | 0.8828 | 0.9245 | 0.9115 | 0.8125 |
| pp_crnn | CRNN | all | 0.8662 | 0.9136 | 0.8954 | 0.7895 |
| baseline_cnn | ResNet18 | none | 0.8575 | 0.9254 | 0.8970 | 0.7500 |

**Best model:** ResNet18 + bandpass filter (21.5–409.1 Hz), macro-F1 = 0.9104

## Project Structure

```
data/                   Raw audio + labels + EDA summary
src/
  config.py             Constants, paths, PreprocessingConfig dataclass (single source of truth)
  audio.py              Shared audio I/O, spectrogram computation, seed utilities
  preprocessing.py      Bandpass filter, RMS normalise, augmentation
  dataset.py            PyTorch Dataset classes, stratified splitting, class weights
  models.py             ResNet18, CRNN, EfficientNet-B0 (Factory pattern)
  train.py              Training loop with early stopping and logging
  evaluate.py           Test-set evaluation and metrics
  aggregate_results.py  Results aggregator
app/
  inference.py          Standalone sliding-window inference engine
  streamlit_app.py      Streamlit web app (decomposed UI sections)
notebooks/              EDA, preprocessing validation, ablation report
experiments/            Model checkpoints + training logs
results/                Aggregated results + best model config
docs/                   Phase documentation + architecture guide
```

### Architecture Highlights

- **Single source of truth:** All constants (`SR`, `SEED`, `NUM_CLASSES`, paths) defined once in `src/config.py`.
- **Typed configuration:** `PreprocessingConfig` dataclass replaces raw dicts — provides validation, autocomplete, and serialisation.
- **Shared spectrogram pipeline:** One `compute_mel_spectrogram()` function in `src/audio.py` used by both training and inference, guaranteeing consistency.
- **Factory pattern:** `MODEL_REGISTRY` in `models.py` maps string names to model classes — adding a model requires one line.
- **No over-engineering:** No ABC, no DI framework, no deep nesting. The flat structure is appropriate for a research project of this size.

See `docs/06_architecture.md` for full architectural documentation.

## Docker

All services use the `nvcr.io/nvidia/pytorch:24.08-py3` NVIDIA GPU base image.

```bash
# Launch Streamlit app
docker compose up app
# Open http://localhost:8501

# Launch Jupyter notebooks
docker compose up notebooks
# Open http://localhost:8888

# Run baseline training
docker compose run --rm training

# Run ablation study
docker compose run --rm ablation

# Run inference on a wav file
docker compose run --rm inference data/23M74M.wav
```

### Available services

| Service | Description | Port |
|---------|-------------|------|
| `app` | Streamlit web app | 8501 |
| `notebooks` | Jupyter notebook server | 8888 |
| `training` | Train baseline models (Phase 2) | — |
| `ablation` | Run ablation study (Phase 4) | — |
| `inference` | CLI inference on a wav file | — |

All services mount `data/`, `experiments/`, and `results/` as volumes for persistence.

## Testing

```bash
# Install dev dependencies
uv sync --group dev

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov=app --cov-report=term-missing
```

**149 tests** covering config, audio pipeline, preprocessing, models, dataset, and inference:

| Test file | What it tests | Tests |
|-----------|--------------|:-----:|
| `test_config.py` | Constants, PreprocessingConfig dataclass, TrainingConfig, JSON roundtrips | 25 |
| `test_audio.py` | set_seed, parse_labels, label normalisation, spectrogram shape/dtype/determinism | 29 |
| `test_preprocessing.py` | Bandpass filter, RMS normalisation, augmentation, pipeline combinations | 34 |
| `test_models.py` | Factory pattern, forward pass for all 3 models, output shapes, parameter counts | 28 |
| `test_dataset.py` | Class weights, collate_fn, config integration | 18 |
| `test_inference.py` | Detection merging (gap thresholds, confidence), detector init, detect() | 15 |

CI runs automatically on push/PR via GitHub Actions (`.github/workflows/ci.yml`).

## Dependencies

```bash
uv sync
```
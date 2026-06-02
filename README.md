# Bowel Sound Classification

[![CI](https://github.com/bobmarley/bowl-sound-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/bobmarley/bowl-sound-classification/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Macro-F1: 0.9104](https://img.shields.io/badge/Macro--F1-0.9104-brightgreen.svg)](#results)

Deep learning system for automatic detection and classification of bowel sounds from abdominal audio recordings.

**Classes:** `b` (single burst) | `mb` (multiple burst) | `h` (harmonic)

**Best model:** ResNet18 + bandpass filter (21.5–409.1 Hz) — **0.9104 macro-F1**

---

## Quick Start

```bash
# Install dependencies
uv sync

# Launch the Streamlit web app
bash run_app.sh
```

Then open http://localhost:8501 and upload a `.wav` file.

### CLI Inference (no UI)

```bash
uv run python app/inference.py data/23M74M.wav
```

---

## Full Pipeline

The project follows a 5-phase research pipeline. Each phase builds on the previous one.

### Phase 1 — Exploratory Data Analysis

Analyse the dataset, identify class imbalance, determine frequency bands, and export findings.

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
```

**Key findings:** 2,226 events across 42 min of audio. 10:1 class imbalance (h is rare). 80% of bowel sound energy in 21.5–409.1 Hz.

### Phase 2 — Baseline Models

Train 3 architectures without preprocessing to establish performance baselines.

```bash
bash run_baselines.sh
```

**Models:** ResNet18 (11.2M params), CRNN (2.7M params), EfficientNet-B0 (4.0M params)

### Phase 3 — Preprocessing Design

Validate bandpass filter, RMS normalisation, and augmentation strategies.

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb --output 02_preprocessing.ipynb
```

### Phase 4 — Ablation Study

Run 10 experiments to isolate the contribution of each preprocessing step.

```bash
bash run_ablation.sh
uv run jupyter nbconvert --to notebook --execute notebooks/03_ablation_report.ipynb --output 03_ablation_report.ipynb
```

**Key finding:** Bandpass filter alone provides +5.3 pp improvement. Normalisation and augmentation hurt pretrained models.

### Phase 5 — Deployment

Launch the interactive web application for bowel sound detection.

```bash
bash run_app.sh
```

---

## Results

### Top 4 Experiments (out of 10)

| Rank | Experiment | Model | Preprocessing | Macro-F1 | F1(b) | F1(mb) | F1(h) |
|:----:|-----------|-------|:-------------:|:--------:|:-----:|:------:|:-----:|
| 1 | ablation_cnn_bandpass | ResNet18 | bandpass | **0.9104** | 0.9440 | 0.9302 | 0.8571 |
| 2 | baseline_panns | EfficientNet-B0 | none | 0.8828 | 0.9245 | 0.9115 | 0.8125 |
| 3 | pp_crnn | CRNN | all | 0.8662 | 0.9136 | 0.8954 | 0.7895 |
| 4 | baseline_cnn | ResNet18 | none | 0.8575 | 0.9254 | 0.8970 | 0.7500 |

### Preprocessing Ablation (ResNet18)

| Variant | Macro-F1 | Delta |
|---------|:--------:|:-----:|
| Bandpass only | **0.9104** | **+5.3 pp** |
| None (baseline) | 0.8575 | — |
| Augmentation only | 0.8458 | -1.2 pp |
| Normalise only | 0.8276 | -3.0 pp |
| All combined | 0.8265 | -3.1 pp |

**Takeaway:** Domain-informed signal processing (bandpass filter derived from EDA frequency analysis) outperforms generic data augmentation. More preprocessing is not always better.

---

## Project Structure

```
src/
  config.py             Single source of truth: constants, paths, typed configs
  audio.py              Shared audio I/O, spectrogram pipeline, seed utilities
  preprocessing.py      Bandpass filter, RMS normalise, augmentation
  dataset.py            PyTorch Dataset, stratified splits, class weights
  models.py             ResNet18, CRNN, EfficientNet-B0 (Factory pattern)
  train.py              Training loop with early stopping
  evaluate.py           Test-set evaluation and metrics
  aggregate_results.py  Results aggregator
app/
  inference.py          Standalone sliding-window inference engine
  streamlit_app.py      Streamlit web app
tests/
  conftest.py           Shared fixtures and mock factories
  test_*.py             150 unit tests (pytest + parametrize + mocking)
notebooks/
  01_eda.ipynb          Exploratory data analysis
  02_preprocessing.ipynb  Preprocessing validation
  03_ablation_report.ipynb  Ablation study report
experiments/            Model checkpoints + training logs (10 experiments)
results/                Aggregated CSV + best model config JSON
docs/                   7 documentation files (see below)
```

### Architecture Highlights

- **Single source of truth:** All constants (`SR`, `SEED`, `NUM_CLASSES`, paths) defined once in `src/config.py`. No duplication across files.
- **Typed configuration:** `PreprocessingConfig` dataclass replaces raw dicts — catches typos at edit time, not runtime.
- **Shared spectrogram pipeline:** One `compute_mel_spectrogram()` in `src/audio.py` used by both training and inference. Guarantees consistency.
- **Factory pattern:** `MODEL_REGISTRY` maps string names to model classes. Adding a model = adding one dict entry.
- **Clean dependency graph:** DAG with no circular imports. `config.py` at the root, `streamlit_app.py` at the leaf.
- **No over-engineering:** No ABC, no DI framework, no deep nesting. Flat structure for a research project of this size.

---

## Testing

```bash
# Install dev dependencies
uv sync --group dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov=app --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_preprocessing.py

# Format check
uv run black --check src/ app/ tests/
```

**150 tests** across 6 test files:

| Test file | What it tests | Tests |
|-----------|--------------|:-----:|
| `test_config.py` | Constants, `PreprocessingConfig` dataclass, JSON roundtrips | 26 |
| `test_audio.py` | `set_seed`, `parse_labels`, label normalisation, spectrogram shape/dtype | 29 |
| `test_preprocessing.py` | Bandpass filter, RMS normalisation, augmentation, pipeline combos | 34 |
| `test_models.py` | Factory pattern, forward pass (all 3 models), output shapes, param counts | 28 |
| `test_dataset.py` | Class weights, `collate_fn`, config integration | 18 |
| `test_inference.py` | Detection merging, detector init, `detect()` with mocked model | 15 |

### CI Pipeline

GitHub Actions runs automatically on push/PR to `main`/`master`:

1. **Lint** — Black formatting check
2. **Test** — Full pytest suite with coverage + import smoke tests

See `.github/workflows/ci.yml`.

---

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

### Available Services

| Service | Description | Port |
|---------|-------------|:----:|
| `app` | Streamlit web app | 8501 |
| `notebooks` | Jupyter notebook server | 8888 |
| `training` | Train baseline models (Phase 2) | — |
| `ablation` | Run ablation study (Phase 4) | — |
| `inference` | CLI inference on a wav file | — |

All services mount `data/`, `experiments/`, and `results/` as volumes for persistence.

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`01_eda.md`](docs/01_eda.md) | EDA notebook: cell-by-cell walkthrough with audio fundamentals explained |
| [`02_baseline.md`](docs/02_baseline.md) | Baseline training: model architectures, training loop, results analysis |
| [`03_preprocessing.md`](docs/03_preprocessing.md) | Preprocessing notebook: bandpass, normalisation, augmentation validation |
| [`04_ablation.md`](docs/04_ablation.md) | Ablation study: 10 experiments, per-class analysis, error analysis |
| [`05_streamlit_app.md`](docs/05_streamlit_app.md) | Deployment: inference engine, Streamlit UI, sliding window detection |
| [`06_software.md`](docs/06_software.md) | Software architecture: design patterns, dependency graph, decisions |
| [`07_tests.md`](docs/07_tests.md) | Test suite: fixtures, mocking, parametrize, CI pipeline |

---

## Dependencies

```bash
# Production
uv sync

# Development (includes pytest, pytest-cov, black)
uv sync --group dev
```

### Key Libraries

| Library | Purpose |
|---------|---------|
| `torch` / `torchvision` | Model training and inference |
| `librosa` | Audio loading, resampling, mel spectrograms |
| `scipy` | Butterworth bandpass filter |
| `scikit-learn` | Stratified train/val/test splitting |
| `torchmetrics` | F1, precision, recall, confusion matrix |
| `streamlit` | Web application framework |
| `plotly` | Interactive waveform visualisation |
| `pytest` | Unit testing framework |
| `black` | Code formatting |
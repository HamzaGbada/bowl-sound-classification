"""Tests for src/models.py — model architectures, factory, and forward pass."""

from __future__ import annotations

import pytest
import torch

from src.config import NUM_CLASSES
from src.models import get_model, MODEL_REGISTRY, ResNetMelCNN, CRNN, PANNsFinetune

# ── Factory ───────────────────────────────────────────────────────────


class TestModelFactory:
    @pytest.mark.parametrize("name", ["resnet_cnn", "crnn", "panns"])
    def test_get_model_returns_module(self, name: str):
        model = get_model(name)
        assert isinstance(model, torch.nn.Module)

    def test_get_model_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent_model")

    def test_registry_contains_all_models(self):
        assert set(MODEL_REGISTRY.keys()) == {"resnet_cnn", "crnn", "panns"}

    @pytest.mark.parametrize(
        "name,expected_class",
        [
            ("resnet_cnn", ResNetMelCNN),
            ("crnn", CRNN),
            ("panns", PANNsFinetune),
        ],
    )
    def test_registry_maps_to_correct_class(self, name: str, expected_class: type):
        assert MODEL_REGISTRY[name] is expected_class


# ── Forward pass ──────────────────────────────────────────────────────


class TestForwardPass:
    @pytest.mark.parametrize("name", ["resnet_cnn", "crnn", "panns"])
    def test_single_sample_output_shape(
        self, name: str, random_spectrogram: torch.Tensor
    ):
        model = get_model(name)
        model.eval()
        with torch.no_grad():
            output = model(random_spectrogram)
        assert output.shape == (1, NUM_CLASSES)

    @pytest.mark.parametrize("name", ["resnet_cnn", "crnn", "panns"])
    def test_batch_output_shape(self, name: str, random_batch: torch.Tensor):
        model = get_model(name)
        model.eval()
        with torch.no_grad():
            output = model(random_batch)
        assert output.shape == (4, NUM_CLASSES)

    @pytest.mark.parametrize("name", ["resnet_cnn", "crnn", "panns"])
    def test_output_dtype(self, name: str, random_spectrogram: torch.Tensor):
        model = get_model(name)
        model.eval()
        with torch.no_grad():
            output = model(random_spectrogram)
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("name", ["resnet_cnn", "crnn", "panns"])
    def test_softmax_sums_to_one(self, name: str, random_spectrogram: torch.Tensor):
        model = get_model(name)
        model.eval()
        with torch.no_grad():
            logits = model(random_spectrogram)
            probs = torch.softmax(logits, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5


# ── Architecture specifics ───────────────────────────────────────────


class TestArchitectureDetails:
    def test_resnet_first_conv_is_1_channel(self):
        model = get_model("resnet_cnn")
        first_conv = model.backbone.conv1
        assert first_conv.in_channels == 1

    def test_resnet_final_fc_outputs_num_classes(self):
        model = get_model("resnet_cnn")
        assert model.backbone.fc.out_features == NUM_CLASSES

    def test_crnn_first_conv_is_1_channel(self):
        model = get_model("crnn")
        first_conv = model.cnn[0]
        assert first_conv.in_channels == 1

    def test_crnn_final_fc_outputs_num_classes(self):
        model = get_model("crnn")
        assert model.fc.out_features == NUM_CLASSES

    def test_panns_first_conv_is_1_channel(self):
        model = get_model("panns")
        first_conv = model.backbone.features[0][0]
        assert first_conv.in_channels == 1

    def test_panns_final_fc_outputs_num_classes(self):
        model = get_model("panns")
        classifier = model.backbone.classifier
        linear = classifier[-1]
        assert linear.out_features == NUM_CLASSES

    @pytest.mark.parametrize(
        "name,min_params,max_params",
        [
            ("crnn", 2_000_000, 3_500_000),
            ("panns", 3_500_000, 5_000_000),
            ("resnet_cnn", 10_000_000, 12_000_000),
        ],
    )
    def test_parameter_count_range(self, name: str, min_params: int, max_params: int):
        model = get_model(name)
        n_params = sum(p.numel() for p in model.parameters())
        assert min_params < n_params < max_params, f"{name}: {n_params:,} params"

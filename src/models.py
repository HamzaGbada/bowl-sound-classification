"""Model architectures: ResNet18, CRNN, EfficientNet-B0 (PANNs proxy)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights

try:
    from src.config import NUM_CLASSES
except ImportError:
    from config import NUM_CLASSES


# ── Utility ───────────────────────────────────────────────────────────

def _adapt_first_conv(model: nn.Module, attr_path: str) -> None:
    """Replace the first conv layer to accept 1-channel input by averaging RGB weights."""
    parts = attr_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    old_conv: nn.Conv2d = getattr(parent, parts[-1])

    new_conv = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    setattr(parent, parts[-1], new_conv)


# ── Model definitions ────────────────────────────────────────────────

class ResNetMelCNN(nn.Module):
    """M1: ResNet18 pretrained, adapted for 1-channel mel spectrogram input."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        _adapt_first_conv(self.backbone, "conv1")
        self.backbone.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CRNN(nn.Module):
    """M2: CNN front-end + BiLSTM classifier. No pretrained weights."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128 * 16, hidden_size=128,
            num_layers=2, bidirectional=True, batch_first=True,
        )
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        B, C, F, T = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(B, T, C * F)
        out, _ = self.lstm(feat)
        out = out[:, -1, :]
        return self.fc(out)


class PANNsFinetune(nn.Module):
    """M3: EfficientNet-B0 pretrained (proxy for PANNs), adapted for 1-channel input."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        _adapt_first_conv(self.backbone, "features.0.0")
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Factory ───────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "resnet_cnn": ResNetMelCNN,
    "crnn": CRNN,
    "panns": PANNsFinetune,
}


def get_model(name: str) -> nn.Module:
    """Factory function: instantiate a model by its registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
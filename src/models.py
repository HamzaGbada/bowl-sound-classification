import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights

NUM_CLASSES = 3


def _adapt_first_conv(model, attr_path):
    """Modify the first conv layer to accept 1-channel input by averaging RGB weights."""
    parts = attr_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    old_conv = getattr(parent, parts[-1])

    new_conv = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    # Average pretrained weights across input channels
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    setattr(parent, parts[-1], new_conv)


class ResNetMelCNN(nn.Module):
    """M1: ResNet18 pretrained, adapted for 1-channel mel spectrogram input."""

    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        _adapt_first_conv(self.backbone, "conv1")
        self.backbone.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        return self.backbone(x)


class CRNN(nn.Module):
    """M2: CNN front-end + BiLSTM classifier. No pretrained weights."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After 3x MaxPool2d(2) on 128x128: -> 16x16
        # Flatten freq dim (128 * 16 = 2048), keep time dim (16)
        self.lstm = nn.LSTM(
            input_size=128 * 16, hidden_size=128,
            num_layers=2, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(256, NUM_CLASSES)  # 128 * 2 (bidirectional)

    def forward(self, x):
        # x: (B, 1, 128, 128)
        feat = self.cnn(x)  # (B, 128, 16, 16)
        B, C, F, T = feat.shape
        # Flatten channels and frequency, treat time as sequence
        feat = feat.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T, C*F)
        out, _ = self.lstm(feat)  # (B, T, 256)
        out = out[:, -1, :]  # last time step
        return self.fc(out)


class PANNsFinetune(nn.Module):
    """M3: EfficientNet-B0 pretrained (proxy for PANNs), adapted for 1-channel input."""

    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        _adapt_first_conv(self.backbone, "features.0.0")
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, NUM_CLASSES),
        )

    def forward(self, x):
        return self.backbone(x)


MODEL_REGISTRY = {
    "resnet_cnn": ResNetMelCNN,
    "crnn": CRNN,
    "panns": PANNsFinetune,
}


def get_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
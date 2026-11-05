# This file contains defined classes of CNNs

import torch.nn as nn
from torchvision import models


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


class BaselineCNN(nn.Module):
    """Compact three-block CNN for pre-computed spectrogram images."""
    def __init__(
        self,
        num_classes: int = 10,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.network = _CompactSpectrogramCNN(
            num_classes=num_classes,
            n_blocks=3,
            base_channels=16,
            hidden_size=hidden_size,
            dropout=dropout,
        )

    def embed(self, x):
        return self.network.embed(x)

    def forward(self, x):
        return self.network(x)


class CustomCNN(nn.Module):
    """Configurable compact CNN; global pooling keeps its dense head small."""
    def __init__(
        self,
        num_classes: int = 10,
        n_blocks: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.2,
        base_channels: int = 16,
        block_channels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.network = _CompactSpectrogramCNN(
            num_classes=num_classes,
            n_blocks=n_blocks,
            base_channels=base_channels,
            hidden_size=hidden_size,
            dropout=dropout,
            block_channels=block_channels,
        )

    def embed(self, x):
        return self.network.embed(x)

    def forward(self, x):
        return self.network(x)


class _CompactSpectrogramCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_blocks: int,
        base_channels: int,
        hidden_size: int,
        dropout: float,
        block_channels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        if block_channels is not None:
            if not block_channels or any(channels < 1 for channels in block_channels):
                raise ValueError("block_channels must contain positive channel counts")
            n_blocks = len(block_channels)
        if n_blocks < 1:
            raise ValueError("n_blocks must be at least 1")
        if base_channels < 1:
            raise ValueError("base_channels must be at least 1")

        layers = []
        in_channels = 1
        for i in range(n_blocks):
            out_channels = (
                block_channels[i]
                if block_channels is not None
                else base_channels * 2**i
            )
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.hidden = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Linear(hidden_size, num_classes)

    def embed(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        features = self.features(x)
        pooled = self.avg_pool(features).flatten(start_dim=1)
        return self.hidden(pooled)

    def forward(self, x):
        return self.output(self.embed(x))


class PretrainedResNet(nn.Module):
    """
    ResNet-18 pretrained on ImageNet with a new output layer for the 10 genres.
    freeze_backbone=True  -> only the new output layer is trained
    freeze_backbone=False -> the whole network is fine-tuned
    Input shape expected: (Batch_Size, 3, H, W), normalised with the ImageNet mean/std.
    """
    def __init__(self, num_classes: int = 10, freeze_backbone: bool = False, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        n_features = self.backbone.fc.in_features      # 512
        self.backbone.fc = nn.Identity()
        self.output = nn.Linear(n_features, num_classes)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()       # keep the ImageNet BatchNorm statistics when frozen
        return self

    def embed(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.output(self.embed(x))


def build_model(config: dict, num_classes: int, input_shape):
    """config['model'] is 'baseline', 'custom' or 'resnet18'."""
    if config["model"] == "baseline":
        return BaselineCNN(
            num_classes,
            hidden_size=config.get("hidden_size", 64),
            dropout=config.get("dropout", 0.2),
        )
    if config["model"] == "custom":
        return CustomCNN(
            num_classes,
            n_blocks=config.get("n_blocks", len(config.get("block_channels", (16, 32, 64)))),
            hidden_size=config.get("hidden_size", 64),
            dropout=config.get("dropout", 0.2),
            base_channels=config.get("base_channels", 16),
            block_channels=config.get("block_channels"),
        )
    if config["model"] == "resnet18":
        return PretrainedResNet(num_classes, freeze_backbone=config["freeze_backbone"],
                                pretrained=config.get("pretrained", True))
    raise ValueError(config["model"])

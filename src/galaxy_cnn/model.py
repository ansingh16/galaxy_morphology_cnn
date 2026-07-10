"""The two models we compare.

``SmallCNN`` is a compact from-scratch conv net sized for the native 69x69
frames — three conv blocks into a small classifier head. ``build_resnet``
takes a pre-trained ResNet-18, freezes the convolutional backbone, and swaps in
a fresh head for our 10 classes: classic transfer learning where ImageNet's
low-level filters (edges, blobs, textures) transfer to galaxies for free and we
only learn the mapping from those features to morphology.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .data import NUM_CLASSES


class SmallCNN(nn.Module):
    """~0.5M-param conv net for 69x69x3 input."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),   # 69 -> 34
            block(32, 64),  # 34 -> 17
            block(64, 128),  # 17 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def build_resnet(num_classes: int = NUM_CLASSES, freeze: bool = True) -> nn.Module:
    """Pre-trained ResNet-18 with a fresh classifier head."""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze:
        for p in net.parameters():
            p.requires_grad = False
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return net


MODEL_KINDS = ("small", "resnet", "resnet_ft")


def build_model(kind: str) -> nn.Module:
    """Factory used by train/evaluate/serve so they all agree on architectures.

    'resnet' freezes the ImageNet backbone (feature extraction); 'resnet_ft'
    leaves it trainable for end-to-end fine-tuning.
    """
    if kind == "small":
        return SmallCNN()
    if kind == "resnet":
        return build_resnet(freeze=True)
    if kind == "resnet_ft":
        return build_resnet(freeze=False)
    raise ValueError(f"unknown model kind: {kind!r} (expected one of {MODEL_KINDS})")


def count_params(model: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

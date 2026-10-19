"""MobileNetV3-Small with ImageNet weights; the 1-channel log-Mel is replicated to 3 channels."""
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3Small(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_small(weights=weights, dropout=dropout)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, n_classes)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, mels, frames), globally standardized
        return self.net(x.expand(-1, 3, -1, -1))

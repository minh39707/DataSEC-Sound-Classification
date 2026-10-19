"""Small 2D CNN on log-Mel: 4 x [Conv3x3-BN-ReLU-MaxPool2x2] -> freq-mean, time max+mean pooling -> FC."""
import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int, channels=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers.append(conv_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, mels, frames)
        x = self.features(x)
        x = x.mean(dim=2)                        # average over frequency
        x = x.amax(dim=2) + x.mean(dim=2)        # max + mean over time
        return self.head(x)

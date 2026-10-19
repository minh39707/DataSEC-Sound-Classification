"""CRNN: CNN that pools mostly along frequency (keeps time resolution) -> BiGRU over time -> mean+max pooling -> FC."""
import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, n_classes: int, channels=(64, 128, 128, 128), gru_hidden: int = 128,
                 bidirectional: bool = True, dropout: float = 0.3, n_mels: int = 64):
        super().__init__()
        # frequency pooled by 2 every block (64 -> 4); time pooled only in the first two blocks (216 -> 54)
        pools = [(2, 2), (2, 2), (2, 1), (2, 1)]
        layers, c_in = [], 1
        for c, p in zip(channels, pools):
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(p), nn.Dropout(0.1)]
            c_in = c
        self.cnn = nn.Sequential(*layers)
        freq_out = n_mels // 16
        self.gru = nn.GRU(c_in * freq_out, gru_hidden, batch_first=True, bidirectional=bidirectional)
        d = gru_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, mels, frames)
        x = self.cnn(x)                               # (B, C, F', T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F')
        x, _ = self.gru(x)                            # (B, T', d)
        x = x.amax(dim=1) + x.mean(dim=1)
        return self.head(x)

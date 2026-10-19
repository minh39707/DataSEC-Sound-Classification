"""Dataset: cached waveform -> fixed-length crop/pad -> float tensor. Log-Mel is computed on the GPU later."""
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import load_classes, resolve


def crop_or_pad(wav: np.ndarray, n: int, mode: str, rng: np.random.Generator | None = None,
                pad_mode: str = "zero") -> np.ndarray:
    """Return exactly `n` samples.

    mode: 'random' (train), 'center' (eval), or 'energy' (window with the highest RMS, eval).
    pad_mode: 'zero' or 'repeat' (tile the clip) for clips shorter than `n`.
    """
    if len(wav) < n:
        if pad_mode == "repeat" and len(wav) > 0:
            wav = np.tile(wav, int(np.ceil(n / len(wav))))[:n]
            return wav
        out = np.zeros(n, dtype=wav.dtype)
        start = (n - len(wav)) // 2 if mode != "random" else (rng or np.random.default_rng()).integers(0, n - len(wav) + 1)
        out[start:start + len(wav)] = wav
        return out
    if len(wav) == n:
        return wav
    if mode == "random":
        start = (rng or np.random.default_rng()).integers(0, len(wav) - n + 1)
    elif mode == "center":
        start = (len(wav) - n) // 2
    elif mode == "energy":
        # window energy via cumulative sum, evaluated every n/10 samples
        csum = np.concatenate([[0.0], np.cumsum(wav.astype(np.float64) ** 2)])
        starts = np.arange(0, len(wav) - n + 1, max(n // 10, 1))
        start = int(starts[np.argmax(csum[starts + n] - csum[starts])])
    else:
        raise ValueError(mode)
    return wav[start:start + n]


class AudioClips(Dataset):
    def __init__(self, split: str, cfg: dict, train: bool,
                 perturb: Callable[[np.ndarray, str], np.ndarray] | None = None):
        self.df = pd.read_csv(resolve(cfg["dataset"]["splits_dir"]) / f"{split}.csv")
        self.split = split
        self.cache = resolve(cfg["dataset"]["cache_dir"])
        self.classes = load_classes()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = self.df["macro_class"].map(self.class_to_idx).to_numpy()
        a = cfg["audio"]
        self.n = int(round(a["sample_rate"] * a["duration_s"]))
        self.mode = a["train_crop"] if train else a["eval_crop"]
        self.pad_mode = a.get("pad_mode", "zero")
        self.perturb = perturb  # waveform-level corruption (test-time only)
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.df)

    def load(self, i: int) -> np.ndarray:
        return np.load(self.cache / f"{self.df.sample_id.iat[i]}.npy").astype(np.float32) / 32767.0

    def __getitem__(self, i: int):
        wav = crop_or_pad(self.load(i), self.n, self.mode, self.rng, self.pad_mode)
        if self.perturb is not None:
            wav = self.perturb(wav, self.df.sample_id.iat[i])
        return torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)), int(self.labels[i]), i


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


def make_loader(ds: AudioClips, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init,
                      persistent_workers=num_workers > 0, drop_last=False)


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights normalized to mean 1."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def compute_logmel_stats(ds: AudioClips, logmel, device: str, batch_size: int = 64) -> tuple[float, float]:
    """Global mean/std of un-normalized log-Mel over a (train) dataset."""
    s = ss = cnt = 0.0
    for wav, _, _ in make_loader(ds, batch_size, shuffle=False, num_workers=0):
        db = logmel.log_mel(wav.to(device)).double()
        s += db.sum().item()
        ss += (db ** 2).sum().item()
        cnt += db.numel()
    mean = s / cnt
    return mean, float(np.sqrt(ss / cnt - mean ** 2))

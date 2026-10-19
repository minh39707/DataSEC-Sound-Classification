"""Classification metrics and compute-cost measurements."""
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> dict:
    y_pred = probs.argmax(1)
    labels = np.arange(n_classes)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """MACs for one clip (batch 1) via thop. GRU is counted by thop's RNN hooks."""
    from thop import profile

    device = next(model.parameters()).device
    x = torch.zeros(1, *input_shape, device=device)
    was_training = model.training
    model.eval()
    macs, _ = profile(model, inputs=(x,), verbose=False)
    model.train(was_training)
    # thop registers buffers on modules; remove them so checkpoints stay clean
    for m in model.modules():
        for k in ("total_ops", "total_params"):
            m._buffers.pop(k, None)
    return int(macs)


@torch.no_grad()
def latency_ms(model: nn.Module, frontend: nn.Module, n_samples: int, device: str,
               warmup: int = 20, runs: int = 200) -> float:
    """Median end-to-end latency (waveform -> log-Mel -> logits) for one clip at batch size 1."""
    model.eval()
    wav = torch.randn(1, n_samples, device=device) * 0.1
    times = []
    for i in range(warmup + runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(frontend(wav))
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))

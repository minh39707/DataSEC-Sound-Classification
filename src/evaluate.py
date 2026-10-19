"""Inference + prediction CSV export in the schema required by the brief.

CSV columns: sample_id, true_class, pred_class, probabilities, duration, model, split, perturbation, severity, seed
(`probabilities` is a JSON list ordered as configs/classes.json.)

Usage:
    python -m src.evaluate --model small_cnn --seed 42 --split test
"""
import argparse
import json
from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.data import AudioClips, make_loader
from src.features import build_logmel
from src.metrics import classification_metrics
from src.models import build_model
from src.utils import RESULTS, load_config



@torch.no_grad()
def predict(model, logmel, loader, device: str,
            spec_perturb: Callable[[torch.Tensor, list[int]], torch.Tensor] | None = None) -> np.ndarray:
    """Return softmax probabilities (N, C) in dataset order.

    spec_perturb(db, idx) optionally corrupts the un-normalized log-Mel (B, mels, frames) before normalization.
    """
    model.eval()
    out = np.zeros((len(loader.dataset), len(loader.dataset.classes)), dtype=np.float32)
    for wav, _, idx in loader:
        wav = wav.to(device, non_blocking=True)
        if spec_perturb is None:
            x = logmel(wav)
        else:
            db = spec_perturb(logmel.log_mel(wav), idx.tolist())
            x = ((db - logmel.mean) / logmel.std).unsqueeze(1)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            logits = model(x)
        out[idx.numpy()] = torch.softmax(logits.float(), 1).cpu().numpy()
    return out


def predictions_frame(ds: AudioClips, probs: np.ndarray, model_name: str, seed: int,
                      perturbation: str = "none", severity: int = 0) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ds.df.sample_id,
        "true_class": ds.df.macro_class,
        "pred_class": [ds.classes[i] for i in probs.argmax(1)],
        "probabilities": [json.dumps([round(float(p), 6) for p in row]) for row in probs],
        "duration": ds.df.duration.round(3),
        "model": model_name,
        "split": ds.split,
        "perturbation": perturbation,
        "severity": severity,
        "seed": seed,
    })


def pred_path(model_name: str, seed: int, split: str, perturbation: str = "none", severity: int = 0):
    return RESULTS / "predictions" / f"{model_name}_seed{seed}_{split}_{perturbation}_{severity}.csv"


def load_trained(model_name: str, seed: int, cfg: dict, device: str):
    ckpt = torch.load(RESULTS / "checkpoints" / f"{model_name}_seed{seed}.pt", map_location=device)
    model = build_model(model_name, len(ckpt["classes"]), cfg).to(device)
    model.load_state_dict(ckpt["model"])
    logmel = build_logmel(cfg).to(device)
    logmel.set_stats(ckpt["logmel_mean"], ckpt["logmel_std"])
    return model, logmel


def evaluate_split(model, logmel, model_name: str, seed: int, split: str, cfg: dict, device: str) -> dict:
    ds = AudioClips(split, cfg, train=False)
    loader = make_loader(ds, cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])
    probs = predict(model, logmel, loader, device)
    path = pred_path(model_name, seed, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame(ds, probs, model_name, seed).to_csv(path, index=False)
    return classification_metrics(ds.labels, probs, len(ds.classes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, logmel = load_trained(args.model, args.seed, cfg, device)
    m = evaluate_split(model, logmel, args.model, args.seed, args.split, cfg, device)
    print(f"{args.model} seed={args.seed} {args.split}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

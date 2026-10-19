"""Resample every clip to the configured rate and cache it as int16 .npy (audio only — no spectrograms).

Speeds up Colab training: reading ~6 GB of 44.1 kHz WAV each epoch is IO-bound.
Amplitude is kept as-is (no peak normalization) so gain perturbations stay meaningful.

Usage:
    python scripts/cache_waveforms.py [--workers 4]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, resolve  # noqa: E402


def convert(args: tuple[str, str, int]) -> str:
    src, dst, target_sr = args
    if Path(dst).exists():
        return dst
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # mono (DataSEC is mono already; guard anyway)
    if sr != target_sr:
        wav = soxr.resample(wav, sr, target_sr, quality="HQ")
    np.save(dst, (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16))
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config()
    sr = cfg["audio"]["sample_rate"]
    raw = resolve(cfg["dataset"]["raw_dir"]) / "DATASEC"
    cache = resolve(cfg["dataset"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(resolve(cfg["dataset"]["index_csv"]))
    jobs = [(str(raw / p), str(cache / f"{sid}.npy"), sr) for sid, p in zip(df.sample_id, df.path)]
    with ProcessPoolExecutor(args.workers) as ex:
        list(tqdm(ex.map(convert, jobs, chunksize=8), total=len(jobs), desc=f"cache {sr} Hz"))
    print(f"cached {len(jobs)} clips to {cache}")


if __name__ == "__main__":
    main()

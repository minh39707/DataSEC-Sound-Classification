"""Data-quality checks for Midterm part A: overlapping clips, source bandwidth, level and fixed-length segments.

These are properties of how DataSEC was curated, not of the sounds themselves, and a model can learn them as
shortcuts. Writes:
  results/tables/eda_overlap_pairs.csv    clip pairs that share audio (identical file, one is a prefix of the
                                          other, or the same first seconds) and the split partition of each clip
  results/tables/eda_signal_quality.csv   per class: level, peak normalisation, high-frequency energy,
                                          source bandwidth and the most common clip length
  results/figures/eda_signal_quality.png
Needs data/index.csv (scripts/build_index.py), the extracted raw audio and splits/*.csv.

Usage:
    python scripts/eda_quality.py [--workers 8]
"""
import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS, load_config, resolve  # noqa: E402

CLIFF_EDGES_HZ = [4000, 5512.5, 8000, 11025, 16000, 19000]  # standard band edges of resampled / encoded audio
CLIFF_DB = 25.0      # PSD drop across an edge that marks a band-limited source
PEAK_NORM_DB = -0.1  # peak at or above this = clip was peak-normalised to full scale


def analyse(path: str) -> dict:
    """Level, high-frequency energy and source bandwidth of one clip (native 44.1 kHz)."""
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    data = data[:, 0]
    x = data.astype(np.float32) / 32768.0
    f, psd = signal.welch(x, fs=sr, nperseg=min(4096, len(x)))
    tot = psd.sum() + 1e-30
    psd_db = 10 * np.log10(psd + 1e-20)
    bandwidth = sr / 2
    for fc in CLIFF_EDGES_HZ:
        below = np.median(psd_db[(f >= 0.80 * fc) & (f < 0.97 * fc)])
        above = np.median(psd_db[(f > 1.03 * fc) & (f <= min(1.2 * fc, 0.497 * sr))])
        if below - above > CLIFF_DB:
            bandwidth = fc
            break
    return {"peak_dbfs": float(20 * np.log10(max(np.abs(x).max(), 1e-10))),
            "rms_dbfs": float(10 * np.log10(max(np.mean(x.astype(np.float64) ** 2), 1e-20))),
            "energy_above_8k": float(psd[f >= 8000].sum() / tot),
            "energy_above_11k": float(psd[f >= 11025].sum() / tot),
            "source_bandwidth_hz": float(bandwidth),
            "md5_first_second": hashlib.md5(data[:sr].tobytes()).hexdigest()}


def overlap_pairs(df: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """Pairs of clips that share audio: identical files (md5) or the same first second (verified sample by sample)."""
    rows = []
    for _, g in df[df.duplicated("md5", keep=False)].groupby("md5"):
        ids = g.sort_values("sample_id")
        for b in ids.iloc[1:].itertuples():
            rows.append((ids.iloc[0].sample_id, b.sample_id, "identical file", b.duration))
    same_start = df[(df.duration >= 1) & ~df.duplicated("md5", keep=False)
                    & df.duplicated("md5_first_second", keep=False)]
    for _, g in same_start.groupby("md5_first_second"):
        g = g.sort_values("duration")
        a = sf.read(raw / g.iloc[0].path, dtype="int16", always_2d=True)[0][:, 0]
        for b_row in g.iloc[1:].itertuples():
            b = sf.read(raw / b_row.path, dtype="int16", always_2d=True)[0][:, 0]
            m = min(len(a), len(b))
            differ = np.flatnonzero(a[:m] != b[:m])
            shared = (differ[0] if len(differ) else m) / g.iloc[0].sr
            relation = "prefix of the longer clip" if not len(differ) and m == len(a) else "same first seconds"
            rows.append((g.iloc[0].sample_id, b_row.sample_id, relation, shared))
    return pd.DataFrame(rows, columns=["sample_a", "sample_b", "relation", "shared_s"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    fig_dir, tab_dir = RESULTS / "figures", RESULTS / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)
    raw = resolve(cfg["dataset"]["raw_dir"]) / "DATASEC"
    df = pd.read_csv(resolve(cfg["dataset"]["index_csv"]))

    with ProcessPoolExecutor(args.workers) as ex:
        stats = list(tqdm(ex.map(analyse, [str(raw / p) for p in df.path], chunksize=16), total=len(df), desc="clips"))
    df = pd.concat([df, pd.DataFrame(stats)], axis=1)

    # 1. overlapping clips and where the split put them
    split_of = pd.concat([pd.read_csv(resolve(cfg["dataset"]["splits_dir"]) / f"{s}.csv").assign(split=s)
                          for s in ("train", "val", "test")]).set_index("sample_id").split
    pairs = overlap_pairs(df, raw)
    pairs["macro_class"] = pairs.sample_a.map(df.set_index("sample_id").macro_class)
    pairs["split_a"], pairs["split_b"] = pairs.sample_a.map(split_of), pairs.sample_b.map(split_of)
    pairs["crosses_split"] = pairs.split_a != pairs.split_b
    pairs.round(2).to_csv(tab_dir / "eda_overlap_pairs.csv", index=False)
    print(f"overlapping pairs: {len(pairs)} ({pairs.relation.value_counts().to_dict()}) | "
          f"crossing partitions: {int(pairs.crosses_split.sum())}")
    if pairs.crosses_split.any():
        print(pairs[pairs.crosses_split].to_string(index=False))

    # 2. per-class curation artefacts
    bucket = pd.cut(df.source_bandwidth_hz, [0, 8000, 11025, 19000, np.inf],
                    labels=["cliff <= 8 kHz", "cliff at 11.025 kHz", "cliff at 16-19 kHz", "full band"])
    common_len = df.duration.round(1)
    q = df.assign(peak_normalised=df.peak_dbfs >= PEAK_NORM_DB, over_1pct_above_8k=df.energy_above_8k > 0.01,
                  over_1pct_above_11k=df.energy_above_11k > 0.01, common_len=common_len)
    tab = q.groupby("macro_class").agg(
        n=("sample_id", "size"), rms_dbfs_median=("rms_dbfs", "median"), peak_dbfs_median=("peak_dbfs", "median"),
        pct_peak_normalised=("peak_normalised", "mean"),
        energy_above_8k_pct_median=("energy_above_8k", "median"),
        pct_over_1pct_energy_above_8k=("over_1pct_above_8k", "mean"),
        pct_over_1pct_energy_above_11k=("over_1pct_above_11k", "mean"),
        most_common_length_s=("common_len", lambda s: s.mode().iloc[0]),
        pct_most_common_length=("common_len", lambda s: (s == s.mode().iloc[0]).mean()))
    for c in ["pct_peak_normalised", "energy_above_8k_pct_median", "pct_over_1pct_energy_above_8k",
              "pct_over_1pct_energy_above_11k", "pct_most_common_length"]:
        tab[c] *= 100
    bw = pd.crosstab(df.macro_class, bucket, normalize="index") * 100
    tab = tab.join(bw.add_prefix("pct_")).sort_values("n", ascending=False)
    tab.round(2).to_csv(tab_dir / "eda_signal_quality.csv")
    print(tab.round(1).to_string())
    print(f"all clips: >1% energy above 8 kHz {100 * q.over_1pct_above_8k.mean():.1f}% | "
          f"above 11.025 kHz {100 * q.over_1pct_above_11k.mean():.1f}%")

    order = list(tab.index[::-1])
    y = np.arange(len(order))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharey=True)
    axes[0].boxplot([df.loc[df.macro_class == c, "rms_dbfs"] for c in order], positions=y, vert=False,
                    showfliers=False, widths=0.6)
    axes[0].set_yticks(y, order)
    axes[0].set_xlabel("RMS level (dBFS)"); axes[0].set_title("Clip level")
    axes[1].barh(y, tab.loc[order, "pct_peak_normalised"], color="#4C72B0")
    axes[1].set_xlabel("% clips with peak at 0 dBFS"); axes[1].set_title("Peak-normalised clips")
    left = np.zeros(len(order))
    for col, colour in zip(bw.columns, ["#1f3b73", "#4C72B0", "#a6c1e3", "#d9d9d9"]):
        v = bw.reindex(order)[col].fillna(0).values
        axes[2].barh(y, v, left=left, color=colour, label=col); left += v
    axes[2].set_xlabel("% clips"); axes[2].set_title("Estimated source bandwidth (PSD cliff > 25 dB)")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(fig_dir / "eda_signal_quality.png", dpi=150); plt.close()
    print(f"-> {tab_dir}/eda_overlap_pairs.csv, eda_signal_quality.csv, {fig_dir}/eda_signal_quality.png")


if __name__ == "__main__":
    main()

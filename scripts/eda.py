"""Exploratory data analysis for Midterm part A: writes results/figures/eda_*.png and results/tables/eda_*.csv.

Covers clip counts per macro-class, class balance, duration statistics (+ duration-only shortcut check),
waveform / linear spectrogram / log-Mel of contrasting classes, the Mel filterbank, class-mean log-Mel
spectra and spectral similarity between classes. Needs data/index.csv and the extracted raw audio.

Usage:
    python scripts/eda.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import soxr
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import class_weights, crop_or_pad
from src.features import build_logmel
from src.utils import RESULTS, load_classes, load_config, resolve

# contrasting sound types: impulsive / harmonic / broadband / speech
EXAMPLES = {"Gunshot (impulsive)": "Thunder fireworks and gunshot/Gunshot",
            "Sirens (harmonic, tonal)": "Sirens and alarms/Sirens",
            "Jet aircraft (broadband)": "Jet aircrafts",
            "Voices (speech)": "Voices"}


def main() -> None:
    cfg, classes = load_config(), load_classes()
    fig_dir, tab_dir = RESULTS / "figures", RESULTS / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)
    raw = resolve(cfg["dataset"]["raw_dir"]) / "DATASEC"
    df = pd.read_csv(resolve(cfg["dataset"]["index_csv"]))
    sr = cfg["audio"]["sample_rate"]; win = cfg["audio"]["duration_s"]; n = int(sr * win)
    logmel = build_logmel(cfg)
    sns.set_theme(style="whitegrid", context="notebook")

    def load(path: str) -> np.ndarray:
        w, s = sf.read(raw / path, dtype="float32", always_2d=True); w = w.mean(1)
        return soxr.resample(w, s, sr) if s != sr else w

    print(f"{len(df)} clips | {df.macro_class.nunique()} macro | {df.subclass.nunique()} leaf | "
          f"{df.duration.sum() / 3600:.2f} h | sample rates {df.sr.unique()} | channels {df.channels.unique()}")

    # 1. class distribution and balance
    counts = df.macro_class.value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.sort_values().plot.barh(ax=ax, color="#4C72B0")
    for i, v in enumerate(counts.sort_values()):
        ax.text(v + 10, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("number of clips"); ax.set_title(f"DataSEC clips per macro-class (N={len(df)})")
    plt.tight_layout(); plt.savefig(fig_dir / "eda_class_counts.png", dpi=150); plt.close()

    p = counts / counts.sum()
    entropy = -(p * np.log(p)).sum() / np.log(len(p))
    print(f"imbalance max/min = {counts.max()}/{counts.min()} = {counts.max() / counts.min():.1f}x "
          f"({counts.idxmax()} / {counts.idxmin()}) | normalized entropy = {entropy:.3f} | "
          f"top-2 classes share = {p.iloc[:2].sum():.1%}")
    df.groupby(["macro_class", "subclass"]).size().rename("n").reset_index().to_csv(
        tab_dir / "eda_subclass_counts.csv", index=False)

    # how training compensates: inverse-frequency loss weights on the train split
    tr = pd.read_csv(resolve(cfg["dataset"]["splits_dir"]) / "train.csv")
    w = class_weights(tr.macro_class.map({c: i for i, c in enumerate(classes)}).to_numpy(), len(classes)).numpy()
    cw = pd.DataFrame({"train_clips": tr.macro_class.value_counts().reindex(classes).values,
                       "loss_weight": w.round(2)}, index=classes).sort_values("train_clips")
    print("class-weighted loss (train split):\n" + cw.to_string())

    # 2. duration statistics per class
    dur = df.groupby("macro_class").duration.agg(["count", "min", "max", "mean", "median", "sum"])
    dur["sum"] = dur["sum"] / 3600
    dur = dur.rename(columns={"sum": "total_h"}).sort_values("median")
    dur.round(2).to_csv(tab_dir / "eda_duration_stats.csv")
    print(dur.round(2).to_string())

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(data=df, y="macro_class", x="duration", order=dur.index.tolist(), ax=ax, fliersize=2, color="#DD8452")
    ax.set_xscale("log"); ax.axvline(win, color="k", ls="--", lw=1, label=f"{win:g} s model window")
    ax.set_xlabel("duration (s, log scale)"); ax.set_ylabel(""); ax.legend()
    plt.tight_layout(); plt.savefig(fig_dir / "eda_duration_box.png", dpi=150); plt.close()
    print(f"clips shorter than {win:g} s (padded): {(df.duration < win).mean():.1%} | "
          f"longer (cropped): {(df.duration > win).mean():.1%}")

    # duration as a shortcut cue: how well does log-duration alone predict the class?
    f1 = cross_val_score(HistGradientBoostingClassifier(random_state=0), np.log(df[["duration"]].values),
                         df.macro_class.values, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1_macro")
    print(f"duration-only macro-F1 = {f1.mean():.3f} +/- {f1.std():.3f} (chance ~ {1 / len(classes):.3f})")

    # 3. waveform, linear spectrogram, log-Mel for contrasting classes (energy-centred 5 s window)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, len(EXAMPLES), figsize=(4.2 * len(EXAMPLES), 9))
    for j, (title, folder) in enumerate(EXAMPLES.items()):
        cand = df[df.path.str.startswith(folder + "/")]
        row = cand.iloc[rng.integers(len(cand))]
        x = crop_or_pad(load(row.path), n, "energy")
        t = np.arange(len(x)) / sr
        axes[0, j].plot(t, x, lw=0.4); axes[0, j].set_title(f"{title}\n{row.sample_id}", fontsize=9)
        axes[0, j].set_xlabel("time (s)"); axes[0, j].set_ylabel("amplitude")
        spec = torch.stft(torch.tensor(x), 1024, 512, window=torch.hann_window(1024), return_complex=True).abs() ** 2
        axes[1, j].imshow(10 * np.log10(spec.numpy() + 1e-10), origin="lower", aspect="auto", cmap="magma",
                          extent=[0, len(x) / sr, 0, sr / 2 / 1000])
        axes[1, j].set_ylabel("frequency (kHz)"); axes[1, j].set_xlabel("time (s)")
        lm = logmel.log_mel(torch.tensor(x)[None])[0].numpy()
        axes[2, j].imshow(lm, origin="lower", aspect="auto", cmap="magma", extent=[0, len(x) / sr, 0, lm.shape[0]])
        axes[2, j].set_ylabel("Mel bin"); axes[2, j].set_xlabel("time (s)")
        axes[1, j].grid(False); axes[2, j].grid(False)
    for r, lab in enumerate(["waveform", "|STFT|^2 (dB)", "log-Mel (dB)"]):
        axes[r, 0].annotate(lab, (-0.35, 0.5), xycoords="axes fraction", rotation=90, va="center", fontsize=11)
    plt.tight_layout(); plt.savefig(fig_dir / "eda_representations.png", dpi=150); plt.close()

    # Mel scale and the project's actual triangular filterbank
    fb = logmel.mel.mel_scale.fb.numpy()                       # (n_fft/2 + 1 freq bins, n_mels filters)
    freqs = np.linspace(0, sr / 2, fb.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    f = np.linspace(0, sr / 2, 500)
    axes[0].plot(f / 1000, 2595 * np.log10(1 + f / 700), label="mel(f) = 2595 log10(1 + f/700)")
    axes[0].plot(f / 1000, f, "--", color="gray", label="mel = Hz (reference)")
    axes[0].set_xlabel("frequency (kHz)"); axes[0].set_ylabel("mel"); axes[0].set_ylim(0, 4000)
    axes[0].set_title("Mel scale (HTK formula, for illustration)"); axes[0].legend()
    for b in range(fb.shape[1]):
        axes[1].plot(freqs / 1000, fb[:, b] / fb[:, b].max(), lw=0.8)
    axes[1].set_xlabel("frequency (kHz)"); axes[1].set_ylabel("weight (normalized)")
    axes[1].set_title(f"{fb.shape[1]} Slaney Mel filters used by src/features.py: narrow at low, wide at high frequency")
    plt.tight_layout(); plt.savefig(fig_dir / "eda_mel_filterbank.png", dpi=150); plt.close()
    bw = [freqs[fb[:, b] > 0].max() - freqs[fb[:, b] > 0].min() for b in (0, fb.shape[1] - 1)]
    print(f"Mel filter width: first ~{bw[0]:.0f} Hz, last ~{bw[1]:.0f} Hz")

    # 4. class-mean log-Mel spectra (up to 40 random clips per class) and spectral similarity
    rows = []
    for c in classes:
        sel = df[df.macro_class == c].sample(min(40, (df.macro_class == c).sum()), random_state=0)
        specs = [logmel.log_mel(torch.tensor(crop_or_pad(load(p), n, "energy"))[None])[0].mean(1).numpy()
                 for p in sel.path]
        rows.append(np.mean(specs, 0))
    mean_spec = pd.DataFrame(rows, index=classes)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(mean_spec, cmap="magma", ax=ax, cbar_kws={"label": "mean log-Mel (dB)"})
    ax.set_xlabel("Mel bin"); ax.set_title("Class-mean log-Mel spectrum")
    plt.tight_layout(); plt.savefig(fig_dir / "eda_class_mean_spectra.png", dpi=150); plt.close()

    # All classes share a steep low-frequency tilt, which makes raw cosine similarity ~0.99 everywhere.
    # Remove each class's overall level AND the cross-class mean spectrum, then compare what is left.
    z = mean_spec.sub(mean_spec.mean(axis=1), axis=0)
    z = z - z.mean(axis=0)
    z = z.div(np.linalg.norm(z, axis=1), axis=0)
    sim = pd.DataFrame(z.values @ z.values.T, index=classes, columns=classes)
    g = sns.clustermap(sim, cmap="vlag", center=0, figsize=(11, 11), cbar_kws={"label": "cosine similarity"})
    g.savefig(fig_dir / "eda_spectral_similarity.png", dpi=150); plt.close("all")

    pairs = sim.where(np.triu(np.ones(sim.shape, bool), 1)).stack().sort_values(ascending=False)
    pairs.head(10).rename("cosine").to_csv(tab_dir / "eda_similar_pairs.csv")
    print("most similar class pairs:\n" + pairs.head(10).round(3).to_string())
    print(f"-> {fig_dir}/eda_*.png, {tab_dir}/eda_*.csv")


if __name__ == "__main__":
    main()

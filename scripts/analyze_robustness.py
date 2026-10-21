"""RQ1 analysis from saved prediction CSVs only (no model is run).

Outputs (results/tables, results/figures):
  rq1_delta_f1.csv            macro-F1 clean / corrupted / delta, mean and std over seeds, per model x condition
  rq1_bootstrap.csv           paired bootstrap 95% CIs of delta-F1 differences between models (severity 3)
  rq1_per_class_delta.csv     per-class F1 drop (mean over seeds) per model x perturbation x severity
  rq1_confusion_shift.csv     largest increases in confusion counts, clean -> severity 3 (summed over seeds)
  rq1_flips.csv               clips correct on clean but wrong when corrupted (all seeds of the model agree)
  rq1_duration_split.csv      delta-F1 for clips <= 5 s vs > 5 s
  rq1_degradation.png, rq1_per_class_delta.png, rq1_confusion_shift_<pert>.png

Usage:
    python scripts/analyze_robustness.py [--n-boot 1000]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import pred_path  # noqa: E402
from src.utils import RESULTS, load_classes, load_config  # noqa: E402

MODELS = ["small_cnn", "mobilenetv3s", "crnn"]
SEEDS = [42, 43, 44]
KINDS = ["noise", "gain", "tmask", "fmask"]
TITLES = {"noise": "Pink noise (SNR dB)", "gain": "Gain (dB)", "tmask": "Time mask (fraction of window)",
          "fmask": "Frequency mask (of 64 Mel bands)"}
TAB, FIG = RESULTS / "tables", RESULTS / "figures"


def load_pred(model, seed, kind, sev, class_to_idx):
    df = pd.read_csv(pred_path(model, seed, "test", kind, sev))
    return df.true_class.map(class_to_idx).to_numpy(), df.pred_class.map(class_to_idx).to_numpy(), df


def macro_f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def per_class_f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    return np.where(cm.sum(0) + cm.sum(1) > 0, 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config()
    classes = load_classes()
    k = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    levels = {kd: cfg["robustness"][kd]["levels"] for kd in KINDS}
    conds = [("none", 0)] + [(kd, s) for kd in KINDS for s in (1, 2, 3)]

    P = {}  # (model, seed, kind, sev) -> (y, pred)
    for m in MODELS:
        for s in SEEDS:
            for kd, sev in conds:
                y, p, df = load_pred(m, s, kd, sev, c2i)
                P[m, s, kd, sev] = (y, p)
    dur = df.duration.to_numpy()
    y_true = P[MODELS[0], SEEDS[0], "none", 0][0]

    # ---- 1. delta F1 table
    rows = []
    for m in MODELS:
        clean = np.array([macro_f1(*P[m, s, "none", 0], k) for s in SEEDS])
        for kd, sev in conds[1:]:
            cor = np.array([macro_f1(*P[m, s, kd, sev], k) for s in SEEDS])
            d = clean - cor
            rows.append({"model": m, "perturbation": kd, "severity": sev, "level": levels[kd][sev - 1],
                         "f1_clean_mean": clean.mean(), "f1_clean_std": clean.std(ddof=1),
                         "f1_corrupted_mean": cor.mean(), "f1_corrupted_std": cor.std(ddof=1),
                         "delta_f1_mean": d.mean(), "delta_f1_std": d.std(ddof=1)})
    delta = pd.DataFrame(rows)
    delta.to_csv(TAB / "rq1_delta_f1.csv", index=False)
    print(delta.pivot_table(index=["perturbation", "severity"], columns="model", values="delta_f1_mean").round(3))

    # ---- 2. degradation curves
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    colors = {"small_cnn": "#1f77b4", "mobilenetv3s": "#d95f02", "crnn": "#2ca02c"}
    for ax, kd in zip(axes, KINDS):
        for m in MODELS:
            d = delta[(delta.model == m) & (delta.perturbation == kd)].sort_values("level")
            clean = d.f1_clean_mean.iat[0]
            x = list(d.level)
            ax.errorbar(x, d.f1_corrupted_mean, yerr=d.f1_corrupted_std, marker="o", capsize=3,
                        color=colors[m], label=m)
            ax.axhline(clean, color=colors[m], ls=":", lw=0.8)
        ax.set_title(TITLES[kd], fontsize=10)
        ax.set_xlabel("level")
        ax.grid(alpha=0.3)
        if kd == "noise":
            ax.invert_xaxis()
    axes[0].set_ylabel("test macro-F1 (mean ± std, 3 seeds)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "rq1_degradation.png", dpi=150)
    plt.close(fig)

    # ---- 3. per-class delta
    pc = []
    for m in MODELS:
        for kd, sev in conds[1:]:
            base = np.mean([per_class_f1(*P[m, s, "none", 0], k) for s in SEEDS], 0)
            cor = np.mean([per_class_f1(*P[m, s, kd, sev], k) for s in SEEDS], 0)
            for ci, c in enumerate(classes):
                pc.append({"model": m, "perturbation": kd, "severity": sev, "class": c,
                           "f1_clean": base[ci], "f1_corrupted": cor[ci], "delta_f1": base[ci] - cor[ci]})
    pc = pd.DataFrame(pc)
    pc.to_csv(TAB / "rq1_per_class_delta.csv", index=False)
    hm = pc[pc.severity == 3].groupby(["class", "perturbation"]).delta_f1.mean().unstack()[KINDS]
    hm = hm.loc[hm.mean(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(6.5, 7))
    im = ax.imshow(hm.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), KINDS)
    ax.set_yticks(range(k), hm.index, fontsize=8)
    for i in range(k):
        for j in range(4):
            ax.text(j, i, f"{hm.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if hm.values[i, j] > 0.6 else "black")
    ax.set_title("Per-class F1 drop at severity 3 (mean over 3 models x 3 seeds)", fontsize=10)
    fig.colorbar(im, ax=ax, label="clean F1 - corrupted F1")
    fig.tight_layout()
    fig.savefig(FIG / "rq1_per_class_delta.png", dpi=150)
    plt.close(fig)

    # ---- 4. confusion shift (clean -> severity 3), summed over models and seeds
    shifts = []
    for kd in KINDS:
        cm0 = np.zeros((k, k), int)
        cm3 = np.zeros((k, k), int)
        for m in MODELS:
            for s in SEEDS:
                y, p = P[m, s, "none", 0]
                cm0 += np.bincount(y * k + p, minlength=k * k).reshape(k, k)
                y, p = P[m, s, kd, 3]
                cm3 += np.bincount(y * k + p, minlength=k * k).reshape(k, k)
        n_runs = len(MODELS) * len(SEEDS)
        diff = (cm3 - cm0) / n_runs
        off = diff.copy()
        np.fill_diagonal(off, 0)
        for flat in np.argsort(off, axis=None)[::-1][:10]:
            i, j = divmod(int(flat), k)
            shifts.append({"perturbation": kd, "true_class": classes[i], "pred_class": classes[j],
                           "extra_clips_per_run": off[i, j]})
        rown = cm0.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(100 * (cm3 - cm0) / np.maximum(rown, 1), cmap="RdBu_r", vmin=-100, vmax=100)
        ax.set_xticks(range(k), classes, rotation=90, fontsize=6)
        ax.set_yticks(range(k), classes, fontsize=6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Row-normalised confusion change, {kd} severity 3 minus clean (%)", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(FIG / f"rq1_confusion_shift_{kd}.png", dpi=130)
        plt.close(fig)
    pd.DataFrame(shifts).to_csv(TAB / "rq1_confusion_shift.csv", index=False)

    # ---- 5. flips: correct on clean and wrong when corrupted, for every seed of the model
    flips = []
    ids = df.sample_id.to_numpy()
    for m in MODELS:
        for kd, sev in conds[1:]:
            ok_clean = np.all([P[m, s, "none", 0][1] == y_true for s in SEEDS], 0)
            bad = np.all([P[m, s, kd, sev][1] != y_true for s in SEEDS], 0)
            for i in np.flatnonzero(ok_clean & bad):
                flips.append({"model": m, "perturbation": kd, "severity": sev, "sample_id": ids[i],
                              "true_class": classes[y_true[i]], "duration": dur[i],
                              "pred_class_seed42": classes[P[m, 42, kd, sev][1][i]]})
    flips = pd.DataFrame(flips)
    flips.to_csv(TAB / "rq1_flips.csv", index=False)

    # ---- 6. duration split
    drows = []
    for name, mask in [("<=5 s", dur <= 5.0), (">5 s", dur > 5.0)]:
        for m in MODELS:
            for kd, sev in conds[1:]:
                d = [macro_f1(P[m, s, "none", 0][0][mask], P[m, s, "none", 0][1][mask], k)
                     - macro_f1(P[m, s, kd, sev][0][mask], P[m, s, kd, sev][1][mask], k) for s in SEEDS]
                drows.append({"duration_group": name, "n": int(mask.sum()), "model": m, "perturbation": kd,
                              "severity": sev, "delta_f1_mean": np.mean(d), "delta_f1_std": np.std(d, ddof=1)})
    pd.DataFrame(drows).to_csv(TAB / "rq1_duration_split.csv", index=False)

    # ---- 7. paired bootstrap over test clips (same resample for every model, seed and condition)
    rng = np.random.default_rng(0)
    n = len(y_true)
    idx = rng.integers(0, n, size=(args.n_boot, n))
    pairs = [("small_cnn", "mobilenetv3s"), ("small_cnn", "crnn"), ("mobilenetv3s", "crnn")]
    brow = []
    for kd in KINDS:
        d = {m: np.zeros(args.n_boot) for m in MODELS}
        for m in MODELS:
            for b in range(args.n_boot):
                ii = idx[b]
                d[m][b] = np.mean([macro_f1(y_true[ii], P[m, s, "none", 0][1][ii], k)
                                   - macro_f1(y_true[ii], P[m, s, kd, 3][1][ii], k) for s in SEEDS])
        for a, b2 in pairs:
            diff = d[a] - d[b2]
            brow.append({"perturbation": kd, "severity": 3, "model_a": a, "model_b": b2,
                         "delta_a_minus_delta_b_mean": diff.mean(),
                         "ci95_low": np.percentile(diff, 2.5), "ci95_high": np.percentile(diff, 97.5),
                         "significant": bool(np.percentile(diff, 2.5) > 0 or np.percentile(diff, 97.5) < 0)})
    pd.DataFrame(brow).to_csv(TAB / "rq1_bootstrap.csv", index=False)
    print(pd.DataFrame(brow).round(3).to_string(index=False))
    print("flips:", len(flips), "-> tables/rq1_*.csv, figures/rq1_*.png")


if __name__ == "__main__":
    main()

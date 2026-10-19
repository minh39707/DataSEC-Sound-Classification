"""Midterm part D figures: per-class F1 and row-normalized confusion matrices on the clean test set.

Everything is recomputed from the saved prediction CSVs (results/predictions/*_none_0.csv) and checked
against the JSON training logs; no model is re-run.

Usage:
    python scripts/plot_metrics.py [--seed 42]
    # -> results/figures/test_per_class_f1.png, results/figures/confusion_{model}.png
    #    results/tables/test_per_class_f1.csv
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS, load_classes

MODELS = ["small_cnn", "mobilenetv3s", "crnn"]
COLORS = {"small_cnn": "#2a78d6", "mobilenetv3s": "#eb6834", "crnn": "#1baf7a"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
SHORT = {"Cat fights and moans": "Cat", "Cicadas and crickets": "Cicadas/crickets",
         "Crows seagulls and magpies": "Crows/gulls", "Dog barkings and howlings": "Dog",
         "Lawn mower brush cutter and olive shaker": "Lawn mower/cutter", "Sirens and alarms": "Sirens/alarms",
         "Thunder fireworks and gunshot": "Thunder/gunshot", "Vacuum cleaner fan and hairdryer": "Vacuum/fan/dryer"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    classes = load_classes()
    fig_dir, tab_dir = RESULTS / "figures", RESULTS / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)

    pred, logs = {}, {}
    for m in MODELS:
        pred[m] = pd.read_csv(RESULTS / "predictions" / f"{m}_seed{args.seed}_test_none_0.csv")
        logs[m] = json.loads((RESULTS / "logs" / f"{m}_seed{args.seed}.json").read_text(encoding="utf-8"))
        f1 = f1_score(pred[m].true_class, pred[m].pred_class, labels=classes, average="macro", zero_division=0)
        assert abs(f1 - logs[m]["test"]["macro_f1"]) < 1e-9, f"{m}: CSV and log disagree"

    # per-class F1, hardest classes on top (sorted by the mean over the three models)
    f1c = pd.DataFrame({m: f1_score(pred[m].true_class, pred[m].pred_class, labels=classes, average=None,
                                    zero_division=0) for m in MODELS}, index=classes)
    f1c["mean"] = f1c[MODELS].mean(axis=1)
    f1c["n_test"] = pred[MODELS[0]].true_class.value_counts().reindex(classes).fillna(0).astype(int).values
    f1c = f1c.sort_values("mean")
    f1c.iloc[::-1].round(4).to_csv(tab_dir / "test_per_class_f1.csv", index_label="class")

    fig, ax = plt.subplots(figsize=(10, 11))
    y, h = np.arange(len(f1c)), 0.26
    for k, m in enumerate(MODELS):
        ax.barh(y + (k - 1) * h, f1c[m], height=h * 0.9, color=COLORS[m], label=m)
    ax.set_yticks(y, [f"{c}  (n={n})" for c, n in zip(f1c.index, f1c.n_test)], fontsize=9)
    ax.set_xlim(0, 1.0); ax.set_xlabel("test F1"); ax.grid(axis="y", visible=False)
    ax.set_title(f"Per-class test F1, seed {args.seed} (hardest classes on top)")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout(); plt.savefig(fig_dir / "test_per_class_f1.png", dpi=150); plt.close()

    # confusion matrices, row-normalized (diagonal = per-class recall); counts are in the JSON logs
    short = [SHORT.get(c, c) for c in classes]
    for m in MODELS:
        cm = confusion_matrix(pred[m].true_class, pred[m].pred_class, labels=classes)
        assert (cm == np.array(logs[m]["test"]["confusion_matrix"])).all(), f"{m}: confusion matrix != log"
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(11, 9.5))
        im = ax.imshow(cmn, cmap=BLUES, vmin=0, vmax=1)
        for i, j in zip(*np.where(cmn >= 0.10)):
            ax.text(j, i, f"{cmn[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.6 else "#1a1a19")
        ax.set_xticks(range(len(classes)), short, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), short, fontsize=8)
        ax.set_xlabel("predicted class"); ax.set_ylabel("true class"); ax.grid(False)
        ax.set_title(f"{m}: test confusion matrix, row-normalized (macro-F1 {logs[m]['test']['macro_f1']:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.04, label="fraction of true class")
        plt.tight_layout(); plt.savefig(fig_dir / f"confusion_{m}.png", dpi=150); plt.close()

    print(f1c.iloc[::-1].round(2).to_string())
    print(f"-> {fig_dir}/test_per_class_f1.png, confusion_*.png | {tab_dir}/test_per_class_f1.csv")


if __name__ == "__main__":
    main()

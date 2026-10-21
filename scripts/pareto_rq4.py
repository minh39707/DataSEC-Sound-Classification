"""Extension RQ4: accuracy-efficiency Pareto from logs and RQ1 metrics (no model is run).

Cost axes available from the logs: params, MACs, latency (batch 1), training time. Peak VRAM was not logged.
Outputs results/tables/rq4_pareto.csv and results/figures/rq4_pareto.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS  # noqa: E402

MODELS, SEEDS = ["small_cnn", "mobilenetv3s", "crnn"], [42, 43, 44]
logs = {(m, s): json.load(open(RESULTS / "logs" / f"{m}_seed{s}.json")) for m in MODELS for s in SEEDS}
r = pd.read_csv(RESULTS / "tables" / "rq1_metrics.csv")
rows = []
for m in MODELS:
    f1c = r[(r.model == m) & (r.perturbation == "none")].macro_f1
    f1x = r[(r.model == m) & (r.perturbation != "none")].groupby("seed").macro_f1.mean()
    L = [logs[m, s] for s in SEEDS]
    rows.append({"model": m, "clean_f1": f1c.mean(), "mean_corrupted_f1": f1x.mean(),
                 "params_M": L[0]["params"] / 1e6, "MACs_M": L[0]["macs"] / 1e6,
                 "latency_ms": np.mean([x["latency_ms"] for x in L]),
                 "train_min": np.mean([x["train_time_s"] for x in L]) / 60})
t = pd.DataFrame(rows)


def pareto(df, cost, score):
    """True where no other model is at least as good on both axes and strictly better on one."""
    out = []
    for _, row in df.iterrows():
        dominated = ((df[cost] <= row[cost]) & (df[score] >= row[score]) &
                     ((df[cost] < row[cost]) | (df[score] > row[score]))).any()
        out.append(not dominated)
    return out


for cost in ["latency_ms", "MACs_M", "params_M", "train_min"]:
    for score in ["clean_f1", "mean_corrupted_f1"]:
        t[f"pareto_{score}_{cost}"] = pareto(t, cost, score)
t.to_csv(RESULTS / "tables" / "rq4_pareto.csv", index=False)
print(t.round(3).T.to_string())

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, cost in zip(axes.ravel(), ["latency_ms", "MACs_M", "params_M", "train_min"]):
    for _, row in t.iterrows():
        ax.scatter(row[cost], row.clean_f1, marker="o", s=60)
        ax.scatter(row[cost], row.mean_corrupted_f1, marker="x", s=60)
        ax.annotate(row.model, (row[cost], row.clean_f1), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(cost)
    ax.set_ylabel("macro-F1 (o clean, x mean of 12 corruptions)")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / "figures" / "rq4_pareto.png", dpi=150)

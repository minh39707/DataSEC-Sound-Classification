"""Midterm questions E1-E4 re-answered over seeds 42/43/44 (clean test set), from saved predictions only.

Outputs results/tables/midterm_ms_*.csv. Per-class numbers are mean +- std over seeds; the CRNN-vs-CNN comparison by
clip duration is reported per seed so that its sign consistency is visible.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS, load_classes  # noqa: E402

MODELS, SEEDS = ["small_cnn", "mobilenetv3s", "crnn"], [42, 43, 44]
classes = load_classes()
k = len(classes)
c2i = {c: i for i, c in enumerate(classes)}
TAB = RESULTS / "tables"


def pc_f1(y, p):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    d = cm.sum(0) + cm.sum(1)
    return np.where(d > 0, 2 * np.diag(cm) / np.maximum(d, 1), 0.0)


def macro(y, p):
    present = np.unique(y)
    return float(np.mean(pc_f1(y, p)[present]))  # classes present in the group, as in midterm_questions.py E3


P, dur = {}, None
for m in MODELS:
    for s in SEEDS:
        df = pd.read_csv(RESULTS / "predictions" / f"{m}_seed{s}_test_none_0.csv")
        P[m, s] = (df.true_class.map(c2i).to_numpy(), df.pred_class.map(c2i).to_numpy())
        dur = df.duration.to_numpy()
y = P[MODELS[0], 42][0]

# E1: per-class F1 over seeds
rows = []
for ci, c in enumerate(classes):
    r = {"class": c, "n_test": int((y == ci).sum())}
    allv = []
    for m in MODELS:
        v = [pc_f1(*P[m, s])[ci] for s in SEEDS]
        r[f"{m}_mean"], r[f"{m}_std"] = np.mean(v), np.std(v, ddof=1)
        allv += v
    r["all_mean"], r["all_std"] = np.mean(allv), np.std(allv, ddof=1)
    rows.append(r)
e1 = pd.DataFrame(rows).sort_values("all_mean", ascending=False)
e1.to_csv(TAB / "midterm_ms_e1_per_class.csv", index=False)
print("E1 best:", list(e1["class"][:4]), "worst:", list(e1["class"][-5:]))

# E2: confusion pairs summed over models and seeds
cm = np.zeros((k, k), int)
for key, (yy, pp) in P.items():
    cm += np.bincount(yy * k + pp, minlength=k * k).reshape(k, k)
pairs = [{"A": classes[i], "B": classes[j], "A_to_B": cm[i, j] / 9, "B_to_A": cm[j, i] / 9, "total_per_run": (cm[i, j] + cm[j, i]) / 9}
         for i in range(k) for j in range(i + 1, k)]
e2 = pd.DataFrame(pairs).sort_values("total_per_run", ascending=False)
e2.to_csv(TAB / "midterm_ms_e2_confusion_pairs.csv", index=False)
print(e2.head(8).round(2).to_string(index=False))

# E3: CRNN - small_cnn delta macro-F1 by duration group, per seed
groups = {"short (<=5 s)": dur <= 5, "medium (5-15 s)": (dur > 5) & (dur <= 15), "long (>15 s)": dur > 15}
rows = []
for g, mask in groups.items():
    ds = [macro(P["crnn", s][0][mask], P["crnn", s][1][mask]) - macro(P["small_cnn", s][0][mask], P["small_cnn", s][1][mask])
          for s in SEEDS]
    rows.append({"group": g, "n": int(mask.sum()), **{f"delta_seed{s}": d for s, d in zip(SEEDS, ds)},
                 "delta_mean": np.mean(ds), "delta_std": np.std(ds, ddof=1)})
e3 = pd.DataFrame(rows)
e3.to_csv(TAB / "midterm_ms_e3_crnn_vs_cnn_by_duration.csv", index=False)
print(e3.round(3).to_string(index=False))

# overall 3-seed baselines
rows = []
for m in MODELS:
    f = [macro(*P[m, s]) for s in SEEDS]
    a = [(P[m, s][0] == P[m, s][1]).mean() for s in SEEDS]
    rows.append({"model": m, "test_macro_f1_mean": np.mean(f), "test_macro_f1_std": np.std(f, ddof=1),
                 "test_acc_mean": np.mean(a), "test_acc_std": np.std(a, ddof=1)})
pd.DataFrame(rows).to_csv(TAB / "midterm_ms_baselines.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))

"""Midterm questions E1-E4 from the saved clean test predictions (no model is re-run).

E1  which classes are recognised best / worst          -> midterm_e1_per_class.csv, midterm_e1_class_ranking.csv
E2  which classes are confused most                    -> midterm_e2_confusion_pairs.csv, midterm_e2_hard_clips.csv,
                                                          midterm_e2_eda_overlap.csv (needs eda_spectral_similarity.csv
                                                          from scripts/eda.py), midterm_e2_audio_audit.csv + figures
                                                          (need audio, see below)
E3  does the CRNN beat the plain CNN on clips of       -> midterm_e3_duration_class_counts.csv,
    different original duration                           midterm_e3_duration_metrics.csv,
                                                          midterm_e3_paired_outcomes.csv, midterm_e3_duration_comparison.png
E4  accuracy-cost trade-off                            -> midterm_e4_tradeoff.csv, midterm_e4_tradeoff.png

Prediction CSVs are checked against the JSON logs first, as in plot_metrics.py. The E2 audio audit reads the
waveform cache (data/cache_22k, the exact model input) or, if it is missing, the raw 44.1 kHz wav files; its figures
also need torchaudio. Steps whose inputs are missing are skipped and their existing outputs are left untouched.

Usage:
    python scripts/midterm_questions.py [--seed 42]
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
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS, load_classes, load_config, resolve  # noqa: E402

MODELS = ["small_cnn", "mobilenetv3s", "crnn"]
COLORS = {"small_cnn": "#2a78d6", "mobilenetv3s": "#eb6834", "crnn": "#1baf7a"}
# original clip duration vs the 5 s model window: padded / up to 3x the window / longer
DURATION_BINS = [0, 5, 15, np.inf]
DURATION_LABELS = ["short (<=5 s)", "medium (5-15 s)", "long (>15 s)"]
# Clips chosen by hand for listening: misclassified by at least 2 of the 3 seed-42 models, one per true class.
# The full candidate list is midterm_e2_hard_clips.csv.
AUDIT_IDS = ["Crows-0022", "Jet aircrafts-0003", "Lawn mower-0007", "Music-0004", "Helicopters-0035", "Train-0005",
             "fan-0019", "motorbike idling-0013", "car pass-by-0011", "Glass breaking-0001"]
AUDIT_FIGURES = 3  # spectrogram figures for the first clips of AUDIT_IDS


def load_predictions(seed: int, classes: list[str]) -> tuple[dict, dict]:
    pred, logs = {}, {}
    for m in MODELS:
        p = pd.read_csv(RESULTS / "predictions" / f"{m}_seed{seed}_test_none_0.csv").set_index("sample_id")
        logs[m] = json.loads((RESULTS / "logs" / f"{m}_seed{seed}.json").read_text(encoding="utf-8"))
        f1 = f1_score(p.true_class, p.pred_class, labels=classes, average="macro", zero_division=0)
        assert abs(f1 - logs[m]["test"]["macro_f1"]) < 1e-9, f"{m}: CSV and log disagree"
        p["p_pred"] = [max(json.loads(s)) for s in p.probabilities]
        pred[m] = p
    ids = pred[MODELS[0]].index
    assert all(pred[m].index.sort_values().equals(ids.sort_values()) for m in MODELS), "models saw different clips"
    pred = {m: pred[m].loc[ids] for m in MODELS}
    return pred, logs


def e1_classes(pred: dict, classes: list[str], tab_dir: Path) -> None:
    rows = []
    for m in MODELS:
        p, r, f, n = precision_recall_fscore_support(pred[m].true_class, pred[m].pred_class, labels=classes,
                                                     zero_division=0)
        rows.append(pd.DataFrame({"model": m, "class_name": classes, "precision": p, "recall": r, "f1": f,
                                  "n_test": n}))
    per_class = pd.concat(rows, ignore_index=True)
    per_class.to_csv(tab_dir / "midterm_e1_per_class.csv", index=False)

    rank = per_class.pivot(index="class_name", columns="model", values="f1")[MODELS]
    rank["F1_mean_3_models"] = rank[MODELS].mean(axis=1)
    rank["recall_mean_3_models"] = per_class.groupby("class_name").recall.mean()  # recall = per-class accuracy
    rank["n_test"] = per_class.groupby("class_name").n_test.first()
    rank = rank.sort_values("F1_mean_3_models", ascending=False, kind="stable")
    rank.to_csv(tab_dir / "midterm_e1_class_ranking.csv")
    print("E1 best:\n" + rank.head(3).round(3).to_string() + "\nE1 worst:\n" + rank.tail(3).round(3).to_string())


def e2_confusions(pred: dict, classes: list[str], tab_dir: Path) -> pd.DataFrame:
    """Unordered class pairs with at least one error, counted in both directions."""
    rows = []
    for m in MODELS:
        p = pred[m]
        n = p.true_class.value_counts()
        cnt = pd.crosstab(p.true_class, p.pred_class).reindex(index=classes, columns=classes, fill_value=0)
        for i, a in enumerate(classes):
            for b in classes[i + 1:]:
                ab, ba = int(cnt.at[a, b]), int(cnt.at[b, a])
                if ab + ba == 0:
                    continue
                r_ab, r_ba = ab / n[a], ba / n[b]
                rows.append({"model": m, "A": a, "B": b, "A_to_B": ab, "B_to_A": ba, "n_A": n[a], "n_B": n[b],
                             "total_errors": ab + ba, "rate_A_to_B": r_ab, "rate_B_to_A": r_ba,
                             "pair_rate": (r_ab + r_ba) / 2,
                             "direction": "two-way" if ab and ba else "one-way"})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(tab_dir / "midterm_e2_confusion_pairs.csv", index=False)
    top = (pairs.groupby(["A", "B"]).total_errors.sum().sort_values(ascending=False, kind="stable").head(8))
    print("E2 most confused pairs (errors summed over the 3 models):\n" + top.to_string())

    # clips that at least 2 of the 3 models get wrong: candidates for listening
    first = pred[MODELS[0]]
    hard = pd.DataFrame({"true_class": first.true_class, "duration_s": first.duration})
    for m in MODELS:
        hard[m], hard[f"{m}_p_pred"] = pred[m].pred_class, pred[m].p_pred
    hard["n_models_wrong"] = sum((pred[m].pred_class != first.true_class).astype(int) for m in MODELS)
    hard = hard[hard.n_models_wrong >= 2].sort_values(["n_models_wrong", "true_class"], ascending=[False, True],
                                                      kind="stable")
    hard.to_csv(tab_dir / "midterm_e2_hard_clips.csv")
    print(f"E2 clips wrong for >= 2 models: {len(hard)} ({(hard.n_models_wrong == 3).sum()} wrong for all 3)")
    return pairs


def e2_eda_overlap(pairs: pd.DataFrame, classes: list[str], tab_dir: Path) -> None:
    """Spectral similarity from the dataset analysis (A6) next to the error counts of each class pair."""
    sim_path = tab_dir / "eda_spectral_similarity.csv"
    if not sim_path.exists():
        print(f"SKIP midterm_e2_eda_overlap.csv: {sim_path.name} missing (run scripts/eda.py)")
        return
    sim = pd.read_csv(sim_path, index_col=0).reindex(index=classes, columns=classes)
    errors = pairs.pivot_table(index=["A", "B"], columns="model", values="total_errors", fill_value=0)
    rows = []
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            row = {"A": a, "B": b, "cosine": sim.at[a, b]}
            for m in MODELS:
                row[f"{m}_errors"] = int(errors[m].get((a, b), 0)) if m in errors else 0
            rows.append(row)
    out = pd.DataFrame(rows).sort_values("cosine", ascending=False, kind="stable")
    out.to_csv(tab_dir / "midterm_e2_eda_overlap.csv", index=False)
    err = out[[f"{m}_errors" for m in MODELS]].sum(axis=1)
    print(f"E2 Spearman(cosine, errors summed over models) = {out.cosine.corr(err, method='spearman'):.3f} "
          f"over {len(out)} pairs")


def e2_audio_audit(pred: dict, cfg: dict, fig_dir: Path, tab_dir: Path) -> None:
    """What the model actually saw for the hand-picked clips: padding, crop position and kept energy."""
    sr, n_sec = cfg["audio"]["sample_rate"], cfg["audio"]["duration_s"]
    cache = resolve(cfg["dataset"]["cache_dir"])
    if cache.is_dir():
        source = "cache"
    else:
        index_csv = resolve(cfg["dataset"]["index_csv"])
        if not index_csv.exists():
            print("SKIP midterm_e2_audio_audit.csv: neither the waveform cache nor data/index.csv is available")
            return
        import soundfile as sf
        index = pd.read_csv(index_csv).set_index("sample_id")
        raw = resolve(cfg["dataset"]["raw_dir"]) / "DATASEC"
        source = "raw"
    from src.data import crop_or_pad

    def load(sid: str) -> tuple[np.ndarray, int]:
        if source == "cache":
            return np.load(cache / f"{sid}.npy").astype(np.float32) / 32767.0, sr
        w, s = sf.read(raw / index.at[sid, "path"], dtype="float32", always_2d=True)
        return w.mean(axis=1), s

    first = pred[MODELS[0]]
    rows, windows = [], {}
    for sid in AUDIT_IDS:
        wav, s = load(sid)
        n = int(round(s * n_sec))
        win = crop_or_pad(wav, n, cfg["audio"]["eval_crop"], pad_mode=cfg["audio"].get("pad_mode", "zero"))
        windows[sid] = (win, s)
        row = {"sample_id": sid, "true_class": first.at[sid, "true_class"], "duration_s": len(wav) / s}
        for m in MODELS:
            row[m], row[f"{m}_p_pred"] = pred[m].at[sid, "pred_class"], pred[m].at[sid, "p_pred"]
        row["n_models_wrong"] = sum(pred[m].at[sid, "pred_class"] != row["true_class"] for m in MODELS)
        row["crop_start_s"] = max(len(wav) - n, 0) // 2 / s
        row["pad_fraction"] = max(n - len(wav), 0) / n
        row["duration_fraction_seen"] = min(1.0, n / len(wav))
        row["energy_kept"] = float(np.sum(win.astype(np.float64) ** 2) / np.sum(wav.astype(np.float64) ** 2))
        row["observation"] = (f"zero padding is {100 * row['pad_fraction']:.1f}% of the input" if len(wav) < n else
                              f"centre {n_sec:g} s of {row['duration_s']:.2f} s, "
                              f"{100 * row['energy_kept']:.0f}% of the clip energy")
        rows.append(row)
    pd.DataFrame(rows).to_csv(tab_dir / "midterm_e2_audio_audit.csv", index=False)
    print(f"E2 audio audit ({source} audio): {len(rows)} clips -> midterm_e2_audio_audit.csv")

    try:
        import torch
        from src.features import build_logmel
    except ImportError as e:
        print(f"SKIP midterm_e2_audit_*.png: {e}")
        return
    logmel = build_logmel(cfg)
    for sid in AUDIT_IDS[:AUDIT_FIGURES]:
        win, s = windows[sid]
        if s != sr:
            print(f"SKIP midterm_e2_audit_{sid}.png: needs the {sr} Hz waveform cache")
            continue
        db = logmel.log_mel(torch.from_numpy(win)[None])[0].numpy()
        fig, axes = plt.subplots(1, 2, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1.2]})
        axes[0].plot(np.arange(len(win)) / s, win, lw=0.4)
        axes[0].set_xlabel("time in model input (s)"); axes[0].set_ylabel("amplitude")
        axes[0].set_title("Window seen by the model")
        im = axes[1].imshow(db, origin="lower", aspect="auto", cmap="magma", extent=[0, len(win) / s, 0, db.shape[0]])
        axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("Mel band"); axes[1].grid(False)
        axes[1].set_title("log-Mel before normalisation")
        fig.colorbar(im, ax=axes[1], label="dB")
        fig.suptitle(f"{sid} (true: {first.at[sid, 'true_class']})", fontsize=10)
        plt.tight_layout(); plt.savefig(fig_dir / f"midterm_e2_audit_{sid}.png", dpi=150); plt.close()


def e3_duration(pred: dict, classes: list[str], seed: int, fig_dir: Path, tab_dir: Path) -> None:
    """Metrics inside groups of original clip duration. Only classes present in a group enter its macro-F1."""
    first = pred[MODELS[0]]
    group = pd.cut(first.duration, DURATION_BINS, labels=DURATION_LABELS)
    counts = pd.crosstab(first.true_class, group).reindex(index=classes, columns=DURATION_LABELS, fill_value=0)
    counts.rename_axis(index="macro_class", columns=None).to_csv(tab_dir / "midterm_e3_duration_class_counts.csv")

    rows, paired = [], []
    for g in DURATION_LABELS:
        mask = (group == g).to_numpy()
        present = sorted(first.true_class[mask].unique())
        f1 = {}
        for m in MODELS:
            p = pred[m][mask]
            f1[m] = f1_score(p.true_class, p.pred_class, labels=present, average="macro", zero_division=0)
            rows.append({"group": g, "model": m, "n": int(mask.sum()), "n_classes": len(present),
                         "accuracy": float((p.pred_class == p.true_class).mean()), "F1_present": f1[m]})
        cnn = (pred["small_cnn"].pred_class == first.true_class)[mask]
        crnn = (pred["crnn"].pred_class == first.true_class)[mask]
        only_cnn, only_crnn = int((cnn & ~crnn).sum()), int((~cnn & crnn).sum())
        paired.append({"group": g, "n": int(mask.sum()), "both_correct": int((cnn & crnn).sum()),
                       "only_CNN_correct": only_cnn, "only_CRNN_correct": only_crnn,
                       "both_wrong": int((~cnn & ~crnn).sum()), "delta_F1_CRNN_minus_CNN": f1["crnn"] - f1["small_cnn"],
                       # exact McNemar test on the discordant clips (one seed, so it ignores training variance)
                       "mcnemar_exact_p": binomtest(only_cnn, only_cnn + only_crnn).pvalue if only_cnn + only_crnn else 1.0})
    metrics, paired = pd.DataFrame(rows), pd.DataFrame(paired)
    metrics.to_csv(tab_dir / "midterm_e3_duration_metrics.csv", index=False)
    paired.to_csv(tab_dir / "midterm_e3_paired_outcomes.csv", index=False)
    print("E3 CRNN vs small_cnn by duration group:\n" + paired.round(3).to_string(index=False))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x, w = np.arange(len(DURATION_LABELS)), 0.26
    for k, m in enumerate(MODELS):
        v = metrics[metrics.model == m].set_index("group").F1_present.reindex(DURATION_LABELS)
        bars = ax.bar(x + (k - 1) * w, v, width=w * 0.95, color=COLORS[m], label=m)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    n = metrics.drop_duplicates("group").set_index("group").reindex(DURATION_LABELS)
    ax.set_xticks(x, [f"{g}\nn={r.n}, {r.n_classes} classes" for g, r in n.iterrows()])
    ax.set_ylim(0, 1.15); ax.set_ylabel("macro-F1 over classes present")
    ax.set_title(f"Test macro-F1 by original clip duration, seed {seed}")
    ax.legend(loc="upper center", ncol=3, frameon=True)
    plt.tight_layout(); plt.savefig(fig_dir / "midterm_e3_duration_comparison.png", dpi=150); plt.close()


def pareto(cost: pd.Series, score: pd.Series) -> pd.Series:
    """True if no other model is at least as cheap and at least as accurate, and strictly better in one."""
    return pd.Series([not any((cost[o] <= cost[m]) and (score[o] >= score[m]) and
                              ((cost[o] < cost[m]) or (score[o] > score[m])) for o in cost.index if o != m)
                      for m in cost.index], index=cost.index)


def e4_tradeoff(logs: dict, fig_dir: Path, tab_dir: Path) -> None:
    t = pd.DataFrame([{"model": m, "val_F1": L["val"]["macro_f1"], "test_F1": L["test"]["macro_f1"],
                       "test_acc": L["test"]["accuracy"], "params_M": L["params"] / 1e6, "MACs_M": L["macs"] / 1e6,
                       "latency_ms": L["latency_ms"], "train_min": L["train_time_s"] / 60,
                       "epochs_run": len(L["history"])} for m, L in logs.items()]).set_index("model")
    for c in ["latency_ms", "MACs_M", "params_M"]:
        t[f"pareto_{c}"] = pareto(t[c], t.test_F1)
    t.to_csv(tab_dir / "midterm_e4_tradeoff.csv")
    print("E4 trade-off:\n" + t.round(3).to_string())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, c, label in zip(axes, ["latency_ms", "MACs_M"], ["latency (ms / clip, T4, batch 1)", "MACs (M / clip)"]):
        for m in t.index:
            ax.scatter(t.at[m, c], t.at[m, "test_F1"], s=90, color=COLORS[m], zorder=3)
            ax.annotate(m, (t.at[m, c], t.at[m, "test_F1"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
        front = t[t[f"pareto_{c}"]].sort_values(c)
        ax.plot(front[c], front.test_F1, "--", color="gray", lw=1, label="Pareto front")
        ax.set_xlim(0, t[c].max() * 1.2); ax.set_ylim(t.test_F1.min() - 0.03, t.test_F1.max() + 0.03)
        ax.set_xlabel(label); ax.set_ylabel("test macro-F1"); ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Accuracy-cost trade-off (lower cost and higher F1 is better)", fontsize=11)
    plt.tight_layout(); plt.savefig(fig_dir / "midterm_e4_tradeoff.png", dpi=150); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg, classes = load_config(), load_classes()
    fig_dir, tab_dir = RESULTS / "figures", RESULTS / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)

    pred, logs = load_predictions(args.seed, classes)
    e1_classes(pred, classes, tab_dir)
    pairs = e2_confusions(pred, classes, tab_dir)
    e2_eda_overlap(pairs, classes, tab_dir)
    e2_audio_audit(pred, cfg, fig_dir, tab_dir)
    e3_duration(pred, classes, args.seed, fig_dir, tab_dir)
    e4_tradeoff(logs, fig_dir, tab_dir)
    print(f"-> {tab_dir}/midterm_e*.csv, {fig_dir}/midterm_e*.png")


if __name__ == "__main__":
    main()

"""Build the Midterm baseline table (T2.3) from training logs + clean prediction CSVs.

Accuracy / macro-F1 are recomputed from results/predictions/*_none_0.csv and checked against the
JSON log. Runs whose split_md5 differs from the committed splits/ are refused.

Usage:
    python scripts/make_baseline_table.py     # -> results/tables/midterm_baselines.csv
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS, load_config, resolve  # noqa: E402

MODELS = ["small_cnn", "mobilenetv3s", "crnn"]  # table row order


def split_ids(splits_dir: Path, split: str) -> set[str]:
    return set(pd.read_csv(splits_dir / f"{split}.csv").sample_id)


def scores_from_csv(path: Path, classes: list[str], expected_ids: set[str]) -> tuple[float, float]:
    df = pd.read_csv(path)
    assert set(df.sample_id) == expected_ids and len(df) == len(expected_ids), f"{path.name}: ids != split"
    acc = accuracy_score(df.true_class, df.pred_class)
    f1 = f1_score(df.true_class, df.pred_class, labels=classes, average="macro", zero_division=0)
    return float(acc), float(f1)


def main() -> None:
    cfg = load_config()
    results = RESULTS
    splits_dir = resolve(cfg["dataset"]["splits_dir"])
    classes = json.loads(resolve("configs/classes.json").read_text(encoding="utf-8"))
    canonical = {s: hashlib.md5((splits_dir / f"{s}.csv").read_bytes()).hexdigest() for s in ["train", "val", "test"]}
    ids = {s: split_ids(splits_dir, s) for s in ["val", "test"]}

    rows = []
    for log_path in sorted((results / "logs").glob("*_seed*.json")):
        L = json.loads(log_path.read_text(encoding="utf-8"))
        if L.get("split_md5") != canonical:
            print(f"SKIP {log_path.name}: trained on a non-canonical split")
            continue
        row = {"model": L["model"], "seed": L["seed"]}
        for split in ["val", "test"]:
            csv = results / "predictions" / f"{L['model']}_seed{L['seed']}_{split}_none_0.csv"
            acc, f1 = scores_from_csv(csv, classes, ids[split])
            assert abs(acc - L[split]["accuracy"]) < 1e-6 and abs(f1 - L[split]["macro_f1"]) < 1e-6, \
                f"{csv.name}: CSV scores differ from {log_path.name}"
            row[f"{split}_acc"], row[f"{split}_macro_f1"] = acc, f1
        row.update({
            "params_M": L["params"] / 1e6,
            "MACs_M": L["macs"] / 1e6,
            "latency_ms": L["latency_ms"],
            "train_time_min": L["train_time_s"] / 60,
            "best_epoch": L["best_epoch"],
            "epochs_run": len(L["history"]),
            "gpu": L["env"].get("gpu"),
        })
        rows.append(row)

    assert rows, "no valid runs in results/logs"
    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], [m for m in MODELS if m in set(df.model)] +
                                 sorted(set(df.model) - set(MODELS)), ordered=True)
    df = df.sort_values(["model", "seed"]).reset_index(drop=True)
    out = results / "tables" / "midterm_baselines.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

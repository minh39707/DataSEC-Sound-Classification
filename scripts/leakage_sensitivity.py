"""Re-score saved test predictions without the test clips that share audio with a train/val clip.

The fixed split groups exact duplicates only; scripts/eda_quality.py also finds clips that are a prefix of another
clip or start with the same seconds. This script measures how much those few test clips change the reported
metrics, without retraining. Reads results/tables/eda_overlap_pairs.csv and results/predictions/*_test_*.csv.
Writes results/tables/midterm_leakage_sensitivity.csv.

Usage:
    python scripts/leakage_sensitivity.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS, load_classes  # noqa: E402


def main() -> None:
    classes = load_classes()
    pairs = pd.read_csv(RESULTS / "tables" / "eda_overlap_pairs.csv")
    cross = pairs[pairs.crosses_split]
    leaked = set(cross.loc[cross.split_a == "test", "sample_a"]) | set(cross.loc[cross.split_b == "test", "sample_b"])
    print(f"test clips sharing audio with train/val: {sorted(leaked)}")

    rows = []
    for f in sorted((RESULTS / "predictions").glob("*_test_none_0.csv")):
        p = pd.read_csv(f)
        for subset, d in (("all test clips", p), ("without overlapping clips", p[~p.sample_id.isin(leaked)])):
            rows.append({"model": p.model.iat[0], "seed": p.seed.iat[0], "subset": subset, "n": len(d),
                         "test_acc": accuracy_score(d.true_class, d.pred_class),
                         "test_macro_f1": f1_score(d.true_class, d.pred_class, labels=classes, average="macro",
                                                   zero_division=0)})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(RESULTS / "tables" / "midterm_leakage_sensitivity.csv", index=False)
    wide = out.pivot_table(index=["model", "seed"], columns="subset", values="test_macro_f1")
    wide["delta"] = wide["without overlapping clips"] - wide["all test clips"]
    print(out.round(4).to_string(index=False))
    print("\nmacro-F1 change when the overlapping clips are removed:\n" + wide.round(4).to_string())


if __name__ == "__main__":
    main()

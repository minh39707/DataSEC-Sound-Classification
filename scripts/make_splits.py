"""Create the fixed train/val/test split (70/15/15), stratified by leaf class, grouped by duplicate audio.

Stratifying on the leaf class (subclass, or macro-class when it has no subfolders) also keeps
macro-class proportions. Clips sharing a group id (exact md5 duplicates, plus any near-duplicate
groups in data/near_duplicates.csv) always land in the same partition, so no audio leaks into test.

Usage:
    python scripts/make_splits.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, resolve  # noqa: E402

N_FOLDS = 20  # 1 fold = 5% -> 3 folds test, 3 folds val, 14 folds train


def build_groups(df: pd.DataFrame, near_dup_csv: Path) -> pd.Series:
    group = df["md5"].copy()
    if near_dup_csv.exists():  # columns: sample_id, group
        nd = pd.read_csv(near_dup_csv).set_index("sample_id")["group"]
        hit = df["sample_id"].isin(nd.index)
        group[hit] = "nd_" + df.loc[hit, "sample_id"].map(nd).astype(str)
    return group


def main() -> None:
    cfg = load_config()
    seed = cfg["split"]["seed"]
    df = pd.read_csv(resolve(cfg["dataset"]["index_csv"]))
    groups = build_groups(df, resolve("data/near_duplicates.csv"))

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=df.index)
    for k, (_, idx) in enumerate(sgkf.split(df, df["subclass"], groups)):
        fold.iloc[idx] = k
    df["split"] = fold.map(lambda k: "test" if k < 3 else "val" if k < 6 else "train")

    # sanity: every macro-class in every split, and no group crosses partitions
    per = pd.crosstab(df["macro_class"], df["split"])
    assert (per > 0).all().all(), f"a class is missing from a split:\n{per}"
    assert (df.groupby(groups)["split"].nunique() == 1).all(), "duplicate group crosses splits"

    out_dir = resolve(cfg["dataset"]["splits_dir"])
    out_dir.mkdir(exist_ok=True)
    cols = ["sample_id", "path", "macro_class", "subclass", "duration"]
    for name in ["train", "val", "test"]:
        df.loc[df.split == name, cols].sort_values("sample_id").to_csv(out_dir / f"{name}.csv", index=False)

    meta = {
        "seed": seed,
        "method": f"StratifiedGroupKFold(n_splits={N_FOLDS}) on leaf class; folds 0-2 test, 3-5 val, rest train",
        "groups": "md5 exact duplicates (+ data/near_duplicates.csv if present)",
        "n_groups_multi": int((groups.value_counts() > 1).sum()),
        "sizes": df.split.value_counts().to_dict(),
        "per_macro_class": per.to_dict(orient="index"),
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(per.assign(total=per.sum(axis=1)).to_string())
    print(meta["sizes"])


if __name__ == "__main__":
    main()

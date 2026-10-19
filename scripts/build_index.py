"""Scan extracted DataSEC and write data/index.csv with one row per clip.

Columns: sample_id, path, macro_class, subclass, sr, channels, frames, duration, subtype, md5
Exact-duplicate audio (same md5) is reported so splits can keep duplicates in one partition.

Usage:
    python scripts/build_index.py
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, load_classes, resolve  # noqa: E402


def main() -> None:
    cfg = load_config()
    root = resolve(cfg["dataset"]["raw_dir"]) / "DATASEC"
    classes = load_classes()

    rows = []
    for wav in tqdm(sorted(root.rglob("*.wav")), desc="index"):
        rel = wav.relative_to(root)
        macro = rel.parts[0]
        sub = rel.parts[1] if len(rel.parts) == 3 else macro
        info = sf.info(str(wav))
        rows.append({
            "sample_id": wav.stem,
            "path": rel.as_posix(),
            "macro_class": macro,
            "subclass": sub,
            "sr": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration": info.frames / info.samplerate,
            "subtype": info.subtype,
            "md5": hashlib.md5(wav.read_bytes()).hexdigest(),
        })

    df = pd.DataFrame(rows)
    assert df["sample_id"].is_unique, "sample_id (file stem) is not unique"
    assert set(df["macro_class"]) == set(classes), "macro-class folders differ from configs/classes.json"

    out = resolve(cfg["dataset"]["index_csv"])
    df.to_csv(out, index=False)

    print(f"clips={len(df)}  macro={df.macro_class.nunique()}  leaf={df.subclass.nunique()}  "
          f"hours={df.duration.sum() / 3600:.2f}")
    print("sample rates:", df.sr.value_counts().to_dict())
    print("channels:", df.channels.value_counts().to_dict())
    print("subtypes:", df.subtype.value_counts().to_dict())
    dups = df[df.duplicated("md5", keep=False)].sort_values("md5")
    print(f"exact-duplicate clips: {len(dups)} in {dups.md5.nunique()} groups")
    if len(dups):
        dups.to_csv(out.with_name("exact_duplicates.csv"), index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

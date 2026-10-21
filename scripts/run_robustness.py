"""RQ1: evaluate saved checkpoints on the clean test set corrupted at test time. No training, no test-set tuning.

Writes results/predictions/{model}_seed{s}_test_{perturbation}_{severity}.csv (severity 1..3; 0 = clean) and
results/tables/rq1_metrics.csv (one row per model x seed x condition).

Usage:
    python scripts/run_robustness.py [--models small_cnn mobilenetv3s crnn] [--seeds 42 43 44] [--force]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import AudioClips, make_loader  # noqa: E402
from src.evaluate import load_trained, pred_path, predictions_frame  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402
from src.perturb import SPEC_TYPES, make_power_hook, make_waveform_perturb  # noqa: E402
from src.utils import RESULTS, load_config  # noqa: E402


@torch.no_grad()
def predict_condition(model, logmel, ds, kind, level, device, batch_size):
    # num_workers=0: the corruption closures are not picklable, and 758 clips are cheap on one process
    loader = make_loader(ds, batch_size, shuffle=False, num_workers=0)
    model.eval()
    out = np.zeros((len(ds), len(ds.classes)), dtype=np.float32)
    for wav, _, idx in loader:
        wav = wav.to(device)
        if kind in SPEC_TYPES:
            ids = [ds.df.sample_id.iat[i] for i in idx.tolist()]
            db = logmel.log_mel(wav, power_hook=make_power_hook(kind, level, ids))
            x = ((db - logmel.mean) / logmel.std).unsqueeze(1)
        else:
            x = logmel(wav)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            logits = model(x)
        out[idx.numpy()] = torch.softmax(logits.float(), 1).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["small_cnn", "mobilenetv3s", "crnn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conditions = [("none", 0, 0.0)] + [(k, s + 1, lv) for k, spec in cfg["robustness"].items()
                                       for s, lv in enumerate(spec["levels"])]
    rows = []
    for model_name in args.models:
        for seed in args.seeds:
            model, logmel = load_trained(model_name, seed, cfg, device)
            for kind, sev, level in conditions:
                ds = AudioClips("test", cfg, train=False,
                                perturb=None if kind == "none" else make_waveform_perturb(kind, level))
                path = pred_path(model_name, seed, "test", kind, sev)
                if path.exists() and not args.force:
                    probs = np.stack(pd.read_csv(path).probabilities.map(json.loads).to_numpy())
                else:
                    probs = predict_condition(model, logmel, ds, kind, level, device, cfg["train"]["batch_size"])
                    predictions_frame(ds, probs, model_name, seed, kind, sev).to_csv(path, index=False)
                m = classification_metrics(ds.labels, probs, len(ds.classes))
                rows.append({"model": model_name, "seed": seed, "perturbation": kind, "severity": sev, "level": level,
                             "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
                print(f"{model_name} s{seed} {kind:5s} sev{sev} level={level:<6} acc={m['accuracy']:.4f} "
                      f"F1={m['macro_f1']:.4f}", flush=True)
    out = RESULTS / "tables" / "rq1_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out)


if __name__ == "__main__":
    main()

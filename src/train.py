"""Train one model with one seed; select the checkpoint by validation macro-F1 (the test set is never used here).

After training, writes val + test prediction CSVs and a JSON log with metrics, cost, and environment.

Usage:
    python -m src.train --model small_cnn --seed 42
    python -m src.train --model crnn --seed 42 --epochs 2 --limit 256   # quick smoke test
"""
import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from src.data import AudioClips, class_weights, compute_logmel_stats, make_loader
from src.evaluate import evaluate_split, predict
from src.features import build_logmel
from src.metrics import classification_metrics, count_macs, count_params, latency_ms
from src.models import build_model
from src.utils import RESULTS, env_info, load_config, resolve, set_seed



def get_logmel_stats(cfg: dict, logmel, device: str) -> tuple[float, float]:
    """Train-set log-Mel mean/std, cached per feature config + exact train ids (eval crop, deterministic)."""
    train_ids = (resolve(cfg["dataset"]["splits_dir"]) / "train.csv").read_bytes()
    key = json.dumps({"audio": cfg["audio"], "features": cfg["features"],
                      "train_split_md5": hashlib.md5(train_ids).hexdigest()}, sort_keys=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "logmel_stats.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    if key not in cache:
        ds = AudioClips("train", cfg, train=False)
        mean, std = compute_logmel_stats(ds, logmel, device)
        cache[key] = {"mean": mean, "std": std}
        path.write_text(json.dumps(cache, indent=2))
    return cache[key]["mean"], cache[key]["std"]


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["small_cnn", "mobilenetv3s", "crnn"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only N train/val clips (smoke test)")
    ap.add_argument("--tag", default="", help="suffix for run name, e.g. _specaug")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg["train"]
    epochs = args.epochs or tc["epochs"]
    run = f"{args.model}{args.tag}"
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = tc["amp"] and device == "cuda"

    logmel = build_logmel(cfg).to(device)
    logmel.set_stats(*get_logmel_stats(cfg, logmel, device))

    train_ds = AudioClips("train", cfg, train=True)
    val_ds = AudioClips("val", cfg, train=False)
    n_classes = len(train_ds.classes)
    weights = class_weights(train_ds.labels, n_classes) if tc["class_weighted_loss"] else None
    if args.limit:
        rng = np.random.default_rng(args.seed)
        train_ds = Subset(train_ds, rng.permutation(len(train_ds))[:args.limit])
        val_sub = rng.permutation(len(val_ds))[:args.limit]
    train_loader = make_loader(train_ds, tc["batch_size"], shuffle=True, num_workers=tc["num_workers"])
    val_loader = make_loader(val_ds, tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])

    model = build_model(args.model, n_classes, cfg).to(device)
    lr = cfg["models"][args.model].get("lr", tc["lr"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc["warmup_epochs"] * len(train_loader), steps))
    scaler = torch.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    ckpt_path = RESULTS / "checkpoints" / f"{run}_seed{args.seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history, best, bad = [], -1.0, 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for wav, y, _ in train_loader:
            wav, y = wav.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = logmel(wav)
            with torch.autocast(device_type=device, enabled=amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(y)
            n += len(y)

        probs = predict(model, logmel, val_loader, device)
        labels = val_ds.labels
        if args.limit:
            probs, labels = probs[val_sub], labels[val_sub]
        vm = classification_metrics(labels, probs, n_classes)
        history.append({"epoch": epoch, "train_loss": loss_sum / n, "val_acc": vm["accuracy"],
                        "val_macro_f1": vm["macro_f1"], "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
        flag = ""
        if vm["macro_f1"] > best:
            best, bad, flag = vm["macro_f1"], 0, " *"
            torch.save({"model": model.state_dict(), "classes": val_ds.classes, "epoch": epoch,
                        "logmel_mean": logmel.mean.item(), "logmel_std": logmel.std.item(),
                        "config": cfg, "seed": args.seed}, ckpt_path)
        else:
            bad += 1
        print(f"[{run} s{args.seed}] ep {epoch:3d} loss {loss_sum / n:.4f} val_acc {vm['accuracy']:.4f} "
              f"val_f1 {vm['macro_f1']:.4f} ({time.time() - t0:.0f}s){flag}", flush=True)
        if bad >= tc["patience"]:
            print(f"early stop at epoch {epoch}")
            break
    train_time = time.time() - t_start

    # reload best checkpoint, evaluate val + test, measure cost
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    results = {split: evaluate_split(model, logmel, run, args.seed, split, cfg, device) for split in ["val", "test"]}
    n_samples = int(round(cfg["audio"]["sample_rate"] * cfg["audio"]["duration_s"]))
    in_shape = tuple(logmel(torch.zeros(1, n_samples, device=device)).shape[1:])
    log = {
        "run": run, "model": args.model, "seed": args.seed, "best_epoch": ckpt["epoch"],
        "train_time_s": train_time, "params": count_params(model), "macs": count_macs(model, in_shape),
        "latency_ms": latency_ms(model, logmel, n_samples, device), "input_shape": in_shape,
        "val": results["val"], "test": results["test"], "history": history,
        "config": cfg, "env": env_info(), "limit": args.limit,
        "split_md5": {s: hashlib.md5((resolve(cfg["dataset"]["splits_dir"]) / f"{s}.csv").read_bytes()).hexdigest()
                      for s in ["train", "val", "test"]},
    }
    log_path = RESULTS / "logs" / f"{run}_seed{args.seed}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"[{run} s{args.seed}] best ep {ckpt['epoch']}  "
          f"val F1 {results['val']['macro_f1']:.4f}  test acc {results['test']['accuracy']:.4f} "
          f"test F1 {results['test']['macro_f1']:.4f}  params {log['params']:,}  MACs {log['macs'] / 1e6:.1f}M  "
          f"lat {log['latency_ms']:.2f} ms  time {train_time / 60:.1f} min")


if __name__ == "__main__":
    main()

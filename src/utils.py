"""Shared helpers: config loading, seeding, environment logging."""
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Env overrides (used by smoke tests / alternative experiments): ATI_CONFIG, ATI_RESULTS
DEFAULT_CONFIG = Path(os.environ.get("ATI_CONFIG", ROOT / "configs" / "default.yaml"))


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


RESULTS = resolve(os.environ.get("ATI_RESULTS", "results"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def load_classes() -> list[str]:
    return json.loads((ROOT / "configs" / "classes.json").read_text(encoding="utf-8"))


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["numpy", "pandas", "sklearn", "torch", "torchaudio", "torchvision", "soundfile", "soxr"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


if __name__ == "__main__":
    print(json.dumps(env_info(), indent=2))

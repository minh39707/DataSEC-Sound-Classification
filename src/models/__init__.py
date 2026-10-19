import torch.nn as nn

from src.models.crnn import CRNN
from src.models.mobilenet import MobileNetV3Small
from src.models.small_cnn import SmallCNN

MODELS = {"small_cnn": SmallCNN, "mobilenetv3s": MobileNetV3Small, "crnn": CRNN}


def build_model(name: str, n_classes: int, cfg: dict) -> nn.Module:
    params = {k: v for k, v in cfg["models"][name].items() if k != "lr"}
    return MODELS[name](n_classes=n_classes, **params)

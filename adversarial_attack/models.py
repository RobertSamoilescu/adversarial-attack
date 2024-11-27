import torch.nn as nn
from typing import Literal
from torchvision.models import resnet50, ResNet50_Weights


def load_model(
    model_name: Literal["resnet50"], device: Literal["cuda", "cpu"] = "cuda"
) -> nn.Module:
    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    model = model.to(device)
    return model.eval()

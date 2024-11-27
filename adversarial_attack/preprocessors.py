import torch
from typing import Literal, Callable, Tuple

from torchvision.transforms import transforms
from torchvision.models import ResNet50_Weights


def get_preprocessor(
    model_name: Literal["resnet50"],
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Returns a pair of functions that preprocess and inverse preprocess
    an image for the given model.

    :param model_name: The name of the model to load.
    :return: A pair of functions that preprocess and inverse preprocess
        an image for the given model
    """

    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
    else:
        raise ValueError(f"Model {model_name} not supported.")

    mean, std = preprocess.mean, preprocess.std
    if mean is not None and std is not None:
        transform = transforms.Normalize(mean=mean, std=std)
        inverse_transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0, 0, 0], std=[1 / s for s in std]),
                transforms.Normalize(mean=[-m for m in mean], std=[1, 1, 1]),
            ]
        )
    else:
        transform = transforms.Lambda(lambda x: x)
        inverse_transform = transforms.Lambda(lambda x: x)

    return transform, inverse_transform

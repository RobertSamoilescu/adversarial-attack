import torch
from typing import Literal, Callable

from torchvision.io import decode_image
from torchvision.transforms import transforms
from torchvision.models import ResNet50_Weights


def get_image_loader(
    model_name: Literal["resnet50"],
) -> Callable[[str], torch.Tensor]:
    """Returns a function that loads an image and preprocesses
    it for the given model. The returned image will be a tensor
    with shape (3, H, W) and values in the range [0, 1].

    :param model_name: The name of the model to load.
    :return: A function that loads an image and preprocesses it
        for the given model
    """

    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
    else:
        raise ValueError(f"Model {model_name} not supported.")

    mean, std = preprocess.mean, preprocess.std
    if mean is not None and std is not None:
        inverse_transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0, 0, 0], std=[1 / s for s in std]),
                transforms.Normalize(mean=[-m for m in mean], std=[1, 1, 1]),
            ]
        )
    else:
        inverse_transform = transforms.Lambda(lambda x: x)

    def image_loader(image_path: str) -> torch.Tensor:
        image = decode_image(image_path)
        image = preprocess(image)
        return inverse_transform(image)

    return image_loader

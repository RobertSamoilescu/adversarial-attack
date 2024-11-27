import pytest
import torch
from adversarial_attack.preprocessors import get_preprocessor
from adversarial_attack.image_loaders import get_image_loader


def test_image_loader():
    image_loader = get_image_loader("resnet50")
    image = image_loader("tests/assets/panda.jpg")

    transform, inverse_transform = get_preprocessor("resnet50")
    normalized_image = transform(image)
    denormalized_image = inverse_transform(normalized_image)
    assert torch.allclose(image, denormalized_image, atol=1e-7)


def test_error_image_loader():
    with pytest.raises(ValueError):
        get_preprocessor("resnet18")

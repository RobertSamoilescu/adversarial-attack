import pytest
from adversarial_attack.image_loaders import get_image_loader


def test_image_loader():
    image_loader = get_image_loader("resnet50")
    image = image_loader("tests/assets/panda.jpg")
    assert image.shape == (3, 224, 224)
    assert image.min() >= 0
    assert image.max() <= 1


def test_error_image_loader():
    with pytest.raises(ValueError):
        get_image_loader("resnet18")

import torch
import pytest
from adversarial_attack.attacks.pgd import ProjectedGradientDescent


def test_error_range():
    image = torch.rand(3, 224, 224)
    image[0, 0, 0] = -1

    with pytest.raises(ValueError):
        ProjectedGradientDescent._assert_ranges(image)


def test_error_shape():
    image = torch.rand(224, 224)

    with pytest.raises(ValueError):
        ProjectedGradientDescent._add_batch_dimension(image)


@pytest.mark.parametrize(
    "image",
    [
        torch.rand(3, 224, 224),
        torch.rand(1, 3, 224, 224),
    ],
)
def test_shape(image: torch.Tensor):
    image = ProjectedGradientDescent._add_batch_dimension(image)
    assert image.shape == (1, 3, 224, 224)

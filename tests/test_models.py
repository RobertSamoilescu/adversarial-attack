import pytest
from adversarial_attack.models import load_model


def test_load_model():
    model = load_model("resnet50", device="cpu")
    assert model.training is False


def test_error_load_model():
    with pytest.raises(ValueError):
        load_model("resnet18")

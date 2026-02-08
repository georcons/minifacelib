import pytest

from minifacelib.models.base.facepredictor import FacePredictor
from minifacelib.models.factory import make_predictor


def test_factory_cnn() -> None:
    """Factory should create a CNN predictor."""
    p: FacePredictor = make_predictor("cnn")
    assert isinstance(p, FacePredictor)


def test_factory_openai() -> None:
    """Factory should create an OpenAI predictor."""
    p: FacePredictor = make_predictor("openai")
    assert isinstance(p, FacePredictor)


def test_factory_unknown_raises() -> None:
    """Factory should raise for unknown predictor names."""
    # Factory may raise different exception types depending on implementation
    with pytest.raises(Exception):  # pylint: disable=broad-exception-caught
        make_predictor("nope")

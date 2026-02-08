"""Factory utilities for creating face predictor instances."""

from minifacelib.models.base import FacePredictor
from minifacelib.models.openai import OpenAIFacePredictor
from minifacelib.models.cnn import CNNFacePredictor

_PREDICTORS = {
    "openai": OpenAIFacePredictor,
    "cnn": CNNFacePredictor,
}


def make_predictor(name: str, *args, **kwargs) -> FacePredictor:
    """Instantiate a face predictor by name."""
    if name not in _PREDICTORS:
        raise ValueError("Invalid predictor.")

    class_ = _PREDICTORS[name]
    return class_(*args, **kwargs)


def predictors() -> list[str]:
    """Return the list of available predictor names."""
    return list(_PREDICTORS.keys())

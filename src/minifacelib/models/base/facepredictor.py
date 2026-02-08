"""Base interface for face prediction models."""

from abc import ABC, abstractmethod
from typing import ClassVar
from PIL import Image

from .types import FacePrediction


class FacePredictor(ABC):
    """Abstract face predictor."""

    MODEL_NAME: ClassVar[str] = "_base"

    @abstractmethod
    def predict(self, image: Image.Image) -> FacePrediction:
        """Predict from an input image."""
        pass

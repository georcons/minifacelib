"""
Tests for the CNNFacePredictor.
"""

from collections.abc import Sequence
from typing import Any

import pytest
from PIL import Image

from minifacelib.models.cnn.predictor import CNNFacePredictor


class _FakeModel:
    """Minimal fake model used to replace the predictor's internal model."""

    def eval(self) -> None:
        """Mimic torch model eval() method."""
        return None

    def __call__(
        self, input_tensor: Any  # pylint: disable=unused-argument
    ) -> tuple[Sequence[float], float, Sequence[float]]:
        """Return fixed outputs matching the predictor's expected model interface."""
        gender_logits: tuple[float, float] = (0.1, 0.9)
        age_value: float = 23.7
        emotion_logits: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        return gender_logits, age_value, emotion_logits


def test_cnn_predictor_predict(
    monkeypatch: pytest.MonkeyPatch, rgb_image: Image.Image
) -> None:
    """Predict on an RGB image and validate output fields are in expected domains."""
    predictor = CNNFacePredictor()
    monkeypatch.setattr(predictor, "_model", _FakeModel(), raising=False)

    pred = predictor.predict(rgb_image)

    assert pred.gender in ("male", "female", "unknown")
    assert pred.emotion in (
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
        "unknown",
    )
    assert isinstance(pred.age, int)
    assert pred.model == predictor.MODEL_NAME

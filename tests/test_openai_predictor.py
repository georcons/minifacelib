from typing import Any

import pytest
from PIL import Image

import minifacelib.models.openai.predictor as mod
from minifacelib.models.openai.predictor import OpenAIFacePredictor


class _FakeResponse:  # pylint: disable=too-few-public-methods
    """Minimal fake OpenAI response object."""

    def __init__(self, text: str) -> None:
        self.output_text = text


def test_openai_predictor_happy_path(
    monkeypatch: pytest.MonkeyPatch, rgb_image: Image.Image
) -> None:
    """Predictor should parse valid JSON responses."""
    def fake_create(*args: Any, **kwargs: Any) -> _FakeResponse:
        """Return a deterministic fake OpenAI response."""
        return _FakeResponse('{"gender":"male","emotion":"happy","age":30}')

    monkeypatch.setattr(mod.openai.responses, "create", fake_create)

    predictor = OpenAIFacePredictor(model="dummy")
    pred = predictor.predict(rgb_image)

    assert pred.gender == "male"
    assert pred.emotion == "happy"
    assert pred.age == 30


def test_openai_predictor_codeblock_json(
    monkeypatch: pytest.MonkeyPatch, rgb_image: Image.Image
) -> None:
    """Predictor should handle JSON wrapped in a Markdown code fence."""
    def fake_create(*args: Any, **kwargs: Any) -> _FakeResponse:
        """Return a deterministic fake OpenAI response."""
        return _FakeResponse('```json\n{"gender":"f","emotion":"neutral","age":"22"}\n```')

    monkeypatch.setattr(mod.openai.responses, "create", fake_create)

    predictor = OpenAIFacePredictor(model="dummy")
    pred = predictor.predict(rgb_image)

    assert pred.gender == "female"
    assert pred.emotion == "neutral"
    assert pred.age == 22


def test_openai_predictor_invalid_json_fallback(
    monkeypatch: pytest.MonkeyPatch, rgb_image: Image.Image
) -> None:
    """Predictor should raise when it cannot parse the response."""
    def fake_create(*args: Any, **kwargs: Any) -> _FakeResponse:
        """Return a deterministic fake OpenAI response."""
        return _FakeResponse("not-json")

    monkeypatch.setattr(mod.openai.responses, "create", fake_create)

    predictor = OpenAIFacePredictor(model="dummy")
    with pytest.raises(Exception):  # pylint: disable=broad-exception-caught
        predictor.predict(rgb_image)

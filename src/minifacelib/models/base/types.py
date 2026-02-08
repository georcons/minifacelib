"""Shared types and parsing helpers for face prediction outputs."""

from dataclasses import dataclass
from typing import Literal, cast

Gender = Literal["male", "female", "unknown"]

Emotion = Literal[
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
    "unknown",
]


@dataclass(frozen=True)
class FacePrediction:
    """Container for a single face prediction result."""
    gender: Gender
    emotion: Emotion
    age: int
    model: str


_EMOTION_MAPPING: dict[str, Emotion] = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}


def parse_gender(x: str) -> Gender:
    """Normalize a gender string to a known label."""
    x = x.lower().strip()
    if x in ("m", "male"):
        return "male"
    if x in ("f", "female"):
        return "female"
    return "unknown"


def parse_emotion(text: str) -> Emotion:
    """Normalize an emotion string to a known label."""
    text = text.lower().strip()
    return _EMOTION_MAPPING.get(text, "unknown")

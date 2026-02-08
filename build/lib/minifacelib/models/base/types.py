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
    x = x.lower().strip()
    if x in ("m", "male"):
        return "male"
    if x in ("f", "female"):
        return "female"
    return "unknown"


def parse_emotion(text: str) -> Emotion:
    text = text.lower().strip()
    return _EMOTION_MAPPING.get(text, "unknown")

import base64
import io
import json
from typing import Any, ClassVar, cast

import openai
from PIL import Image

from minifacelib.models.base import (
    FacePredictor,
    parse_gender,
    parse_emotion,
    FacePrediction,
)
from .prompts import get_facepredictor_system_prompt


class OpenAIFacePredictor(FacePredictor):
    MODEL_NAME: ClassVar[str] = "openai"

    def __init__(self, model: str = "gpt-5.2"):
        self._model = model

    def _image_to_b64str(self, img: Image.Image) -> str:
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def predict(self, image: Image.Image) -> FacePrediction:
        data_url = "data:image/jpeg;base64," + self._image_to_b64str(image)

        input_payload = [
            {"role": "system", "content": get_facepredictor_system_prompt()},
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": data_url}],
            },
        ]

        response = openai.responses.create(
            model=self._model, input=cast(Any, input_payload)
        )

        raw_json = response.output_text
        raw_json = raw_json.strip().removeprefix("```json").removesuffix("```").strip()

        data = json.loads(raw_json)

        try:
            age = int(data.get("age", -1))
        except Exception:
            age = -1

        return FacePrediction(
            gender=parse_gender(data.get("gender", "unknown")),
            emotion=parse_emotion(data.get("emotion", "unknown")),
            age=age,
            model=self.MODEL_NAME,
        )

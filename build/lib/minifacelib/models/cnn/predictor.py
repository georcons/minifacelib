from typing import ClassVar
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as T

from minifacelib.models.base import (
    FacePredictor,
    FacePrediction,
    parse_emotion,
    parse_gender,
)

from minifacelib.models.cnn.emotion_model import EmotionModel
from minifacelib.models.cnn.gender_age_model import GenderAgeModel


class CNNFacePredictor(FacePredictor):
    MODEL_NAME: ClassVar[str] = "cnn"
    _IMAGE_SIZE = 64

    def __init__(self, device: str | None = None):
        self._device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._transform = T.Compose(
            [
                T.Resize((self._IMAGE_SIZE, self._IMAGE_SIZE)),
                T.ToTensor(),
            ]
        )

        base_dir = Path(__file__).resolve().parent

        emotion_pt_path = base_dir / "weights" / "emotion.pt"
        gender_age_pt_path = base_dir / "weights" / "gender_age.pt"

        if not emotion_pt_path.exists():
            raise FileNotFoundError(
                f"Missing emotion weights: {emotion_pt_path}. Run:\n\n"
                "minifacelib install\n"
            )
        if not gender_age_pt_path.exists():
            raise FileNotFoundError(
                f"Missing gender+age weights: {gender_age_pt_path}. Run:\n\n"
                "minifacelib install\n"
            )

        # Load emotions model

        emotion_pt = torch.load(emotion_pt_path, map_location="cpu")

        num_emotions = int(emotion_pt.get("num_emotions", 7))
        self._emotion_labels: dict[int, str] = {
            int(k): str(v) for k, v in (emotion_pt.get("labels", {}) or {}).items()
        }

        self._emotion_model = EmotionModel(num_emotions=num_emotions)
        self._emotion_model.load_state_dict(emotion_pt["state_dict"])
        self._emotion_model.to(self._device)
        self._emotion_model.eval()

        # Load gender + age model

        gender_age_pt = torch.load(gender_age_pt_path, map_location="cpu")

        num_genders = int(gender_age_pt.get("num_genders", 2))
        self._gender_labels: dict[int, str] = {
            int(k): str(v)
            for k, v in (gender_age_pt.get("gender_labels", {}) or {}).items()
        }
        if not self._gender_labels:
            self._gender_labels = {0: "male", 1: "female"}

        self._gender_age_model = GenderAgeModel(num_genders=num_genders)
        self._gender_age_model.load_state_dict(gender_age_pt["state_dict"])
        self._gender_age_model.to(self._device)
        self._gender_age_model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> FacePrediction:
        x = self._transform(image.convert("RGB")).unsqueeze(0).to(self._device)

        # Predict emotion
        emotion_logits = self._emotion_model(x)
        emotion_idx = int(torch.argmax(emotion_logits, dim=1).item())
        emotion_raw = self._emotion_labels.get(emotion_idx, "unknown")

        # Gender + Age
        gender_logits, age_pred = self._gender_age_model(x)
        gender_idx = int(torch.argmax(gender_logits, dim=1).item())
        gender_raw = self._gender_labels.get(gender_idx, "unknown")

        age = int(round(age_pred.item()))
        if age < 0:
            age = -1

        return FacePrediction(
            gender=parse_gender(gender_raw),
            emotion=parse_emotion(emotion_raw),
            age=age,
            model=self.MODEL_NAME,
        )

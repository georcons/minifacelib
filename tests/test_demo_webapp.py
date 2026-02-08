"""
Webapp endpoint tests for the demo FastAPI app.
"""

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import demo.webapp as webapp_mod
from minifacelib.models.base.types import FacePrediction


class _FakePredictor:
    """Predictor stub that returns a deterministic FacePrediction."""

    MODEL_NAME = "fake"

    def predict(self, img: Image.Image) -> FacePrediction:  # pylint: disable=unused-argument
        """Return a fixed prediction regardless of input."""
        return FacePrediction(
            gender="male",
            emotion="happy",
            age=21,
            model=self.MODEL_NAME,
        )


def test_root_returns_html(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GET / should serve the index.html content."""
    (tmp_path / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr(webapp_mod, "BASE_DIR", tmp_path, raising=False)

    app = webapp_mod.DemoFaceApp(predictor=_FakePredictor()).app
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "ok" in response.text


def test_predict_image() -> None:
    """POST /predict should return JSON including gender."""
    app = webapp_mod.DemoFaceApp(predictor=_FakePredictor()).app
    client = TestClient(app)

    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="JPEG")

    files: dict[str, tuple[str, bytes, str]] = {
        "file": ("x.jpg", buf.getvalue(), "image/jpeg")
    }
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data: Any = response.json()
    assert data["gender"] == "male"

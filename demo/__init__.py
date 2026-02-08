from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import demo.webapp as webapp_mod


class _FakePredictor:
    MODEL_NAME = "fake"

    def predict(self, img):  # noqa: ANN001
        from minifacelib.models.base.types import FacePrediction

        return FacePrediction(gender="male", emotion="happy", age=21, model="fake")


def _make_fake(*args, **kwargs):  # noqa: ANN001
    return _FakePredictor()


def test_root_returns_html(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr(webapp_mod, "BASE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(webapp_mod, "make_predictor", _make_fake, raising=False)

    demo = webapp_mod.DemoFaceApp()
    client = TestClient(demo.app)

    r = client.get("/")
    assert r.status_code == 200
    assert "ok" in r.text


def test_predict_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr(webapp_mod, "BASE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(webapp_mod, "make_predictor", _make_fake, raising=False)

    demo = webapp_mod.DemoFaceApp()
    client = TestClient(demo.app)

    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="JPEG")
    files = {"file": ("x.jpg", buf.getvalue(), "image/jpeg")}

    r = client.post("/predict", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["gender"] == "male"

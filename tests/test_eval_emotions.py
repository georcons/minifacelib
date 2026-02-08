"""
Tests for eval/eval_emotions.py.

These tests run the evaluation script via runpy with a fake predictor and a
temporary dataset layout. Paths are resolved relative to the repository root
so the tests are independent of pytest's current working directory.
"""

import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

# Resolve the repository root as: <repo>/tests/test_eval_emotions.py -> parents[1] == <repo>
ROOT_DIR = Path(__file__).resolve().parents[1]


class _FakePred:  # pylint: disable=too-few-public-methods
    """Minimal prediction object with only the fields the script consumes."""

    def __init__(self, emotion: str) -> None:
        self.emotion = emotion


class _FakePredictor:  # pylint: disable=too-few-public-methods
    """Minimal predictor returning deterministic emotion predictions."""

    def __init__(self, emotion: str) -> None:
        self._emotion = emotion

    def predict(self, img: Any) -> _FakePred:  # pylint: disable=unused-argument
        """Return a fake prediction regardless of input image."""
        return _FakePred(self._emotion)


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path, format="JPEG")


def test_eval_emotions_runs_and_writes_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Script should run and write metrics + confusion matrix."""
    os.environ["MPLBACKEND"] = "Agg"

    data_dir = tmp_path / "data"
    split_dir = data_dir / "processed" / "fer2013" / "splits"
    out_dir = tmp_path / "reports"

    img1 = split_dir / "img1.jpg"
    img2 = split_dir / "img2.jpg"
    _write_jpeg(img1)
    _write_jpeg(img2)

    df = pd.DataFrame({"path": ["img1.jpg", "img2.jpg"], "emotion": ["happy", "sad"]})
    split_csv = split_dir / "test.csv"
    split_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_csv, index=False)

    import minifacelib  # pylint: disable=import-outside-toplevel

    def _make_predictor(_: str) -> _FakePredictor:
        return _FakePredictor("happy")

    monkeypatch.setattr(minifacelib, "make_predictor", _make_predictor)

    script_path = ROOT_DIR / "eval" / "eval_emotions.py"
    assert script_path.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--model",
            "cnn",
            "--data-dir",
            str(data_dir),
            "--split",
            "test",
            "--out-dir",
            str(out_dir),
        ],
    )

    runpy.run_path(str(script_path), run_name="__main__")

    metrics_path = out_dir / "fer2013" / "cnn" / "test" / "metrics.json"
    cm_path = out_dir / "fer2013" / "cnn" / "test" / "confusion_matrix.png"

    assert metrics_path.exists()
    assert cm_path.exists()
    assert cm_path.stat().st_size > 0

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "labels" in metrics


def test_eval_emotions_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--limit should restrict evaluation to at least one example."""
    os.environ["MPLBACKEND"] = "Agg"

    data_dir = tmp_path / "data"
    split_dir = data_dir / "processed" / "fer2013" / "splits"
    out_dir = tmp_path / "reports"

    for i in range(3):
        _write_jpeg(split_dir / f"img{i}.jpg")

    df = pd.DataFrame(
        {"path": ["img0.jpg", "img1.jpg", "img2.jpg"], "emotion": ["happy", "sad", "neutral"]}
    )
    split_csv = split_dir / "test.csv"
    split_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_csv, index=False)

    import minifacelib  # pylint: disable=import-outside-toplevel

    def _make_predictor(_: str) -> _FakePredictor:
        return _FakePredictor("sad")

    monkeypatch.setattr(minifacelib, "make_predictor", _make_predictor)

    script_path = ROOT_DIR / "eval" / "eval_emotions.py"
    assert script_path.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--model",
            "cnn",
            "--data-dir",
            str(data_dir),
            "--split",
            "test",
            "--out-dir",
            str(out_dir),
            "--limit",
            "1",
        ],
    )

    runpy.run_path(str(script_path), run_name="__main__")

    metrics_path = out_dir / "fer2013" / "cnn" / "test" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert isinstance(metrics["labels"], list)
    assert len(metrics["labels"]) >= 1


def test_eval_emotions_missing_columns_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Script should raise if the split CSV is missing required columns."""
    os.environ["MPLBACKEND"] = "Agg"

    data_dir = tmp_path / "data"
    split_dir = data_dir / "processed" / "fer2013" / "splits"
    out_dir = tmp_path / "reports"
    split_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"path": ["x.jpg"]}).to_csv(split_dir / "test.csv", index=False)

    script_path = ROOT_DIR / "eval" / "eval_emotions.py"
    assert script_path.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--model",
            "cnn",
            "--data-dir",
            str(data_dir),
            "--split",
            "test",
            "--out-dir",
            str(out_dir),
        ],
    )

    with pytest.raises(ValueError):
        runpy.run_path(str(script_path), run_name="__main__")

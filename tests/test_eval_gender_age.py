"""
Tests for eval/eval_gender_age.py.

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

ROOT_DIR = Path(__file__).resolve().parents[1]


class _FakePred:  # pylint: disable=too-few-public-methods
    """Minimal prediction object with only the fields the script consumes."""

    def __init__(self, gender: str, age: int) -> None:
        self.gender = gender
        self.age = age


class _FakePredictor:  # pylint: disable=too-few-public-methods
    """Minimal predictor returning deterministic gender/age predictions."""

    def __init__(self, gender: str, age: int) -> None:
        self._gender = gender
        self._age = age

    def predict(self, img: Any) -> _FakePred:  # pylint: disable=unused-argument
        return _FakePred(self._gender, self._age)


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path, format="JPEG")


def _find_metrics_json(out_dir: Path) -> Path:
    candidates = list(out_dir.rglob("metrics.json"))
    assert candidates, "metrics.json was not created anywhere under out_dir"
    return sorted(candidates)[0]

def test_eval_gender_age_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    os.environ["MPLBACKEND"] = "Agg"

    data_dir = tmp_path / "data"
    split_dir = data_dir / "processed" / "utkface" / "splits"
    out_dir = tmp_path / "reports"

    for i in range(3):
        _write_jpeg(split_dir / f"img{i}.jpg")

    df = pd.DataFrame(
        {
            "path": ["img0.jpg", "img1.jpg", "img2.jpg"],
            "gender": ["male", "female", "male"],
            "age": [10, 20, 30],
        }
    )
    split_csv = split_dir / "test.csv"
    split_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_csv, index=False)

    import minifacelib  # pylint: disable=import-outside-toplevel

    def _make_predictor(_: str) -> _FakePredictor:
        return _FakePredictor("female", 25)

    monkeypatch.setattr(minifacelib, "make_predictor", _make_predictor)

    script_path = ROOT_DIR / "eval" / "eval_gender_age.py"
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

    metrics_path = _find_metrics_json(out_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert isinstance(metrics, dict)


def test_eval_gender_age_missing_columns_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.environ["MPLBACKEND"] = "Agg"

    data_dir = tmp_path / "data"
    split_dir = data_dir / "processed" / "utkface" / "splits"
    out_dir = tmp_path / "reports"
    split_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"path": ["x.jpg"]}).to_csv(split_dir / "test.csv", index=False)

    script_path = ROOT_DIR / "eval" / "eval_gender_age.py"
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

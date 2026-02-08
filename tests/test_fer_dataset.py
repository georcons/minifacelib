import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from training.fer_dataset import FerDataset, load_fer_csv


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path, format="JPEG")


def test_load_fer_csv_filters_rows_and_lowercases(tmp_path: Path) -> None:
    csv_path = tmp_path / "split.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "emotion"])
        w.writeheader()
        w.writerow({"path": "x.jpg", "emotion": " HAPPY "})  # valid -> happy
        w.writerow({"path": "", "emotion": "sad"})  # invalid (missing path)
        w.writerow({"path": "y.jpg", "emotion": ""})  # invalid (missing emotion)

    rows = load_fer_csv(csv_path)
    assert rows == [("x.jpg", "happy")]


def test_load_fer_csv_raises_when_empty(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "emotion"])
        w.writeheader()
        w.writerow({"path": "", "emotion": ""})

    with pytest.raises(RuntimeError):
        load_fer_csv(csv_path)


def test_fer_dataset_len_and_getitem(tmp_path: Path) -> None:
    img_path = tmp_path / "a.jpg"
    _write_jpeg(img_path)

    rows = [(str(img_path), "happy")]

    def transform(_img: Image.Image) -> torch.Tensor:
        return torch.zeros((3, 64, 64), dtype=torch.float32)

    ds = FerDataset(rows, transform)
    assert len(ds) == 1

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 64, 64)
    assert isinstance(y, int)
from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from training.utkface_dataset import UtkFaceDataset, load_utkface_csv


def _write_jpeg(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(p, format="JPEG")


def test_load_utkface_csv_filters_and_parses(tmp_path: Path) -> None:
    """load_utkface_csv should filter invalid rows and normalize values."""
    img_ok = tmp_path / "ok.jpg"
    _write_jpeg(img_ok)

    csv_path = tmp_path / "split.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "age", "gender"])
        writer.writeheader()
        writer.writerow({"path": str(img_ok), "age": "10", "gender": "male"})  # valid
        writer.writerow({"path": "", "age": "10", "gender": "male"})  # missing path
        writer.writerow({"path": str(img_ok), "age": "", "gender": "male"})  # missing age
        writer.writerow({"path": str(img_ok), "age": "x", "gender": "male"})  # bad int
        writer.writerow({"path": str(img_ok), "age": "12", "gender": ""})  # missing gender
        writer.writerow({"path": str(img_ok), "age": "12", "gender": "other"})  # invalid gender
        writer.writerow({"path": str(img_ok), "age": "11", "gender": "FEMALE"})  # valid after lower()

    rows = load_utkface_csv(csv_path)
    assert rows == [(str(img_ok), 10, "male"), (str(img_ok), 11, "female")]


def test_load_utkface_csv_raises_when_empty(tmp_path: Path) -> None:
    """load_utkface_csv should raise when no valid rows exist."""
    csv_path = tmp_path / "empty.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "age", "gender"])
        writer.writeheader()
        writer.writerow({"path": "", "age": "", "gender": ""})

    with pytest.raises(RuntimeError):
        load_utkface_csv(csv_path)


def test_utkface_dataset_len_and_getitem(tmp_path: Path) -> None:
    """UtkFaceDataset should return tensors and numeric targets."""
    img = tmp_path / "a.jpg"
    _write_jpeg(img)

    rows = [(str(img), 33, "male")]

    def transform(_image: Image.Image) -> torch.Tensor:
        return torch.zeros((3, 64, 64), dtype=torch.float32)

    ds = UtkFaceDataset(rows, transform)
    assert len(ds) == 1

    x, y_gender, y_age = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 64, 64)
    assert y_gender == 0
    assert y_age == 33.0

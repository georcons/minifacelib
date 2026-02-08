import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image


_GENDER_TO_IDX: dict[str, int] = {
    "male": 0,
    "female": 1,
}


@dataclass(frozen=True)
class UtkFaceSample:
    image: Image.Image
    age: int
    gender: str


def load_utkface_csv(csv_path: Path) -> list[tuple[str, int, str]]:  # path age gender
    rows: list[tuple[str, int, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_path = (r.get("path") or "").strip()
            age_s = (r.get("age") or "").strip()
            gender = (r.get("gender") or "").strip().lower()

            if not img_path or not age_s or not gender:
                continue
            if gender not in _GENDER_TO_IDX:
                continue

            try:
                age = int(age_s)
            except ValueError:
                continue

            rows.append((img_path, age, gender))

    if not rows:
        raise RuntimeError(f"No rows loaded from {csv_path}")

    return rows


class UtkFaceDataset(torch.utils.data.Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, age, gender = self.rows[idx]

        img = Image.open(path).convert("RGB")
        x = self.transform(img)

        y_gender = _GENDER_TO_IDX[gender]
        y_age = float(age)

        return x, y_gender, y_age

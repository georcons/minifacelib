import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

_EMOTION_TO_IDX: dict[str, int] = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}


@dataclass(frozen=True)
class FerSample:
    image: Image.Image
    emotion: str


def load_fer_csv(csv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_path = r.get("path")
            emo = (r.get("emotion") or "").strip().lower()
            if img_path and emo:
                rows.append((img_path, emo))
    if not rows:
        raise RuntimeError(f"No rows loaded from {csv_path}")
    return rows


class FerDataset(torch.utils.data.Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = _EMOTION_TO_IDX[label]
        return x, y

import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

FER_EMOTIONS = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}

EMOTION_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "scared": "fear",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
}

UTK_RE = re.compile(
    r"^(?P<age>\d+)_(?P<gender>[01])_.*\.(jpg|jpeg|png)$", re.IGNORECASE
)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class UtkRow:
    path: str
    age: int
    gender: str


@dataclass(frozen=True)
class FerRow:
    path: str
    emotion: str


def _run(cmd: Sequence[str]) -> str:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    p = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    out = (p.stdout or "").strip()
    if p.returncode != 0:
        safe = out.encode("utf-8", "replace").decode("utf-8", "replace")
        raise RuntimeError(safe or f"Command failed: {' '.join(cmd)}")
    return out


def _ensure_kaggle() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError("kaggle CLI not found (pip install kaggle)")

    kaggle_home = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_home.exists():
        raise RuntimeError("Kaggle credentials not found at ~/.kaggle/kaggle.json")


def _download(dataset_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_id,
            "-p",
            str(out_dir),
            "--unzip",
        ]
    )


def _images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _split(
    n: int, train: float, dev: float, test: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    if n <= 0:
        return [], [], []
    total = train + dev + test
    train, dev, test = train / total, dev / total, test / total

    idx = list(range(n))
    random.Random(seed).shuffle(idx)

    n_train = int(n * train)
    n_dev = int(n * dev)
    return idx[:n_train], idx[n_train : n_train + n_dev], idx[n_train + n_dev :]


def _write_csv(
    path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def _parse_utk(root: Path) -> list[UtkRow]:
    rows: list[UtkRow] = []
    for img in _images(root):
        m = UTK_RE.match(img.name)
        if not m:
            continue
        age = int(m.group("age"))
        gender = "male" if m.group("gender") == "0" else "female"
        rows.append(UtkRow(str(img.resolve()), age, gender))
    if not rows:
        raise RuntimeError(f"No UTKFace images parsed under: {root}")
    return rows


def _normalize_emotion(s: str) -> str | None:
    return EMOTION_ALIASES.get(s.strip().lower())


def _parse_fer(root: Path) -> list[FerRow]:
    rows: list[FerRow] = []
    for img in _images(root):
        label = _normalize_emotion(img.parent.name)
        if label in FER_EMOTIONS:
            rows.append(FerRow(str(img.resolve()), label))
    if not rows:
        raise RuntimeError(f"No FER images parsed under: {root}")
    return rows


def main() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    default_data_dir = src_dir / "data"

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(default_data_dir))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--dev", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--utk-id", default="jangedoo/utkface-new")
    ap.add_argument("--fer-id", default="msambare/fer2013")
    args = ap.parse_args()

    data = Path(args.data_dir)
    raw = data / "raw"
    processed = data / "processed"

    _ensure_kaggle()

    if not args.skip_download:
        _download(args.utk_id, raw / "utkface")
        _download(args.fer_id, raw / "fer2013")

    utk_rows = _parse_utk(raw / "utkface")
    fer_rows = _parse_fer(raw / "fer2013")

    utk_tr, utk_dev, utk_te = _split(
        len(utk_rows), args.train, args.dev, args.test, args.seed
    )
    fer_tr, fer_dev, fer_te = _split(
        len(fer_rows), args.train, args.dev, args.test, args.seed
    )

    _write_csv(
        processed / "utkface" / "splits" / "train.csv",
        ["path", "age", "gender"],
        ((utk_rows[i].path, utk_rows[i].age, utk_rows[i].gender) for i in utk_tr),
    )
    _write_csv(
        processed / "utkface" / "splits" / "dev.csv",
        ["path", "age", "gender"],
        ((utk_rows[i].path, utk_rows[i].age, utk_rows[i].gender) for i in utk_dev),
    )
    _write_csv(
        processed / "utkface" / "splits" / "test.csv",
        ["path", "age", "gender"],
        ((utk_rows[i].path, utk_rows[i].age, utk_rows[i].gender) for i in utk_te),
    )

    _write_csv(
        processed / "fer2013" / "splits" / "train.csv",
        ["path", "emotion"],
        ((fer_rows[i].path, fer_rows[i].emotion) for i in fer_tr),
    )
    _write_csv(
        processed / "fer2013" / "splits" / "dev.csv",
        ["path", "emotion"],
        ((fer_rows[i].path, fer_rows[i].emotion) for i in fer_dev),
    )
    _write_csv(
        processed / "fer2013" / "splits" / "test.csv",
        ["path", "emotion"],
        ((fer_rows[i].path, fer_rows[i].emotion) for i in fer_te),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = str(e).encode("utf-8", "replace").decode("utf-8", "replace")
        print(msg, file=sys.stderr)
        sys.exit(1)

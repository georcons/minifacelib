import csv
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from PIL import Image


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path, format="JPEG")


@pytest.fixture()
def train_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass

    import training.fer_dataset as fer_ds
    sys.modules["fer_dataset"] = fer_ds

    import importlib
    import training.train_fer as mod
    importlib.reload(mod)
    return mod


def test_accuracy_and_run_epoch_empty(train_module: ModuleType) -> None:
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([1, 0], dtype=torch.int64)
    assert train_module.accuracy(logits, y) == 1.0

    model = torch.nn.Linear(4, 7)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    empty_loader = []
    loss, acc = train_module.run_epoch(
        model=model,
        loader=empty_loader,  # type: ignore[arg-type]
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train=True,
    )
    assert (loss, acc) == (0.0, 0.0)


def test_main_smoke_one_epoch_writes_outputs(train_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ["MPLBACKEND"] = "Agg"

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    img_tr = tmp_path / "train.jpg"
    img_dv = tmp_path / "dev.jpg"
    _write_jpeg(img_tr)
    _write_jpeg(img_dv)

    def write_split(p: Path, rows: list[dict[str, str]]) -> None:
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "emotion"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_split(splits_dir / "train.csv", [{"path": str(img_tr), "emotion": "happy"}])
    write_split(splits_dir / "dev.csv", [{"path": str(img_dv), "emotion": "sad"}])

    out_path = tmp_path / "emotion.pt"
    reports_dir = tmp_path / "reports"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_fer.py",
            "--splits-dir",
            str(splits_dir),
            "--out",
            str(out_path),
            "--reports-dir",
            str(reports_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--lr",
            "0.001",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--seed",
            "1",
        ],
    )

    train_module.main()

    assert out_path.exists()
    assert (reports_dir / "history.csv").exists()
    assert (reports_dir / "loss_curve.png").exists()
    assert (reports_dir / "acc_curve.png").exists()
    assert (reports_dir / "loss_curve.png").stat().st_size > 0
    assert (reports_dir / "acc_curve.png").stat().st_size > 0


def test___main___guard_covered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Run as script to cover: if __name__ == "__main__": main()
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass

    import training.fer_dataset as fer_ds
    sys.modules["fer_dataset"] = fer_ds

    # tiny splits
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    img_tr = tmp_path / "train.jpg"
    img_dv = tmp_path / "dev.jpg"
    _write_jpeg(img_tr)
    _write_jpeg(img_dv)

    with (splits_dir / "train.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "emotion"])
        w.writeheader()
        w.writerow({"path": str(img_tr), "emotion": "happy"})
    with (splits_dir / "dev.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "emotion"])
        w.writeheader()
        w.writerow({"path": str(img_dv), "emotion": "sad"})

    out_path = tmp_path / "emotion.pt"
    reports_dir = tmp_path / "reports"

    script_path = Path("training") / "train_fer.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--splits-dir",
            str(splits_dir),
            "--out",
            str(out_path),
            "--reports-dir",
            str(reports_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--lr",
            "0.001",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--seed",
            "1",
        ],
    )

    runpy.run_path(str(script_path), run_name="__main__")
    assert out_path.exists()

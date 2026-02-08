import csv
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from PIL import Image


def _write_jpeg(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(p, format="JPEG")


@pytest.fixture()
def train_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:  # pylint: disable=import-outside-toplevel
    """Import and reload the training module with dataset aliasing for tests."""
    import importlib

    import training.train_utkface as mod
    import training.utkface_dataset as real_ds

    sys.modules["utkface_dataset"] = real_ds
    importlib.reload(mod)
    return mod


def test_gender_accuracy_and_age_mae(train_module: ModuleType) -> None:
    """gender_accuracy and age_mae should compute expected scalar metrics."""
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([1, 0], dtype=torch.int64)
    assert train_module.gender_accuracy(logits, y) == 1.0

    pred_age = torch.tensor([10.0, 30.0], dtype=torch.float32)
    y_age = torch.tensor([20.0, 10.0], dtype=torch.float32)
    assert train_module.age_mae(pred_age, y_age) == 15.0


def test_run_epoch_empty_loader_returns_zeros(train_module: ModuleType) -> None:
    """run_epoch should return zeros for an empty loader."""
    model = torch.nn.Linear(4, 2)
    gender_criterion = torch.nn.CrossEntropyLoss()
    age_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    empty_loader: list[object] = []

    loss, gacc, mae = train_module.run_epoch(
        model=model,
        loader=empty_loader,
        gender_criterion=gender_criterion,
        age_criterion=age_criterion,
        optimizer=optimizer,
        device=device,
        train=True,
        age_loss_weight=0.1,
    )
    assert (loss, gacc, mae) == (0.0, 0.0, 0.0)


def test_main_smoke_one_epoch_writes_outputs(
    train_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() should run one epoch and write model and report artifacts."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    img_tr = tmp_path / "train.jpg"
    img_dv = tmp_path / "dev.jpg"
    _write_jpeg(img_tr)
    _write_jpeg(img_dv)

    def write_split(csv_path: Path, rows: list[dict[str, str]]) -> None:
        """Write a split CSV file with expected columns."""
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "age", "gender"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    write_split(
        splits_dir / "train.csv",
        [{"path": str(img_tr), "age": "20", "gender": "male"}],
    )
    write_split(
        splits_dir / "dev.csv",
        [{"path": str(img_dv), "age": "30", "gender": "female"}],
    )

    out_path = tmp_path / "gender_age.pt"
    reports_dir = tmp_path / "reports"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_utkface.py",
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
            "--age-loss-weight",
            "0.1",
        ],
    )

    train_module.main()

    assert out_path.exists()
    assert (reports_dir / "history.csv").exists()
    assert (reports_dir / "loss_curve.png").exists()
    assert (reports_dir / "gender_acc_curve.png").exists()
    assert (reports_dir / "age_mae_curve.png").exists()

    assert (reports_dir / "loss_curve.png").stat().st_size > 0
    assert (reports_dir / "gender_acc_curve.png").stat().st_size > 0
    assert (reports_dir / "age_mae_curve.png").stat().st_size > 0

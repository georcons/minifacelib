import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

from minifacelib.models.cnn.gender_age_model import GenderAgeModel
from training.utkface_dataset import UtkFaceDataset, load_utkface_csv


IDX_TO_GENDER: dict[int, str] = {
    0: "male",
    1: "female",
}


def gender_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


def age_mae(pred_age: torch.Tensor, y_age: torch.Tensor) -> float:
    return (pred_age - y_age).abs().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    gender_criterion: nn.Module,
    age_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    age_loss_weight: float,
) -> tuple[float, float, float]:
    model.train() if train else model.eval()

    total_loss = 0.0
    total_gender_acc = 0.0
    total_age_mae = 0.0
    total_n = 0

    for x, y_gender, y_age in loader:
        x = x.to(device)
        y_gender = y_gender.to(device)
        y_age = y_age.to(device).float()

        with torch.set_grad_enabled(train):
            gender_logits, age_pred = model(x)

            loss_gender = gender_criterion(gender_logits, y_gender)
            loss_age = age_criterion(age_pred, y_age)
            loss = loss_gender + age_loss_weight * loss_age

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_gender_acc += (
            gender_accuracy(gender_logits.detach(), y_gender.detach()) * bsz
        )
        total_age_mae += age_mae(age_pred.detach(), y_age.detach()) * bsz
        total_n += bsz

    if total_n == 0:
        return 0.0, 0.0, 0.0

    return total_loss / total_n, total_gender_acc / total_n, total_age_mae / total_n


def save_history_csv(history_path: Path, history: list[dict[str, float]]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "dev_loss",
                "train_gender_acc",
                "dev_gender_acc",
                "train_age_mae",
                "dev_age_mae",
            ],
        )
        w.writeheader()
        for row in history:
            w.writerow(row)


def save_loss_plot(plot_path: Path, history: list[dict[str, float]]) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(r["epoch"]) for r in history]
    train_loss = [float(r["train_loss"]) for r in history]
    dev_loss = [float(r["dev_loss"]) for r in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, dev_loss, label="dev loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def save_gender_acc_plot(plot_path: Path, history: list[dict[str, float]]) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(r["epoch"]) for r in history]
    train_acc = [float(r["train_gender_acc"]) for r in history]
    dev_acc = [float(r["dev_gender_acc"]) for r in history]

    plt.figure()
    plt.plot(epochs, train_acc, label="train gender acc")
    plt.plot(epochs, dev_acc, label="dev gender acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def save_age_mae_plot(plot_path: Path, history: list[dict[str, float]]) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(r["epoch"]) for r in history]
    train_mae = [float(r["train_age_mae"]) for r in history]
    dev_mae = [float(r["dev_age_mae"]) for r in history]

    plt.figure()
    plt.plot(epochs, train_mae, label="train age MAE")
    plt.plot(epochs, dev_mae, label="dev age MAE")
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def main() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    default_splits = src_dir / "data" / "processed" / "utkface" / "splits"
    default_out = (
        src_dir / "src" / "minifacelib" / "models" / "cnn" / "weights" / "gender_age.pt"
    )
    default_reports_dir = src_dir / "reports" / "utkface"

    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=str(default_splits))
    parser.add_argument("--out", default=str(default_out))
    parser.add_argument("--reports-dir", default=str(default_reports_dir))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--age-loss-weight", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    splits_dir = Path(args.splits_dir)
    train_rows = load_utkface_csv(splits_dir / "train.csv")
    dev_rows = load_utkface_csv(splits_dir / "dev.csv")

    transform = T.Compose(
        [
            T.Resize((64, 64)),
            T.ToTensor(),
        ]
    )

    train_ds = UtkFaceDataset(train_rows, transform)
    dev_ds = UtkFaceDataset(dev_rows, transform)

    pin = str(args.device).startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    device = torch.device(args.device)

    model = GenderAgeModel(num_genders=2).to(device)
    gender_criterion = nn.CrossEntropyLoss()
    age_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dev_loss = float("inf")
    best_epoch_num = -1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reports_dir = Path(args.reports_dir)
    history_path = reports_dir / "history.csv"
    loss_plot_path = reports_dir / "loss_curve.png"
    gender_acc_plot_path = reports_dir / "gender_acc_curve.png"
    age_mae_plot_path = reports_dir / "age_mae_curve.png"

    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_gacc, tr_mae = run_epoch(
            model,
            train_loader,
            gender_criterion,
            age_criterion,
            optimizer,
            device,
            train=True,
            age_loss_weight=args.age_loss_weight,
        )
        dv_loss, dv_gacc, dv_mae = run_epoch(
            model,
            dev_loader,
            gender_criterion,
            age_criterion,
            optimizer,
            device,
            train=False,
            age_loss_weight=args.age_loss_weight,
        )

        row = {
            "epoch": int(epoch),
            "train_loss": float(tr_loss),
            "dev_loss": float(dv_loss),
            "train_gender_acc": float(tr_gacc),
            "dev_gender_acc": float(dv_gacc),
            "train_age_mae": float(tr_mae),
            "dev_age_mae": float(dv_mae),
        }
        history.append(row)

        print(
            f"epoch={epoch} "
            f"train_loss={tr_loss:.4f} train_gender_acc={tr_gacc:.4f} train_age_mae={tr_mae:.2f} "
            f"dev_loss={dv_loss:.4f} dev_gender_acc={dv_gacc:.4f} dev_age_mae={dv_mae:.2f}"
        )

        if dv_loss < best_dev_loss:
            best_epoch_num = epoch
            best_dev_loss = dv_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": "GenderAgeModel",
                    "num_genders": 2,
                    "gender_labels": IDX_TO_GENDER,
                    "age_regression": True,
                    "image_size": 64,
                    "age_loss_weight": float(args.age_loss_weight),
                },
                out_path,
            )

        save_history_csv(history_path, history)
        save_loss_plot(loss_plot_path, history)
        save_gender_acc_plot(gender_acc_plot_path, history)
        save_age_mae_plot(age_mae_plot_path, history)

    print(f"best_dev_loss={best_dev_loss:.4f} (epoch {best_epoch_num})")
    print(f"saved={out_path}")
    print(f"history={history_path}")
    print(f"loss_plot={loss_plot_path}")
    print(f"gender_acc_plot={gender_acc_plot_path}")
    print(f"age_mae_plot={age_mae_plot_path}")


if __name__ == "__main__":
    main()

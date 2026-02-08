import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from minifacelib.models.cnn.emotion_model import EmotionModel
from training.fer_dataset import FerDataset, load_fer_csv


EMOTION_TO_IDX: dict[str, int] = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}

IDX_TO_EMOTION: dict[int, str] = {v: k for k, v in EMOTION_TO_IDX.items()}


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    model.train() if train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits.detach(), y.detach()) * bsz
        total_n += bsz

    if total_n == 0:
        return 0.0, 0.0

    return total_loss / total_n, total_acc / total_n


def save_history_csv(history_path: Path, history: list[dict[str, float]]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "dev_loss", "train_acc", "dev_acc"]
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


def save_acc_plot(plot_path: Path, history: list[dict[str, float]]) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(r["epoch"]) for r in history]
    train_loss = [float(r["train_acc"]) for r in history]
    dev_loss = [float(r["dev_acc"]) for r in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="train acc")
    plt.plot(epochs, dev_loss, label="dev acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def main() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    default_splits = src_dir / "data" / "processed" / "fer2013" / "splits"
    default_out = (
        src_dir / "src" / "minifacelib" / "models" / "cnn" / "weights" / "emotion.pt"
    )
    default_reports_dir = src_dir / "reports" / "fer2013"

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
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    splits_dir = Path(args.splits_dir)
    train_rows = load_fer_csv(splits_dir / "train.csv")
    dev_rows = load_fer_csv(splits_dir / "dev.csv")

    transform = T.Compose(
        [
            T.Resize((64, 64)),
            T.ToTensor(),
        ]
    )

    train_ds = FerDataset(train_rows, transform)
    dev_ds = FerDataset(dev_rows, transform)

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

    model = EmotionModel(num_emotions=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dev_loss: float = float("inf")
    best_epoch_num: int = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reports_dir = Path(args.reports_dir)
    history_path = reports_dir / "history.csv"
    loss_plot_path = reports_dir / "loss_curve.png"
    acc_plot_path = reports_dir / "acc_curve.png"

    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        dv_loss, dv_acc = run_epoch(
            model, dev_loader, criterion, optimizer, device, train=False
        )

        row = {
            "epoch": int(epoch),
            "train_loss": float(tr_loss),
            "dev_loss": float(dv_loss),
            "train_acc": float(tr_acc),
            "dev_acc": float(dv_acc),
        }
        history.append(row)

        print(
            f"epoch={epoch} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"dev_loss={dv_loss:.4f} dev_acc={dv_acc:.4f}"
        )

        if dv_loss < best_dev_loss:
            best_epoch_num = epoch
            best_dev_loss = dv_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": "EmotionModel",
                    "num_emotions": 7,
                    "labels": IDX_TO_EMOTION,
                    "image_size": 64,
                    # "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                },
                out_path,
            )

        save_history_csv(history_path, history)
        save_loss_plot(loss_plot_path, history)
        save_acc_plot(acc_plot_path, history)

    print(f"best_dev_loss={best_dev_loss:.4f} (epoch {best_epoch_num})")
    print(f"saved={out_path}")
    print(f"history={history_path}")
    print(f"loss_plot={loss_plot_path}")
    print(f"acc_plot={acc_plot_path}")


if __name__ == "__main__":
    main()

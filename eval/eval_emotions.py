"""Evaluate performance on FER2013"""

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from minifacelib import make_predictor
from minifacelib.models.base.types import parse_emotion


def main() -> None:
    """Main entrypoint"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "openai"], default="cnn")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data"),
    )
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "reports" / "eval"),
    )
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    split_csv = (
        Path(args.data_dir) / "processed" / "fer2013" / "splits" / f"{args.split}.csv"
    )

    df = pd.read_csv(split_csv)
    if not {"path", "emotion"}.issubset(df.columns):
        raise ValueError(f"Expected columns ['path','emotion'] in {split_csv}")

    base_dir = split_csv.parent
    image_paths = []
    for p in df["path"].astype(str):
        p = Path(p)
        image_paths.append(p if p.is_absolute() else (base_dir / p).resolve())

    y_true = [parse_emotion(e) for e in df["emotion"].astype(str)]

    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]
        y_true = y_true[: args.limit]

    predictor = make_predictor(args.model)

    y_pred = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        pred = predictor.predict(img)
        y_pred.append(getattr(pred, "emotion", "unknown"))

    labels = sorted(set(y_true) | set(y_pred))

    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out_dir = Path(args.out_dir) / "fer2013" / args.model / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "labels": labels,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot the confusion matrix
    fig = plt.figure()
    im = plt.imshow(cm)
    plt.title("Confusion Matrix (Emotion)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(im, label="Count")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"Saved results to: {out_dir}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1: {f1:.4f}")


if __name__ == "__main__":
    main()

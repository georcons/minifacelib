import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from minifacelib import make_predictor
from minifacelib.models.base.types import parse_gender


def main() -> None:
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
        Path(args.data_dir) / "processed" / "utkface" / "splits" / f"{args.split}.csv"
    )

    df = pd.read_csv(split_csv)
    if not {"path", "age", "gender"}.issubset(df.columns):
        raise ValueError(f"Expected columns ['path','age','gender'] in {split_csv}")

    base_dir = split_csv.parent

    image_paths = []
    for p in df["path"].astype(str):
        p = Path(p)
        image_paths.append(p if p.is_absolute() else (base_dir / p).resolve())

    y_gender_true = [parse_gender(g) for g in df["gender"].astype(str)]
    y_age_true = df["age"].astype(float).to_numpy()

    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]
        y_gender_true = y_gender_true[: args.limit]
        y_age_true = y_age_true[: args.limit]

    predictor = make_predictor(args.model)

    y_gender_pred = []
    y_age_pred: list[float] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        pred = predictor.predict(img)
        y_gender_pred.append(getattr(pred, "gender", "unknown"))
        y_age_pred.append(getattr(pred, "age", np.nan))

    y_age_pred = np.array(y_age_pred, dtype=float)

    # Gender metrics
    gender_labels = sorted(set(y_gender_true) | set(y_gender_pred))
    gender_acc = float(accuracy_score(y_gender_true, y_gender_pred))
    gender_f1 = float(
        f1_score(y_gender_true, y_gender_pred, average="micro", zero_division=0)
    )
    gender_cm = confusion_matrix(y_gender_true, y_gender_pred, labels=gender_labels)

    # Age metric (MAE)
    valid = ~np.isnan(y_age_pred)
    age_mae = (
        float(np.mean(np.abs(y_age_true[valid] - y_age_pred[valid])))
        if np.any(valid)
        else None
    )

    out_dir = Path(args.out_dir) / "utkface" / args.model / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "gender": {
            "accuracy": gender_acc,
            "f1": gender_f1,
            "labels": gender_labels,
        },
        "age": {
            "mae": age_mae,
            "n_valid": int(np.sum(valid)),
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Gender confusion matrix with legend (colorbar)
    fig = plt.figure()
    im = plt.imshow(gender_cm)
    plt.title("Confusion Matrix (Gender)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(gender_labels)), gender_labels, rotation=0)
    plt.yticks(range(len(gender_labels)), gender_labels)
    plt.colorbar(im, label="Count")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_gender.png", dpi=200)
    plt.close(fig)

    print(f"Saved results to: {out_dir}")
    print(f"Gender accuracy: {gender_acc:.4f}")
    print(f"Gender F1: {gender_f1:.4f}")
    print(f"Age MAE: {age_mae if age_mae is not None else 'N/A'}")


if __name__ == "__main__":
    main()

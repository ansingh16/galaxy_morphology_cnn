"""Train one model and log the run to MLflow.

    python -m galaxy_cnn.train --model small   --epochs 15
    python -m galaxy_cnn.train --model resnet  --epochs 8

Both runs land in the same MLflow experiment so the two approaches sit side by
side. We select the checkpoint by best *validation macro-F1* rather than plain
accuracy: with a class as small as 17 images, accuracy is dominated by the big
classes and would happily ignore the rare ones. The chosen checkpoint is saved
to models/<model>.pt with everything serving needs to rebuild it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .data import CLASS_NAMES, make_loaders
from .model import build_model, count_params

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) over a loader."""
    model.eval()
    trues, preds = [], []
    for xb, yb in loader:
        logits = model(xb.to(DEVICE))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def train(
    model_kind: str = "small",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 0,
    num_workers: int = 2,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader, info = make_loaders(
        model=model_kind, batch_size=batch_size, seed=seed, num_workers=num_workers
    )
    model = build_model(model_kind).to(DEVICE)
    total, trainable = count_params(model)

    criterion = nn.CrossEntropyLoss(weight=info["class_weights"].to(DEVICE))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_path = MODELS_DIR / f"{model_kind}.pt"

    mlflow.set_experiment("galaxy10-morphology")
    with mlflow.start_run(run_name=model_kind):
        mlflow.log_params(
            {
                "model": model_kind,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "seed": seed,
                "params_total": total,
                "params_trainable": trainable,
                "n_train": info["n_train"],
            }
        )

        best_f1 = -1.0
        best_metrics: dict = {}
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(yb)
            train_loss = running / info["n_train"]

            val_true, val_pred = predict(model, val_loader)
            val = scores(val_true, val_pred)
            mlflow.log_metrics(
                {"train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}},
                step=epoch,
            )
            print(
                f"[{model_kind}] epoch {epoch:2d}/{epochs}  "
                f"loss={train_loss:.3f}  val_acc={val['accuracy']:.3f}  "
                f"val_f1={val['macro_f1']:.3f}  ({time.time() - t0:.0f}s)"
            )

            if val["macro_f1"] > best_f1:
                best_f1 = val["macro_f1"]
                best_metrics = {f"val_{k}": v for k, v in val.items()}
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_kind": model_kind,
                        "class_names": CLASS_NAMES,
                        "val_macro_f1": best_f1,
                    },
                    ckpt_path,
                )

        # final test with the best checkpoint
        best = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best["state_dict"])
        test_true, test_pred = predict(model, test_loader)
        test = scores(test_true, test_pred)
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        mlflow.log_artifact(str(ckpt_path), artifact_path="model")

        print(
            f"[{model_kind}] best val_f1={best_f1:.3f} | "
            f"test acc={test['accuracy']:.3f} f1={test['macro_f1']:.3f}"
        )
        return {"best_val": best_metrics, "test": test, "checkpoint": str(ckpt_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="train a galaxy morphology model")
    ap.add_argument("--model", choices=["small", "resnet", "resnet_ft"], default="small")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    train(
        model_kind=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

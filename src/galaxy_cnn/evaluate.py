"""Evaluate a trained checkpoint on the test split.

    python -m galaxy_cnn.evaluate --model small

Writes a confusion matrix and a per-class precision/recall/F1 table to reports/,
and prints the headline accuracy / macro-F1. This is the honest look at where a
model does well (the big, distinctive classes) and where the imbalance bites
(the rare edge-on and cigar classes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from .data import CLASS_NAMES, make_loaders
from .model import build_model
from .train import predict, scores

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def load_checkpoint(model_kind: str) -> torch.nn.Module:
    ckpt = torch.load(MODELS_DIR / f"{model_kind}.pt", map_location="cpu", weights_only=False)
    model = build_model(model_kind)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def evaluate(model_kind: str = "small", seed: int = 0) -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)
    _, _, test_loader, _ = make_loaders(model=model_kind, seed=seed, num_workers=2)
    model = load_checkpoint(model_kind)

    y_true, y_pred = predict(model, test_loader)
    headline = scores(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    (REPORTS_DIR / f"{model_kind}_report.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(
        ax=ax, cmap="viridis", colorbar=False, values_format=".2f"
    )
    ax.set_title(f"{model_kind}: row-normalised confusion (test)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"{model_kind}_confusion.png", dpi=110)
    plt.close(fig)

    print(f"[{model_kind}] test accuracy={headline['accuracy']:.3f}  "
          f"macro_f1={headline['macro_f1']:.3f}")
    print(f"  wrote reports/{model_kind}_confusion.png and _report.json")
    return {"scores": headline, "report": report}


def main() -> None:
    ap = argparse.ArgumentParser(description="evaluate a trained checkpoint")
    ap.add_argument("--model", choices=["small", "resnet", "resnet_ft"], default="small")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    evaluate(args.model, args.seed)


if __name__ == "__main__":
    main()

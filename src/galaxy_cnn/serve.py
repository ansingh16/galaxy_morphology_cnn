"""FastAPI service: upload a galaxy image, get its morphology class back.

    uvicorn galaxy_cnn.serve:app --port 8000
    curl -s localhost:8000/predict -F file=@some_galaxy.png

The checkpoint to serve is picked by the MODEL_KIND env var ('small' or
'resnet', default 'small') and loaded from MODEL_DIR (default ./models). The
image goes through the same resize + normalise the model saw in validation, so
there's no train/serve skew.
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from .data import (
    CLASS_NAMES,
    GALAXY_MEAN,
    GALAXY_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .model import build_model

MODEL_KIND = os.environ.get("MODEL_KIND", "resnet_ft")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))

STATE: dict = {"model": None, "transform": None, "info": {}}


def _inference_transform(model_kind: str) -> transforms.Compose:
    if model_kind.startswith("resnet"):
        size, mean, std = 96, IMAGENET_MEAN, IMAGENET_STD
    else:
        size, mean, std = 69, GALAXY_MEAN, GALAXY_STD
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_model() -> None:
    ckpt_path = MODEL_DIR / f"{MODEL_KIND}.pt"
    if not ckpt_path.exists():
        # leave STATE empty; /health will report not-loaded rather than crash
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(MODEL_KIND)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    STATE["model"] = model
    STATE["transform"] = _inference_transform(MODEL_KIND)
    STATE["info"] = {
        "model_kind": MODEL_KIND,
        "num_classes": len(CLASS_NAMES),
        "val_macro_f1": ckpt.get("val_macro_f1"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="galaxy-morphology-cnn", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": STATE["model"] is not None}


@app.get("/model-info")
def model_info() -> dict:
    if STATE["model"] is None:
        raise HTTPException(503, "no model loaded")
    return STATE["info"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:  # noqa: B008  (FastAPI idiom)
    if STATE["model"] is None:
        raise HTTPException(503, "no model loaded")
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(422, "could not read an image from the upload") from None

    x = STATE["transform"](img).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(STATE["model"](x), dim=1).squeeze(0)
    top = int(probs.argmax())
    return {
        "predicted_class": top,
        "predicted_label": CLASS_NAMES[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }

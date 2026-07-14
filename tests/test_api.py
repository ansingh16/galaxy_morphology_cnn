"""API smoke tests against an untrained in-memory model.

The real checkpoint is gitignored and absent in CI, so we drop a fresh SmallCNN
straight into the app state. That still exercises the upload path, the image
decoding, the transform and the softmax response shape.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from galaxy_cnn import serve
from galaxy_cnn.data import CLASS_NAMES
from galaxy_cnn.model import build_model


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(serve, "load_model", lambda: None)  # don't hit disk on startup
    serve.STATE["model"] = build_model("small").eval()
    serve.STATE["transform"] = serve._inference_transform("small")
    serve.STATE["info"] = {"model_kind": "small", "num_classes": len(CLASS_NAMES)}
    with TestClient(serve.app) as c:
        yield c
    serve.STATE.update({"model": None, "transform": None, "info": {}})


def _png_bytes(size=80):
    img = Image.new("RGB", (size, size), (20, 40, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_a_valid_class(client):
    r = client.post("/predict", files={"file": ("g.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in range(len(CLASS_NAMES))
    assert body["predicted_label"] in CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_arbitrary_input_size_is_accepted(client):
    # an odd-sized upload still works: the transform resizes it
    r = client.post("/predict", files={"file": ("g.png", _png_bytes(150), "image/png")})
    assert r.status_code == 200


def test_non_image_upload_is_rejected(client):
    bad = {"file": ("x.txt", io.BytesIO(b"not an image"), "text/plain")}
    r = client.post("/predict", files=bad)
    assert r.status_code == 422

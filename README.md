# galaxy-morphology-cnn

![ci](https://github.com/ansingh16/galaxy_morphology_cnn/actions/workflows/ci.yml/badge.svg)

Classify galaxy morphology from telescope images with **PyTorch**. I train a from-scratch
convolutional network and compare it against **ResNet-18 transfer learning** (both
frozen-feature extraction and full fine-tuning) on real SDSS galaxy images, with MLflow
experiment tracking and a FastAPI image-prediction service. It runs end to end on a laptop
CPU.

The data is [Galaxy10 SDSS](https://astronn.readthedocs.io/en/latest/galaxy10sdss.html):
21,785 real SDSS galaxy cutouts (69×69 RGB) hand-labelled into 10 morphology classes (round
smooth galaxies, edge-on disks, barred and unbarred spirals, and so on). It is heavily
imbalanced: the largest class has around 7,000 images and the smallest just **17**. That
imbalance drives most of the modelling decisions here.

## The task

Morphology is how galaxies are first sorted (smooth ellipticals, edge-on disks, face-on
spirals), and at 69 pixels the finer distinctions are hard even for a person. That makes it a
good image-classification problem: a strong signal on the big distinctive classes, a long
tail of rare and ambiguous ones, and a real class-imbalance challenge that punishes plain
accuracy.

Two decisions follow from the imbalance and run through the whole project:

- the loss is **class-weighted** by inverse frequency, and
- models are selected and reported on **macro-F1** (which weights every class equally) rather
  than raw accuracy, because a model that ignores the rare classes can still look good on
  accuracy.

## Results

Three approaches, all trained on the same stratified split and judged on the held-out test
set:

| approach                        | trainable params | test accuracy | test macro-F1 |
|---------------------------------|------------------|---------------|---------------|
| from-scratch CNN                | ~0.31 M          | 0.653         | 0.534         |
| ResNet-18, **frozen** features  | ~5 K             | 0.473         | 0.384         |
| ResNet-18, **fine-tuned**       | ~11 M            | **0.797**     | **0.641**     |

The comparison matters more than any single number:

- **Frozen ImageNet features lose,** even to the small from-scratch net. ImageNet's filters
  are tuned for everyday photos, and at 96px they are not a good description of a smudgy
  galaxy. Reusing them without adapting them does not pay off.
- **Fine-tuning wins clearly.** Starting from the ImageNet weights but letting the whole
  network keep training at a low learning rate lifts test macro-F1 from 0.53 (scratch) and
  0.38 (frozen) to **0.64**, and accuracy to **~0.80**, competitive with published Galaxy10
  baselines. A good initialisation plus adaptation beats both training from scratch and using
  frozen features off the shelf.

The fine-tuned ResNet is the champion and the checkpoint the API serves. The one class no
method handles is "Disk, Edge-on, Boxy Bulge": with 17 images in the entire dataset there is
not enough signal for anything to learn it, and the per-class report says so rather than
hiding it in an average. The full analysis, with training curves and confusion matrices, is
in [`notebooks/`](notebooks/).

## Quickstart

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e . -r requirements.txt

python -m galaxy_cnn.download_data                       # ~200 MB Galaxy10.h5 -> data/
python -m galaxy_cnn.train --model small       --epochs 18
python -m galaxy_cnn.train --model resnet       --epochs 12          # frozen features
python -m galaxy_cnn.train --model resnet_ft    --epochs 10 --lr 3e-4  # fine-tuned
python -m galaxy_cnn.evaluate --model resnet_ft                      # confusion matrix + report
uvicorn galaxy_cnn.serve:app --port 8000
```

The common commands are wrapped as `make` targets (`make data`, `make train-resnet-ft`,
`make eval`, `make serve`, `make test`).

## The models

`SmallCNN` is a compact conv net for the native 69px frames: three `conv-bn-relu` blocks
(3→32→64→128 channels) into an adaptive-average-pool head, around 0.31 M parameters.
Batch-norm and dropout keep it stable on a small, noisy dataset.

The transfer models are a pre-trained ResNet-18 with its classifier head swapped for our 10
classes. `resnet` freezes the backbone and trains only the head (feature extraction);
`resnet_ft` leaves the whole network trainable and fine-tunes it at a low learning rate. Both
feed images in at 96px with **ImageNet** normalisation, which is the pixel distribution the
pre-trained filters expect.

Because a galaxy sits at an arbitrary angle on the sky, orientation carries no label
information, so training augments with random flips and rotations. That is free extra data
that helps the rare classes most. Everything is defined in
[`src/galaxy_cnn/model.py`](src/galaxy_cnn/model.py) and
[`data.py`](src/galaxy_cnn/data.py).

## Experiment tracking

Training logs params, per-epoch metrics and the best checkpoint to MLflow (SQLite-backed, so
there is nothing to stand up). Every model lands in one experiment so the three runs sit side
by side:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

The notebooks read these runs straight back out of the store, so the curves and tables in
them come from the real trained models, not toy re-runs.

## Serving

[`galaxy_cnn.serve`](src/galaxy_cnn/serve.py) is a FastAPI app with `/health`,
`/model-info`, and `/predict`. `/predict` takes an uploaded image, resizes and normalises it
exactly as in validation (so there is no train/serve skew), and returns the predicted class
with per-class probabilities. The served checkpoint is chosen by the `MODEL_KIND` env var
(default `resnet_ft`) and loaded from `MODEL_DIR`.

```
curl -s localhost:8000/predict -F file=@galaxy.png
```

```
{"predicted_class": 1, "predicted_label": "Smooth, Completely round",
 "confidence": 0.978, "probabilities": { ... }}
```

The Docker image is a multi-stage build: dependencies go into a virtualenv in the builder
stage, then a slim runtime copies that venv plus the model and runs as a non-root user with
`MODEL_DIR` baked in, so the container is self-contained.

```
docker compose up --build
```

## Tests and CI

```
make test     # pytest
make lint     # ruff
```

The suite covers the data layer (stratified split, class weights, transforms), the models
(output shapes, the transfer-learning freeze), and the serving API (upload, decode, response
shape). It runs against tiny synthetic arrays and an in-memory model, so it is fast and needs
neither the 200 MB download nor a trained checkpoint. GitHub Actions runs ruff and pytest on
every push.

## Layout

```
src/galaxy_cnn/   download_data, data, model, train, evaluate, serve
notebooks/        01_eda, 02_baseline_cnn, 03_transfer_learning
tests/            data / model / api tests (synthetic, no download needed)
reports/          confusion matrices + per-class reports (written by evaluate)
Dockerfile, docker-compose.yml, Makefile
```

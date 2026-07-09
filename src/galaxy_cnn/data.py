"""Loading, splitting and augmenting the Galaxy10 images.

The raw file is a single HDF5 with two arrays: ``images`` (N, 69, 69, 3) uint8
and ``ans`` (N,) the class label 0-9. Everything downstream works off a
stratified train/val/test split so the rare classes stay represented in each
fold.

Two transform pipelines live here because the two models want different inputs:
the from-scratch CNN trains on the native 69x69 frames, while the ResNet needs
its input upsized and normalised the way its ImageNet pre-training expects. Both
train pipelines flip and rotate — a galaxy has no preferred orientation on the
sky, so that augmentation is physically free labels.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Galaxy10.h5"

# astroNN's Galaxy10 SDSS labelling, in index order 0-9
CLASS_NAMES = [
    "Disk, Face-on, No Spiral",
    "Smooth, Completely round",
    "Smooth, in-between round",
    "Smooth, Cigar shaped",
    "Disk, Edge-on, Rounded Bulge",
    "Disk, Edge-on, Boxy Bulge",
    "Disk, Edge-on, No Bulge",
    "Disk, Face-on, Tight Spiral",
    "Disk, Face-on, Medium Spiral",
    "Disk, Face-on, Loose Spiral",
]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet stats, for the pre-trained ResNet path
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# rough Galaxy10 channel stats (mostly dark sky), for the from-scratch net
GALAXY_MEAN = (0.10, 0.10, 0.10)
GALAXY_STD = (0.16, 0.15, 0.15)


def load_raw(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Read the whole dataset into memory (it's only ~200 MB)."""
    with h5py.File(path, "r") as f:
        images = np.asarray(f["images"], dtype=np.uint8)
        labels = np.asarray(f["ans"], dtype=np.int64)
    return images, labels


def stratified_split(
    labels: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index arrays for train/val/test, stratified so every class appears in each."""
    idx = np.arange(len(labels))
    train_idx, hold_idx = train_test_split(
        idx, test_size=val_frac + test_frac, stratify=labels, random_state=seed
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        hold_idx, test_size=rel_test, stratify=labels[hold_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights, so the 17-image class isn't drowned out."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    weights = len(labels) / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def build_transform(model: str, train: bool) -> transforms.Compose:
    """Transform pipeline for a given model ('small' or 'resnet') and split."""
    if model.startswith("resnet"):
        size, mean, std = 96, IMAGENET_MEAN, IMAGENET_STD
    else:
        size, mean, std = 69, GALAXY_MEAN, GALAXY_STD

    steps: list = [transforms.ToPILImage()]
    if size != 69:
        steps.append(transforms.Resize((size, size)))
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


class GalaxyDataset(Dataset):
    """Wraps the in-memory image/label arrays with a transform."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.transform(self.images[i])
        y = int(self.labels[i])
        return x, y


def make_loaders(
    model: str = "small",
    batch_size: int = 128,
    seed: int = 0,
    num_workers: int = 2,
    data_path: Path = DATA_PATH,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build train/val/test loaders plus a small info dict (weights, split sizes)."""
    images, labels = load_raw(data_path)
    tr, va, te = stratified_split(labels, seed=seed)

    train_ds = GalaxyDataset(images[tr], labels[tr], build_transform(model, train=True))
    val_ds = GalaxyDataset(images[va], labels[va], build_transform(model, train=False))
    test_ds = GalaxyDataset(images[te], labels[te], build_transform(model, train=False))

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    info = {
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "class_weights": class_weights(labels[tr]),
    }
    return train_loader, val_loader, test_loader, info

"""Data-layer tests on small synthetic arrays.

None of these touch the real 200 MB HDF5 — they build tiny fake image/label
arrays so the split, weighting and transform logic can be checked fast and in
CI without the download.
"""

import numpy as np

from galaxy_cnn.data import (
    NUM_CLASSES,
    GalaxyDataset,
    build_transform,
    class_weights,
    stratified_split,
)


def _fake(n_per_class=20):
    labels = np.repeat(np.arange(NUM_CLASSES), n_per_class)
    images = np.random.default_rng(0).integers(
        0, 256, size=(len(labels), 69, 69, 3), dtype=np.uint8
    )
    return images, labels


def test_split_is_a_partition():
    _, labels = _fake()
    tr, va, te = stratified_split(labels, seed=0)
    joined = np.concatenate([tr, va, te])
    assert len(joined) == len(labels)
    assert len(set(joined.tolist())) == len(labels)  # no overlap, nothing dropped


def test_every_class_appears_in_every_split():
    _, labels = _fake()
    tr, va, te = stratified_split(labels, seed=0)
    for idx in (tr, va, te):
        assert set(labels[idx].tolist()) == set(range(NUM_CLASSES))


def test_class_weights_up_weight_rare_classes():
    labels = np.array([0] * 100 + [1] * 5)  # class 1 is rare
    w = class_weights(labels)
    assert w.shape[0] == NUM_CLASSES
    assert w[1] > w[0]


def test_small_transform_keeps_native_resolution():
    images, labels = _fake(2)
    ds = GalaxyDataset(images, labels, build_transform("small", train=False))
    x, y = ds[0]
    assert tuple(x.shape) == (3, 69, 69)
    assert isinstance(y, int)


def test_resnet_transform_resizes_to_96():
    images, labels = _fake(2)
    ds = GalaxyDataset(images, labels, build_transform("resnet", train=True))
    x, _ = ds[0]
    assert tuple(x.shape) == (3, 96, 96)

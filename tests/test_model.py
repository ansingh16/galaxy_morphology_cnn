"""Model tests: shapes and the transfer-learning freeze."""

import torch

from galaxy_cnn.data import NUM_CLASSES
from galaxy_cnn.model import build_model, build_resnet, count_params


def test_small_cnn_forward_shape():
    model = build_model("small")
    out = model(torch.randn(4, 3, 69, 69))
    assert tuple(out.shape) == (4, NUM_CLASSES)


def test_resnet_forward_shape():
    model = build_model("resnet")
    out = model(torch.randn(2, 3, 96, 96))
    assert tuple(out.shape) == (2, NUM_CLASSES)


def test_resnet_backbone_is_frozen():
    # only the new classifier head should carry gradients
    model = build_resnet(freeze=True)
    total, trainable = count_params(model)
    assert trainable < total
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            assert not p.requires_grad


def test_resnet_ft_trains_the_whole_backbone():
    # the fine-tune variant leaves everything trainable
    model = build_model("resnet_ft")
    total, trainable = count_params(model)
    assert trainable == total


def test_unknown_model_kind_raises():
    try:
        build_model("transformer")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown model kind")

import inspect

import keras
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise

from . import config as C


def grayscale_camera(theta, rng):
    """CMPE-style Poisson corruption followed by Gaussian blur."""
    kwargs = {"mode": C.NOISE_MODE}
    signature = inspect.signature(random_noise)
    if "rng" in signature.parameters:
        kwargs["rng"] = rng
    elif "seed" in signature.parameters:
        kwargs["seed"] = rng

    noisy = C.NOISE_GAIN * random_noise(C.NOISE_SCALE * theta, **kwargs)
    return gaussian_filter(noisy, sigma=C.PSF_WIDTH)


def load_fashion_mnist_splits(
    train_samples=C.TRAIN_SAMPLES,
    val_samples=C.VAL_SAMPLES,
    test_samples=C.TEST_SAMPLES,
    seed=C.SEED,
):
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    rng = np.random.default_rng(seed)

    train_idx, val_idx = _stratified_train_val_indices(train_labels, train_samples, val_samples, rng)
    test_idx = _stratified_indices(test_labels, test_samples, rng)

    return {
        "train": _make_dataset(train_images[train_idx], train_labels[train_idx], np.random.default_rng(seed + 11)),
        "validation": _make_dataset(train_images[val_idx], train_labels[val_idx], np.random.default_rng(seed + 23)),
        "test": _make_dataset(test_images[test_idx], test_labels[test_idx], np.random.default_rng(seed + 37)),
    }


def _make_dataset(raw_images, labels, rng):
    clean = _normalize_clean(raw_images)
    observed = np.stack([grayscale_camera(image, rng) for image in raw_images]).astype("float32")
    observed = observed[..., None]

    return {
        "image": clean,
        "observed": observed,
        "label": labels.astype("int64"),
    }


def _normalize_clean(images):
    images = images.astype("float32")
    images = -1.0 + 2.0 * images / 255.0
    return images[..., None]


def _stratified_train_val_indices(labels, train_n, val_n, rng):
    train_parts = []
    val_parts = []
    train_counts = _class_counts(train_n)
    val_counts = _class_counts(val_n)

    for cls in range(10):
        indices = np.flatnonzero(labels == cls)
        rng.shuffle(indices)
        n_train = train_counts[cls]
        n_val = val_counts[cls]
        train_parts.append(indices[:n_train])
        val_parts.append(indices[n_train : n_train + n_val])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _stratified_indices(labels, n, rng):
    parts = []
    counts = _class_counts(n)
    for cls in range(10):
        indices = np.flatnonzero(labels == cls)
        rng.shuffle(indices)
        parts.append(indices[: counts[cls]])

    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def _class_counts(n):
    base = n // 10
    rem = n % 10
    return np.array([base + int(cls < rem) for cls in range(10)], dtype=int)


def select_one_per_class(dataset, max_classes=10):
    labels = dataset["label"]
    indices = []
    for cls in range(min(max_classes, 10)):
        matches = np.flatnonzero(labels == cls)
        if matches.size:
            indices.append(int(matches[0]))
    return np.array(indices, dtype=int)

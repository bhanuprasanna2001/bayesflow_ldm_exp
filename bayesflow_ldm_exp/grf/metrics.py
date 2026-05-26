import json
import os
import time

import bayesflow as bf
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from . import config as C


def sync_and_time(sync_fn):
    sync_fn()
    return time.perf_counter()


def as_field_array(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:  # (B, H*W)
        return x.reshape(x.shape[0], *C.FIELD_SHAPE)
    if x.ndim == 3:  # (B, S, H*W)
        return x.reshape(x.shape[0], x.shape[1], *C.FIELD_SHAPE)
    return x


def flatten_fields(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 5:
        return x.reshape(x.shape[0], x.shape[1], -1)
    if x.ndim == 4:
        return x.reshape(x.shape[0], -1)
    return x


def evaluate_posterior(samples, targets, sample_seconds):
    """Pixel-level diagnostics. Note: for GRFs the conditional posterior over fields
    is broad — pixel RMSE is a *soft* diagnostic, not a headline metric."""
    samples_flat = flatten_fields(samples)
    targets_flat = flatten_fields(targets)
    mean = samples_flat.mean(axis=1)
    std = samples_flat.std(axis=1)

    rmse = float(np.sqrt(np.mean((mean - targets_flat) ** 2)))
    mae = float(np.mean(np.abs(mean - targets_flat)))

    lower = np.quantile(samples_flat, 0.05, axis=1)
    upper = np.quantile(samples_flat, 0.95, axis=1)
    coverage_90 = float(np.mean((targets_flat >= lower) & (targets_flat <= upper)))
    interval_width_90 = float(np.mean(upper - lower))
    posterior_std = float(np.mean(std))

    n_total = samples_flat.shape[0] * samples_flat.shape[1]
    return {
        "posterior_mean_rmse": rmse,
        "posterior_mean_mae": mae,
        "coverage_90": coverage_90,
        "interval_width_90": interval_width_90,
        "posterior_std": posterior_std,
        "sample_seconds": float(sample_seconds),
        "ms_per_sample": float(1000.0 * sample_seconds / max(n_total, 1)),
    }


def chunked_mmd(x, y, splits=4):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    values = []
    for idx in np.array_split(np.arange(n), splits):
        if idx.size < 2:
            continue
        value = bf.metrics.functional.maximum_mean_discrepancy(x[idx], y[idx])
        values.append(float(keras.ops.convert_to_numpy(value)))
    return np.asarray(values, dtype=np.float32)


def radial_power_spectrum(fields):
    """Azimuthally-averaged power spectrum per field. Returns (B, n_bins)."""
    arr = np.asarray(fields, dtype=np.float32)
    if arr.ndim == 4:  # (B, H, W, 1)
        arr = arr[..., 0]
    H, W = arr.shape[-2], arr.shape[-1]
    fft = np.fft.fftshift(np.fft.fft2(arr, axes=(-2, -1)), axes=(-2, -1))
    power = np.abs(fft) ** 2
    ky, kx = np.indices((H, W))
    kr = np.sqrt((kx - W // 2) ** 2 + (ky - H // 2) ** 2).astype(int)
    n_bins = min(H, W) // 2
    bins = np.arange(n_bins + 1)
    mask = kr.ravel() < n_bins
    flat_kr = kr.ravel()[mask]
    flat_power = power.reshape(arr.shape[0], -1)[:, mask]
    out = np.zeros((arr.shape[0], n_bins), dtype=np.float32)
    counts = np.bincount(flat_kr, minlength=n_bins)[:n_bins]
    counts = np.maximum(counts, 1)
    for b in range(arr.shape[0]):
        sums = np.bincount(flat_kr, weights=flat_power[b], minlength=n_bins)[:n_bins]
        out[b] = sums / counts
    del bins
    return out


def psd_rmse(true_fields, generated_fields):
    """Log-PSD RMSE between mean spectra (orientation-invariant comparison)."""
    psd_true = np.log(radial_power_spectrum(true_fields) + 1e-8).mean(axis=0)
    psd_gen = np.log(radial_power_spectrum(generated_fields) + 1e-8).mean(axis=0)
    return float(np.sqrt(np.mean((psd_true - psd_gen) ** 2)))


def conditional_c2st(real_fields, generated_fields, conditions, seed=0):
    """Conditional C2ST: classifier accuracy on (field, condition) vs (sample, condition).
    Accuracy near 0.5 means the two distributions are indistinguishable."""
    real = np.concatenate([flatten_fields(real_fields), conditions.reshape(conditions.shape[0], -1)], axis=-1)
    fake = np.concatenate([flatten_fields(generated_fields), conditions.reshape(conditions.shape[0], -1)], axis=-1)
    x = np.concatenate([real, fake], axis=0).astype("float32")
    y = np.concatenate([np.ones(real.shape[0]), np.zeros(fake.shape[0])], axis=0)

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=C.C2ST_TEST_FRAC, random_state=seed, stratify=y
    )
    clf = MLPClassifier(
        hidden_layer_sizes=C.C2ST_HIDDEN,
        max_iter=C.C2ST_MAX_ITER,
        random_state=seed,
        early_stopping=True,
    )
    clf.fit(x_tr, y_tr)
    return float(clf.score(x_te, y_te))


def training_summary(history, train_seconds):
    row = {"train_seconds": float(train_seconds)}
    for key, values in history.history.items():
        if not values:
            continue
        row[f"final_{key}"] = float(values[-1])
        if key.startswith("val_"):
            row[f"best_{key}"] = float(min(values))
    return row


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def print_summary(results):
    print("\nGRF posterior results")
    print(
        f"{'Model':<24} {'RMSE':>8} {'Cov90':>8} "
        f"{'MMD':>9} {'PSD-RMSE':>10} {'C2ST':>7} {'ms/samp':>9}"
    )
    print("-" * 84)
    for name, row in results["evaluation"].items():
        print(
            f"{name:<24} {row['posterior_mean_rmse']:>8.4f} "
            f"{row['coverage_90']:>8.3f} {row['mmd']:>9.5f} "
            f"{row['psd_rmse']:>10.4f} {row['c2st']:>7.3f} "
            f"{row['ms_per_sample']:>9.3f}"
        )

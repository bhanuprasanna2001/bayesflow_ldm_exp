import json
import os
import time

import bayesflow as bf
import keras
import numpy as np

from . import config as C


def sync_and_time(sync_fn):
    sync_fn()
    return time.perf_counter()


def as_image_array(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x.reshape(x.shape[0], *C.IMAGE_SHAPE)
    if x.ndim == 3:
        return x.reshape(x.shape[0], x.shape[1], *C.IMAGE_SHAPE)
    return x


def flatten_images(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 5:
        return x.reshape(x.shape[0], x.shape[1], -1)
    if x.ndim == 4:
        return x.reshape(x.shape[0], -1)
    return x


def evaluate_posterior(samples, targets, clean_reference, sample_seconds):
    del clean_reference  # MMD is computed separately on a larger condition set
    samples_flat = flatten_images(samples)
    targets_flat = flatten_images(targets)
    mean = samples_flat.mean(axis=1)
    std = samples_flat.std(axis=1)

    rmse = float(np.sqrt(np.mean((mean - targets_flat) ** 2)))
    mae = float(np.mean(np.abs(mean - targets_flat)))
    psnr = float(20.0 * np.log10(2.0 / max(rmse, 1e-12)))

    lower = np.quantile(samples_flat, 0.05, axis=1)
    upper = np.quantile(samples_flat, 0.95, axis=1)
    coverage_90 = float(np.mean((targets_flat >= lower) & (targets_flat <= upper)))
    interval_width_90 = float(np.mean(upper - lower))
    posterior_std = float(np.mean(std))

    n_total = samples_flat.shape[0] * samples_flat.shape[1]
    return {
        "posterior_mean_rmse": rmse,
        "posterior_mean_mae": mae,
        "posterior_mean_psnr": psnr,
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


def baseline_metrics(test_data, n_conditions):
    clean = flatten_images(test_data["image"][:n_conditions])
    observed = flatten_images(test_data["observed"][:n_conditions])
    observed_rescaled = (observed / C.NOISE_GAIN) * 2.0 - 1.0
    zeros = np.zeros_like(clean)
    return {
        "zero_rmse": _rmse(zeros, clean),
        "observed_rmse": _rmse(observed_rescaled, clean),
        "zero_psnr": _psnr(_rmse(zeros, clean)),
        "observed_psnr": _psnr(_rmse(observed_rescaled, clean)),
    }


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
    print("\nBayesian denoising results")
    print(
        f"{'Model':<24} {'RMSE':>8} {'PSNR':>8} {'Cov90':>8} "
        f"{'MMD':>9} {'ms/sample':>10}"
    )
    print("-" * 76)
    for name, row in results["evaluation"].items():
        print(
            f"{name:<24} {row['posterior_mean_rmse']:>8.4f} "
            f"{row['posterior_mean_psnr']:>8.2f} {row['coverage_90']:>8.3f} "
            f"{row['mmd']:>9.5f} {row['ms_per_sample']:>10.3f}"
        )
    base = results["baselines"]
    print(f"\nBaselines: zero RMSE={base['zero_rmse']:.4f}, observed RMSE={base['observed_rmse']:.4f}")


def _rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _psnr(rmse):
    return float(20.0 * np.log10(2.0 / max(rmse, 1e-12)))

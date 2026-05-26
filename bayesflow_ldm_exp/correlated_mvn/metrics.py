import json
import os
import time

import numpy as np

from . import config as C


def sync_and_time(sync_fn):
    sync_fn()
    return time.perf_counter()


def training_summary(histories, train_seconds):
    rows = {}
    for name, history in histories.items():
        row = {"train_seconds": float(train_seconds[name])}
        for key, values in history.history.items():
            if not values:
                continue
            row[f"final_{key}"] = float(values[-1])
            if key.startswith("val_"):
                row[f"best_{key}"] = float(min(values))
        rows[name] = row
    return rows


def covariance(samples):
    centered = samples - samples.mean(axis=1, keepdims=True)
    covs = np.einsum("bsd,bse->bde", centered, centered)
    covs /= max(samples.shape[1] - 1, 1)
    return covs.mean(axis=0)


def correlation(cov):
    scale = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return cov / np.outer(scale, scale)


def posterior_metrics(samples, targets, exact_mean, exact_cov, exact_samples):
    sample_mean = samples.mean(axis=1)
    sample_cov = covariance(samples)

    mean_rmse = np.sqrt(np.mean((sample_mean - exact_mean) ** 2))
    cov_error = np.linalg.norm(sample_cov - exact_cov, ord="fro") / np.linalg.norm(exact_cov, ord="fro")
    corr_error = np.mean(np.abs(correlation(sample_cov) - correlation(exact_cov)))

    lower = np.quantile(samples, 0.05, axis=1)
    upper = np.quantile(samples, 0.95, axis=1)
    coverage_90 = np.mean((targets >= lower) & (targets <= upper))

    mmd = average_mmd(samples, exact_samples)

    return {
        "mean_rmse": float(mean_rmse),
        "relative_covariance_error": float(cov_error),
        "correlation_mae": float(corr_error),
        "coverage_90": float(coverage_90),
        "mmd": float(mmd),
    }


def average_mmd(samples, exact_samples):
    n_cond = min(C.MMD_CONDITIONS, samples.shape[0], exact_samples.shape[0])
    n_draws = min(C.MMD_SAMPLES, samples.shape[1], exact_samples.shape[1])
    values = []
    for i in range(n_cond):
        values.append(rbf_mmd(samples[i, :n_draws], exact_samples[i, :n_draws]))
    return float(np.mean(values))


def rbf_mmd(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xy = np.concatenate([x, y], axis=0)
    sq = squared_distances(xy, xy)
    bandwidth_sq = np.median(sq[sq > 0])
    if not np.isfinite(bandwidth_sq) or bandwidth_sq <= 0:
        bandwidth_sq = float(x.shape[-1])

    k_xx = np.exp(-squared_distances(x, x) / (2.0 * bandwidth_sq)).mean()
    k_yy = np.exp(-squared_distances(y, y) / (2.0 * bandwidth_sq)).mean()
    k_xy = np.exp(-squared_distances(x, y) / (2.0 * bandwidth_sq)).mean()
    return k_xx + k_yy - 2.0 * k_xy


def squared_distances(x, y):
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2.0 * x @ y.T, 0.0)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def print_summary(rho, training, evaluation):
    print(f"\nResults for rho={rho}")
    print(f"{'Model':<24} {'Best Val':>10} {'Mean RMSE':>11} {'Cov Err':>9} {'Corr MAE':>9} {'Cov90':>8} {'MMD':>9}")
    print("-" * 88)
    for name, row in evaluation.items():
        best_val = training[name].get("best_val_loss", float("nan"))
        print(
            f"{name:<24} {best_val:>10.4f} {row['mean_rmse']:>11.4f} "
            f"{row['relative_covariance_error']:>9.4f} {row['correlation_mae']:>9.4f} "
            f"{row['coverage_90']:>8.3f} {row['mmd']:>9.4f}"
        )

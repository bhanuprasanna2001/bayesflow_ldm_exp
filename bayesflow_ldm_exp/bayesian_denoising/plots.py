import os

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import config as C
from . import metrics


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def slug(text):
    return text.lower().replace(" ", "_").replace("-", "_")


def plot_losses(histories, output_dir):
    for name, history in histories.items():
        fig = bf.diagnostics.plots.loss(history)
        for ax in fig.axes:
            title = ax.get_title()
            ax.set_title(f"{name} - {title}" if title else name)
        fig.savefig(os.path.join(output_dir, f"loss_{slug(name)}.png"), bbox_inches="tight")
        plt.close(fig)


def plot_main_grid(clean, observed, posterior_means, output_dir):
    model_rows = [C.DISPLAY_NAMES[key] for key in C.MAIN_GRID_MODELS if C.DISPLAY_NAMES[key] in posterior_means]
    rows = ["Clean", "Observed"] + model_rows
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.0, 1.75 * len(rows)))

    for col in range(4):
        _show(axes[0, col], clean[col])
        _show_obs(axes[1, col], observed[col])
        for row, name in enumerate(model_rows, start=2):
            _show(axes[row, col], posterior_means[name][col])

    for row, label in enumerate(rows):
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=42, va="center")
    _strip_axes(axes)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "main_4x4.png"), bbox_inches="tight")
    plt.close(fig)


def plot_model_mean_grid(clean, observed, posterior_means, output_dir):
    rows = ["Clean", "Observed"] + list(posterior_means)
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.0, 1.55 * len(rows)))
    for col in range(4):
        _show(axes[0, col], clean[col])
        _show_obs(axes[1, col], observed[col])
        for row, name in enumerate(posterior_means, start=2):
            _show(axes[row, col], posterior_means[name][col])

    for row, label in enumerate(rows):
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=50, va="center", fontsize=8)
    _strip_axes(axes)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "posterior_means_all_models.png"), bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(model_name, clean, observed, samples, output_dir):
    samples = metrics.as_image_array(samples)
    mean = samples.mean(axis=1)
    std = samples.std(axis=1)
    n_examples = min(4, clean.shape[0])
    n_draws = min(4, samples.shape[1])
    columns = ["Clean", "Observed", "Mean", "Std"] + [f"S{i + 1}" for i in range(n_draws)]
    fig, axes = plt.subplots(n_examples, len(columns), figsize=(1.35 * len(columns), 1.45 * n_examples))

    if n_examples == 1:
        axes = axes[None, :]

    for row in range(n_examples):
        _show(axes[row, 0], clean[row])
        _show_obs(axes[row, 1], observed[row])
        _show(axes[row, 2], mean[row])
        _show(axes[row, 3], std[row], vmin=0.0, vmax=max(float(std.max()), 1e-6))
        for draw in range(n_draws):
            _show(axes[row, 4 + draw], samples[row, draw])

    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=9)
    _strip_axes(axes)
    fig.suptitle(model_name, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"samples_{slug(model_name)}.png"), bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_grid(clean, observed, posterior_samples, output_dir):
    preferred = {C.DISPLAY_NAMES[key] for key in C.MAIN_GRID_MODELS}
    selected = {name: samples for name, samples in posterior_samples.items() if name in preferred}
    if not selected:
        selected = dict(posterior_samples)
    rows = ["Clean", "Observed"]
    for name in selected:
        rows.extend([f"{name} Mean", f"{name} Std"])

    fig, axes = plt.subplots(len(rows), 4, figsize=(7.2, 1.45 * len(rows)))
    for col in range(4):
        _show(axes[0, col], clean[col])
        _show_obs(axes[1, col], observed[col])
        row = 2
        for samples in selected.values():
            arr = metrics.as_image_array(samples)
            mean = arr.mean(axis=1)
            std = arr.std(axis=1)
            _show(axes[row, col], mean[col])
            _show(axes[row + 1, col], std[col], vmin=0.0, vmax=max(float(std.max()), 1e-6))
            row += 2

    for row, label in enumerate(rows):
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=50, va="center", fontsize=8)
    _strip_axes(axes)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "uncertainty_grid.png"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(evaluation, baselines, output_dir):
    rows = []
    for model, values in evaluation.items():
        rows.extend(
            [
                {"model": model, "metric": "RMSE", "value": values["posterior_mean_rmse"]},
                {"model": model, "metric": "MMD", "value": values["mmd"]},
                {"model": model, "metric": "90% coverage", "value": values["coverage_90"]},
                {"model": model, "metric": "ms/sample", "value": values["ms_per_sample"]},
            ]
        )

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, metric_name in zip(axes, ["RMSE", "MMD", "90% coverage", "ms/sample"]):
        subset = [r for r in rows if r["metric"] == metric_name]
        subset_dict = {"value": [r["value"] for r in subset], "model": [r["model"] for r in subset]}
        sns.barplot(data=subset_dict, x="value", y="model", ax=ax, color="#4C78A8")
        ax.set_title(metric_name)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if metric_name == "RMSE":
            ax.axvline(baselines["observed_rmse"], color="#E45756", linestyle="--", linewidth=1.4, label="Observed")
            ax.axvline(baselines["zero_rmse"], color="#72B7B2", linestyle=":", linewidth=1.4, label="Zero")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_barplot.png"), bbox_inches="tight")
    plt.close(fig)


def plot_compute_quality(evaluation, output_dir):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, values in evaluation.items():
        ax.scatter(values["ms_per_sample"], values["posterior_mean_rmse"], s=55)
        ax.annotate(name, (values["ms_per_sample"], values["posterior_mean_rmse"]), fontsize=8)
    ax.set_xlabel("Milliseconds per posterior sample")
    ax.set_ylabel("Posterior mean RMSE")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "compute_quality.png"), bbox_inches="tight")
    plt.close(fig)


def _show(ax, image, vmin=-1.0, vmax=1.0):
    arr = np.asarray(image)
    if arr.ndim == 1:
        arr = arr.reshape(C.IMAGE_SHAPE)
    arr = np.squeeze(arr)
    ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)


def _show_obs(ax, image):
    _show(ax, image, vmin=0.0, vmax=C.NOISE_GAIN)


def _strip_axes(axes):
    for ax in np.asarray(axes).flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

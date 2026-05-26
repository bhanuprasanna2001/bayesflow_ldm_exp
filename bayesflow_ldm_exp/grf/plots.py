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


def plot_main_grid(true_fields, params, posterior_means, output_dir):
    model_rows = [C.DISPLAY_NAMES[key] for key in C.MAIN_GRID_MODELS if C.DISPLAY_NAMES[key] in posterior_means]
    rows = ["True field"] + model_rows
    n = min(4, true_fields.shape[0])
    fig, axes = plt.subplots(len(rows), n, figsize=(1.8 * n, 1.8 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for col in range(n):
        _show(axes[0, col], true_fields[col])
        axes[0, col].set_title(f"α={params[col, 1]:.2f}", fontsize=8)
        for row, name in enumerate(model_rows, start=1):
            _show(axes[row, col], posterior_means[name][col])

    for row, label in enumerate(rows):
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=50, va="center", fontsize=8)
    _strip_axes(axes)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "main_grid.png"), bbox_inches="tight")
    plt.close(fig)


def plot_model_mean_grid(true_fields, params, posterior_means, output_dir):
    rows = ["True"] + list(posterior_means)
    n = min(4, true_fields.shape[0])
    fig, axes = plt.subplots(len(rows), n, figsize=(1.8 * n, 1.5 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for col in range(n):
        _show(axes[0, col], true_fields[col])
        axes[0, col].set_title(f"α={params[col, 1]:.2f}", fontsize=8)
        for row, name in enumerate(posterior_means, start=1):
            _show(axes[row, col], posterior_means[name][col])
    for row, label in enumerate(rows):
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=55, va="center", fontsize=8)
    _strip_axes(axes)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "posterior_means_all_models.png"), bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(model_name, true_fields, params, samples, output_dir):
    samples = metrics.as_field_array(samples)
    mean = samples.mean(axis=1)
    std = samples.std(axis=1)
    n_examples = min(4, true_fields.shape[0])
    n_draws = min(4, samples.shape[1])
    columns = ["True", "Mean", "Std"] + [f"S{i + 1}" for i in range(n_draws)]
    fig, axes = plt.subplots(n_examples, len(columns), figsize=(1.4 * len(columns), 1.55 * n_examples))

    if n_examples == 1:
        axes = axes[None, :]

    std_max = max(float(std.max()), 1e-6)
    for row in range(n_examples):
        _show(axes[row, 0], true_fields[row])
        axes[row, 0].set_ylabel(f"α={params[row, 1]:.2f}", rotation=0, labelpad=30, va="center", fontsize=8)
        _show(axes[row, 1], mean[row])
        _show(axes[row, 2], std[row], vmin=0.0, vmax=std_max, cmap="magma")
        for draw in range(n_draws):
            _show(axes[row, 3 + draw], samples[row, draw])

    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=9)
    _strip_axes(axes, keep_ylabel=True)
    fig.suptitle(model_name, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"samples_{slug(model_name)}.png"), bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_grid(true_fields, posterior_samples, output_dir):
    preferred = {C.DISPLAY_NAMES[key] for key in C.MAIN_GRID_MODELS}
    selected = {name: s for name, s in posterior_samples.items() if name in preferred}
    if not selected:
        selected = dict(posterior_samples)
    rows = ["True"]
    for name in selected:
        rows.extend([f"{name} mean", f"{name} std"])

    n = min(4, true_fields.shape[0])
    fig, axes = plt.subplots(len(rows), n, figsize=(1.7 * n, 1.5 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for col in range(n):
        _show(axes[0, col], true_fields[col])
        row = 1
        for samples in selected.values():
            arr = metrics.as_field_array(samples)
            mean = arr.mean(axis=1)
            std = arr.std(axis=1)
            _show(axes[row, col], mean[col])
            _show(axes[row + 1, col], std[col], vmin=0.0, vmax=max(float(std.max()), 1e-6), cmap="magma")
            row += 2

    for row, label in enumerate(rows):
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=55, va="center", fontsize=8)
    _strip_axes(axes)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "uncertainty_grid.png"), bbox_inches="tight")
    plt.close(fig)


def plot_psd(true_psd_mean, true_psd_std, generated_psd, output_dir):
    """Radial power-spectrum comparison; the headline GRF diagnostic."""
    k = np.arange(1, true_psd_mean.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(k, true_psd_mean, color="black", linewidth=2, label="True")
    ax.fill_between(k, true_psd_mean - true_psd_std, true_psd_mean + true_psd_std, color="black", alpha=0.15)
    palette = sns.color_palette("tab10", n_colors=len(generated_psd))
    for color, (name, psd) in zip(palette, generated_psd.items()):
        ax.plot(k, psd, color=color, linewidth=1.6, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radial wavenumber k")
    ax.set_ylabel("Power")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "psd_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(evaluation, output_dir):
    metric_keys = [
        ("posterior_mean_rmse", "Posterior mean RMSE"),
        ("mmd", "MMD"),
        ("psd_rmse", "log-PSD RMSE"),
        ("c2st", "C2ST acc (0.5 = ideal)"),
        ("coverage_90", "90% coverage"),
        ("ms_per_sample", "ms / sample"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, metric_keys):
        rows = [{"model": m, "value": v[key]} for m, v in evaluation.items()]
        rows_dict = {"value": [r["value"] for r in rows], "model": [r["model"] for r in rows]}
        sns.barplot(data=rows_dict, x="value", y="model", ax=ax, color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if key == "c2st":
            ax.axvline(0.5, color="#E45756", linestyle="--", linewidth=1.4, label="Ideal")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_barplot.png"), bbox_inches="tight")
    plt.close(fig)


def plot_compute_quality(evaluation, output_dir):
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for name, values in evaluation.items():
        ax.scatter(values["ms_per_sample"], values["c2st"], s=60)
        ax.annotate(name, (values["ms_per_sample"], values["c2st"]), fontsize=8)
    ax.axhline(0.5, color="#E45756", linestyle="--", linewidth=1.2, label="C2ST ideal (0.5)")
    ax.set_xlabel("Milliseconds per posterior sample")
    ax.set_ylabel("Conditional C2ST accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "compute_quality.png"), bbox_inches="tight")
    plt.close(fig)


def _show(ax, field, vmin=None, vmax=None, cmap="viridis"):
    arr = np.asarray(field)
    if arr.ndim == 1:
        arr = arr.reshape(C.FIELD_SHAPE)
    arr = np.squeeze(arr)
    if vmin is None or vmax is None:
        mag = float(np.max(np.abs(arr))) + 1e-6
        vmin, vmax = -mag, mag
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)


def _strip_axes(axes, keep_ylabel=False):
    for ax in np.asarray(axes).flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not keep_ylabel:
            pass

import os

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def slug(text):
    return text.replace(" ", "_").replace("=", "").replace(".", "p").lower()


def plot_losses(histories, output_dir):
    for name, history in histories.items():
        fig = bf.diagnostics.plots.loss(history)
        for ax in fig.axes:
            existing = ax.get_title()
            ax.set_title(f"{name} - {existing}" if existing else name)
        fig.savefig(os.path.join(output_dir, f"loss_{slug(name)}.png"), bbox_inches="tight")
        plt.close(fig)


def plot_covariance_heatmaps(exact_cov, model_covs, rho, output_dir):
    n_models = len(model_covs)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(3.2 * (n_models + 1), 6.2))

    vmax = np.max(np.abs(exact_cov))
    _heatmap(axes[0, 0], exact_cov, "Exact", -vmax, vmax, "vlag")
    axes[1, 0].axis("off")

    for col, (name, cov) in enumerate(model_covs.items(), start=1):
        _heatmap(axes[0, col], cov, name, -vmax, vmax, "vlag")
        err = np.abs(cov - exact_cov)
        _heatmap(axes[1, col], err, "|error|", 0.0, np.max(err), "mako")

    fig.suptitle(rf"Posterior Covariance Recovery, $\rho={rho}$")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"covariance_heatmaps_rho_{str(rho).replace('.', 'p')}.png"), bbox_inches="tight")
    plt.close(fig)


def _heatmap(ax, matrix, title, vmin, vmax, cmap):
    sns.heatmap(matrix, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, square=True, cbar=False)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_pair_marginals(exact_samples, model_samples, rho, output_dir):
    dims = C.PAIR_DIMS
    n_models = len(model_samples)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(3.6 * (n_models + 1), 3.8))

    ref = exact_samples[0, :, :]
    _scatter_or_kde(axes[0], ref, dims, "Exact")
    for ax, (name, samples) in zip(axes[1:], model_samples.items()):
        _scatter_or_kde(ax, samples[0, :, :], dims, name)

    fig.suptitle(rf"2D Posterior Marginal, $\rho={rho}$")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"pair_marginal_rho_{str(rho).replace('.', 'p')}.png"), bbox_inches="tight")
    plt.close(fig)


def _scatter_or_kde(ax, samples, dims, title):
    x = samples[:, dims[0]]
    y = samples[:, dims[1]]
    sns.kdeplot(x=x, y=y, ax=ax, fill=True, levels=12, thresh=0.05, cmap="Blues")
    ax.scatter(x, y, s=5, alpha=0.18, color="black", linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(rf"$\theta_{{{dims[0] + 1}}}$")
    ax.set_ylabel(rf"$\theta_{{{dims[1] + 1}}}$")


def plot_metrics_vs_rho(all_results, output_dir):
    metrics = [
        ("mean_rmse", "Mean RMSE"),
        ("relative_covariance_error", "Covariance Error"),
        ("correlation_mae", "Correlation MAE"),
        ("coverage_90", "90% Coverage"),
        ("mmd", "MMD"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for ax, (key, title) in zip(axes, metrics):
        for model_name in next(iter(all_results.values()))["evaluation"]:
            xs = []
            ys = []
            for rho, result in all_results.items():
                xs.append(float(rho))
                ys.append(result["evaluation"][model_name][key])
            ax.plot(xs, ys, marker="o", label=model_name)
        ax.set_title(title)
        ax.set_xlabel(r"$\rho$")
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_vs_rho.png"), bbox_inches="tight")
    plt.close(fig)


def plot_compute_quality(all_results, output_dir):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for rho, result in all_results.items():
        for name, row in result["evaluation"].items():
            ax.scatter(row["ms_per_sample"], row["relative_covariance_error"], s=55)
            ax.annotate(f"{name}, rho={rho}", (row["ms_per_sample"], row["relative_covariance_error"]), fontsize=8)
    ax.set_xlabel("Milliseconds per posterior sample")
    ax.set_ylabel("Relative covariance error")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "compute_quality.png"), bbox_inches="tight")
    plt.close(fig)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bayesflow as bf

from . import config as C

# Match the notebook's visual style globally
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def plot_losses(histories):
    for name, history in histories.items():
        fig = bf.diagnostics.plots.loss(history)
        for ax in fig.axes:
            existing = ax.get_title()
            ax.set_title(f"{name} — {existing}" if existing else name)
        fig.savefig(
            os.path.join(C.OUTPUT_DIR, f"loss_{name.replace(' ', '_').lower()}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_posterior_scatter(all_samples):
    n = len(all_samples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, s) in zip(axes, all_samples.items()):
        ax.scatter(s[:, 0], s[:, 1], alpha=0.3, s=5)
        ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$", title=name,
               xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), aspect="equal")
    fig.suptitle(r"Posterior Samples at $x = (0, 0)$")
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUTPUT_DIR, "posterior_scatter.png"), bbox_inches="tight")
    plt.close(fig)


def plot_posterior_kde(all_samples):
    n = len(all_samples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, s) in zip(axes, all_samples.items()):
        sns.kdeplot(x=s[:, 0], y=s[:, 1], ax=ax, fill=True, levels=20, cmap="Blues")
        ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$", title=name,
               xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), aspect="equal")
    fig.suptitle(r"Posterior Density at $x = (0, 0)$")
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUTPUT_DIR, "posterior_kde.png"), bbox_inches="tight")
    plt.close(fig)


def plot_calibration_ecdf(name, sbc_samples, targets):
    fig = bf.diagnostics.plots.calibration_ecdf(
        estimates=sbc_samples,
        targets=targets,
        variable_names=[r"$\theta_1$", r"$\theta_2$"],
        difference=True,
        stacked=True,
        figsize=(5, 4),
    )
    for ax in fig.axes:
        existing = ax.get_title()
        ax.set_title(f"{name} — {existing}" if existing else f"{name} — Calibration ECDF")
    fig.tight_layout()
    fig.savefig(
        os.path.join(C.OUTPUT_DIR, f"calibration_ecdf_{name.replace(' ', '_').lower()}.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

def plot_kde_bw_ablation(all_samples, bw_values=None):
    if bw_values is None:
        bw_values = np.round(np.arange(0.1, 1.05, 0.1), 2)
    ncols = 5
    nrows = int(np.ceil(len(bw_values) / ncols))
    for name, s in all_samples.items():
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, bw in zip(axes, bw_values):
            sns.kdeplot(
                x=s[:, 0], y=s[:, 1], ax=ax,
                fill=True, levels=20, cmap="Blues",
                bw_adjust=float(bw),
            )
            ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$",
                   title=f"bw\\_adjust = {bw:.1f}",
                   xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), aspect="equal")
        for ax in axes[len(bw_values):]:
            ax.axis("off")
        fig.suptitle(f"{name} — KDE bandwidth ablation at $x = (0, 0)$")
        fig.tight_layout()
        fig.savefig(
            os.path.join(C.OUTPUT_DIR, f"kde_ablation_{name.replace(' ', '_').lower()}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_posterior_hist2d(all_samples, bins=100):
    n = len(all_samples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, s) in zip(axes, all_samples.items()):
        ax.hist2d(
            s[:, 0], s[:, 1],
            bins=bins, range=[[-0.5, 0.5], [-0.5, 0.5]],
            cmap="Blues",
        )
        ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$", title=name,
               xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), aspect="equal")
    fig.suptitle(rf"Posterior 2D histogram at $x = (0, 0)$")
    fig.tight_layout()
    fig.savefig(
        os.path.join(C.OUTPUT_DIR, f"posterior_hist2d_{bins}.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

import argparse
import gc
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch

import bayesflow as bf
from . import config as C
from . import metrics
from . import models
from . import plots
from . import simulator


def torch_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def clear_memory():
    gc.collect()
    torch_sync()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def make_dataset(split, spec, adapter=None):
    adapter = adapter or models.build_adapter(spec)
    return bf.datasets.OfflineDataset(data=split, adapter=adapter, batch_size=spec.batch_size)


def sample_model(approx, spec, split, num_conditions, num_samples):
    conditions = {spec.condition_key: split[spec.condition_key][:num_conditions].astype("float32")}
    return approx.sample(
        num_samples=num_samples,
        conditions=conditions,
        batch_size=spec.sample_batch_size,
        sample_shape=spec.sample_shape,
    )[spec.target_key]


def train_and_evaluate(spec, splits, output_dir, epochs):
    clear_memory()
    print(f"\n{'=' * 72}\nTraining {spec.name}\n{'=' * 72}")

    approx = models.build_approximator(spec)
    train_ds = make_dataset(splits["train"], spec, adapter=approx.adapter)
    val_ds = make_dataset(splits["validation"], spec, adapter=approx.adapter)

    approx.compile(optimizer=keras.optimizers.Adam(learning_rate=C.LEARNING_RATE))
    start = metrics.sync_and_time(torch_sync)
    history = approx.fit(dataset=train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    train_seconds = metrics.sync_and_time(torch_sync) - start

    clear_memory()
    eval_n = min(C.EVAL_CONDITIONS, splits["test"]["field"].shape[0])
    print(f"\nSampling {spec.name}: {eval_n} conditions x {C.POSTERIOR_SAMPLES} draws")
    start = metrics.sync_and_time(torch_sync)
    eval_samples = sample_model(approx, spec, splits["test"], eval_n, C.POSTERIOR_SAMPLES)
    sample_seconds = metrics.sync_and_time(torch_sync) - start

    eval_targets = splits["test"]["field"][:eval_n]
    evaluation = metrics.evaluate_posterior(
        metrics.as_field_array(eval_samples), eval_targets, sample_seconds
    )

    distrib_n = min(C.MMD_CONDITIONS, splits["test"]["field"].shape[0])
    print(f"Sampling {spec.name}: {distrib_n} conditions x 1 draw (MMD/PSD/C2ST)")
    distrib_samples = sample_model(approx, spec, splits["test"], distrib_n, 1)
    distrib_samples = metrics.as_field_array(distrib_samples)[:, 0]  # (B, H, W, 1)

    targets_flat = metrics.flatten_fields(splits["test"]["field"][:distrib_n])
    samples_flat = metrics.flatten_fields(distrib_samples)
    mmd_values = metrics.chunked_mmd(targets_flat, samples_flat, splits=C.MMD_SPLITS)
    evaluation["mmd"] = float(np.mean(mmd_values))
    evaluation["mmd_std"] = float(np.std(mmd_values))

    evaluation["psd_rmse"] = metrics.psd_rmse(
        splits["test"]["field"][:distrib_n], distrib_samples
    )

    c2st_n = min(C.C2ST_CONDITIONS, distrib_n)
    evaluation["c2st"] = metrics.conditional_c2st(
        real_fields=splits["test"]["field"][:c2st_n],
        generated_fields=distrib_samples[:c2st_n],
        conditions=splits["test"]["params"][:c2st_n],
        seed=C.SEED,
    )

    plot_indices = simulator.select_diverse_indices(splits["test"], n=4)
    plot_split = {k: v[plot_indices] for k, v in splits["test"].items()}
    plot_samples = sample_model(approx, spec, plot_split, len(plot_indices), C.PLOT_POSTERIOR_SAMPLES)
    plot_samples = metrics.as_field_array(plot_samples)
    plots.plot_sample_grid(
        spec.name,
        true_fields=plot_split["field"],
        params=plot_split["params"],
        samples=plot_samples,
        output_dir=output_dir,
    )

    # mean radial PSD over distrib samples — used for the comparison plot
    psd_generated = metrics.radial_power_spectrum(distrib_samples).mean(axis=0)

    training = metrics.training_summary(history, train_seconds)
    clear_memory()
    return history, training, evaluation, plot_samples, psd_generated


def main():
    args = parse_args()
    np.random.seed(C.SEED)
    keras.utils.set_random_seed(C.SEED)
    output_dir = str(C.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    selected = models.selected_specs(args.models)
    print("Generating GRF splits (this is one-time and uses FyeldGenerator)…")
    splits = simulator.load_grf_splits(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        seed=C.SEED,
    )

    histories = {}
    training = {}
    evaluation = {}
    plot_samples = {}
    psd_generated = {}

    for spec in selected:
        history, train_row, eval_row, samples, psd_g = train_and_evaluate(
            spec, splits, output_dir, args.epochs
        )
        histories[spec.name] = history
        training[spec.name] = train_row
        evaluation[spec.name] = eval_row
        plot_samples[spec.name] = samples
        psd_generated[spec.name] = psd_g

    plot_indices = simulator.select_diverse_indices(splits["test"], n=4)
    true_fields = splits["test"]["field"][plot_indices]
    params = splits["test"]["params"][plot_indices]
    posterior_means = {name: samples.mean(axis=1) for name, samples in plot_samples.items()}

    distrib_n = min(C.MMD_CONDITIONS, splits["test"]["field"].shape[0])
    true_psd_all = metrics.radial_power_spectrum(splits["test"]["field"][:distrib_n])
    true_psd_mean = true_psd_all.mean(axis=0)
    true_psd_std = true_psd_all.std(axis=0)

    plots.plot_losses(histories, output_dir)
    plots.plot_main_grid(true_fields, params, posterior_means, output_dir)
    plots.plot_model_mean_grid(true_fields, params, posterior_means, output_dir)
    plots.plot_uncertainty_grid(true_fields, plot_samples, output_dir)
    plots.plot_psd(true_psd_mean, true_psd_std, psd_generated, output_dir)
    plots.plot_metric_bars(evaluation, output_dir)
    plots.plot_compute_quality(evaluation, output_dir)

    result = {
        "config": {
            "field_shape": list(C.FIELD_SHAPE),
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "test_samples": args.test_samples,
            "epochs": args.epochs,
            "warmup_steps": C.WARMUP_STEPS,
            "posterior_samples": C.POSTERIOR_SAMPLES,
            "integrate_kwargs": C.INTEGRATE_KWARGS,
            "prior": {
                "log_std_loc": C.LOG_STD_LOC,
                "log_std_scale": C.LOG_STD_SCALE,
                "alpha_loc": C.ALPHA_LOC,
                "alpha_scale": C.ALPHA_SCALE,
            },
        },
        "training": training,
        "evaluation": evaluation,
    }
    metrics.save_json(result, os.path.join(output_dir, "metrics.json"))
    metrics.print_summary(result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--train-samples", type=int, default=C.TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=C.VAL_SAMPLES)
    parser.add_argument("--test-samples", type=int, default=C.TEST_SAMPLES)
    parser.add_argument("--models", nargs="*", default=list(C.MODEL_ORDER), choices=list(models.SPECS))
    return parser.parse_args()


if __name__ == "__main__":
    main()

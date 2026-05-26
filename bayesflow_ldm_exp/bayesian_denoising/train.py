import argparse
import gc
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch

import bayesflow as bf
from . import config as C
from . import data
from . import metrics
from . import models
from . import plots


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
    eval_n = min(C.EVAL_CONDITIONS, splits["test"][spec.target_key].shape[0])
    print(f"\nSampling {spec.name} for metrics")
    start = metrics.sync_and_time(torch_sync)
    eval_samples = sample_model(approx, spec, splits["test"], eval_n, C.POSTERIOR_SAMPLES)
    sample_seconds = metrics.sync_and_time(torch_sync) - start

    mmd_n = min(C.MMD_CONDITIONS, splits["test"][spec.target_key].shape[0])
    print(f"Sampling {spec.name} for MMD")
    mmd_samples = sample_model(approx, spec, splits["test"], mmd_n, 1)

    eval_clean = splits["test"][spec.target_key][:eval_n]
    mmd_clean = splits["test"][spec.target_key][:mmd_n]
    evaluation = metrics.evaluate_posterior(eval_samples, eval_clean, mmd_clean, sample_seconds)
    mmd_values = metrics.chunked_mmd(
        metrics.flatten_images(mmd_clean),
        metrics.flatten_images(mmd_samples)[:, 0],
        splits=C.MMD_SPLITS,
    )
    evaluation["mmd"] = float(np.mean(mmd_values))
    evaluation["mmd_std"] = float(np.std(mmd_values))

    plot_indices = data.select_one_per_class(splits["test"], max_classes=4)
    plot_split = {k: v[plot_indices] for k, v in splits["test"].items()}
    plot_samples = sample_model(approx, spec, plot_split, len(plot_indices), C.PLOT_POSTERIOR_SAMPLES)
    plots.plot_sample_grid(
        spec.name,
        clean=plot_split["image"],
        observed=plot_split["observed"],
        samples=metrics.as_image_array(plot_samples),
        output_dir=output_dir,
    )

    training = metrics.training_summary(history, train_seconds)
    clear_memory()
    return approx, history, training, evaluation, metrics.as_image_array(plot_samples)


def main():
    args = parse_args()
    np.random.seed(C.SEED)
    keras.utils.set_random_seed(C.SEED)
    output_dir = str(C.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    selected = models.selected_specs(args.models)
    splits = data.load_fashion_mnist_splits(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        seed=C.SEED,
    )

    histories = {}
    training = {}
    evaluation = {}
    plot_samples = {}

    for spec in selected:
        _, history, train_row, eval_row, samples = train_and_evaluate(spec, splits, output_dir, args.epochs)
        histories[spec.name] = history
        training[spec.name] = train_row
        evaluation[spec.name] = eval_row
        plot_samples[spec.name] = samples

    plot_indices = data.select_one_per_class(splits["test"], max_classes=4)
    clean = splits["test"]["image"][plot_indices]
    observed = splits["test"]["observed"][plot_indices]
    posterior_means = {name: samples.mean(axis=1) for name, samples in plot_samples.items()}

    baselines = metrics.baseline_metrics(splits["test"], min(C.EVAL_CONDITIONS, args.test_samples))
    plots.plot_losses(histories, output_dir)
    plots.plot_main_grid(clean, observed, posterior_means, output_dir)
    plots.plot_model_mean_grid(clean, observed, posterior_means, output_dir)
    plots.plot_uncertainty_grid(clean, observed, _selected_uncertainty_samples(plot_samples), output_dir)
    plots.plot_metric_bars(evaluation, baselines, output_dir)
    plots.plot_compute_quality(evaluation, output_dir)

    result = {
        "config": {
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "test_samples": args.test_samples,
            "epochs": args.epochs,
            "warmup_steps": C.WARMUP_STEPS,
            "posterior_samples": C.POSTERIOR_SAMPLES,
            "integrate_kwargs": C.INTEGRATE_KWARGS,
            "noise": {
                "mode": C.NOISE_MODE,
                "psf_width": C.PSF_WIDTH,
                "noise_scale": C.NOISE_SCALE,
                "noise_gain": C.NOISE_GAIN,
            },
        },
        "training": training,
        "evaluation": evaluation,
        "baselines": baselines,
    }
    metrics.save_json(result, os.path.join(output_dir, "metrics.json"))
    metrics.print_summary(result)


def _selected_uncertainty_samples(plot_samples):
    preferred = {C.DISPLAY_NAMES[key] for key in C.MAIN_GRID_MODELS}
    selected = {name: samples for name, samples in plot_samples.items() if name in preferred}
    return selected or plot_samples


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

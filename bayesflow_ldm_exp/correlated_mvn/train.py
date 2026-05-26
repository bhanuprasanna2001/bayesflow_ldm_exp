import gc
import os

import bayesflow as bf
import keras
import numpy as np
import torch

from . import config as C
from . import metrics
from . import plots
from . import simulator as sim


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


def build_diffusion():
    return bf.networks.DiffusionModel(
        subnet="time_mlp",
        subnet_kwargs=C.SUBNET_KWARGS,
        noise_schedule="cosine",
        integrate_kwargs=C.INTEGRATE_KWARGS,
    )


def build_models(adapter):
    models = {
        "DiffusionModel": bf.ContinuousApproximator(
            adapter=adapter,
            inference_network=build_diffusion(),
        ),
    }
    for latent_dim in C.LATENT_DIMS:
        models[f"Latent Diffusion z{latent_dim}"] = bf.ContinuousApproximator(
            adapter=adapter,
            inference_network=bf.networks.LatentInferenceNetwork(
                inference_network=build_diffusion(),
                latent_dim=latent_dim,
                **C.LDM_KWARGS,
            ),
        )
    return models


def train_one_rho(rho):
    output_dir = os.path.join(C.OUTPUT_DIR, f"rho_{str(rho).replace('.', 'p')}")
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(C.SEED + int(1000 * rho))
    problem = sim.build_problem(C.DIM, rho, C.OBSERVED_INDICES, C.OBS_NOISE)
    train_data = sim.simulate(problem, C.TRAIN_SAMPLES, rng)
    val_data = sim.simulate(problem, C.VAL_SAMPLES, rng)

    adapter = bf.ContinuousApproximator.build_adapter(
        inference_variables=["parameters"],
        inference_conditions=["observables"],
    )
    train_ds = bf.datasets.OfflineDataset(data=train_data, adapter=adapter, batch_size=C.BATCH_SIZE)
    val_ds = bf.datasets.OfflineDataset(data=val_data, adapter=adapter, batch_size=C.BATCH_SIZE)

    models = build_models(adapter)
    histories = {}
    train_seconds = {}
    for name, approx in models.items():
        clear_memory()
        print(f"\n{'=' * 60}\nTraining {name} at rho={rho}\n{'=' * 60}")
        approx.compile(optimizer=keras.optimizers.Adam(learning_rate=C.LEARNING_RATE))
        start = metrics.sync_and_time(torch_sync)
        histories[name] = approx.fit(dataset=train_ds, validation_data=val_ds, epochs=C.EPOCHS, verbose=1)
        train_seconds[name] = metrics.sync_and_time(torch_sync) - start
        clear_memory()

    plots.plot_losses(histories, output_dir)

    test_data = sim.simulate(problem, C.TEST_CONDITIONS, rng)
    conditions = {"observables": test_data["observables"].astype("float32")}
    exact = sim.exact_posterior(problem, test_data["observables"])
    exact_samples = sim.sample_posterior(problem, test_data["observables"], C.POSTERIOR_SAMPLES, rng)

    model_samples = {}
    model_covs = {}
    evaluation = {}
    for name, approx in models.items():
        clear_memory()
        print(f"\nSampling {name} at rho={rho}")
        start = metrics.sync_and_time(torch_sync)
        samples = approx.sample(
            num_samples=C.POSTERIOR_SAMPLES,
            conditions=conditions,
            batch_size=C.SAMPLE_BATCH_SIZE,
            **C.INTEGRATE_KWARGS,
        )["parameters"]
        sample_seconds = metrics.sync_and_time(torch_sync) - start
        samples = np.asarray(samples, dtype=np.float32)

        model_samples[name] = samples
        model_covs[name] = metrics.covariance(samples)
        evaluation[name] = metrics.posterior_metrics(
            samples=samples,
            targets=test_data["parameters"],
            exact_mean=exact["mean"],
            exact_cov=exact["covariance"],
            exact_samples=exact_samples,
        )
        evaluation[name]["sample_seconds"] = float(sample_seconds)
        evaluation[name]["ms_per_sample"] = float(1000.0 * sample_seconds / samples[:, :, 0].size)
        clear_memory()

    plots.plot_covariance_heatmaps(exact["covariance"], model_covs, rho, output_dir)
    plots.plot_pair_marginals(exact_samples, model_samples, rho, output_dir)

    training = metrics.training_summary(histories, train_seconds)
    result = {
        "rho": rho,
        "config": {
            "dim": C.DIM,
            "observed_indices": list(C.OBSERVED_INDICES),
            "obs_noise": C.OBS_NOISE,
            "latent_dims": list(C.LATENT_DIMS),
            "posterior_samples": C.POSTERIOR_SAMPLES,
        },
        "training": training,
        "evaluation": evaluation,
    }
    metrics.save_json(result, os.path.join(output_dir, "metrics.json"))
    metrics.print_summary(rho, training, evaluation)
    return result


def main():
    np.random.seed(C.SEED)
    keras.utils.set_random_seed(C.SEED)
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    all_results = {}
    for rho in C.RHOS:
        all_results[str(rho)] = train_one_rho(rho)

    metrics.save_json(all_results, os.path.join(C.OUTPUT_DIR, "metrics.json"))
    plots.plot_metrics_vs_rho(all_results, C.OUTPUT_DIR)
    plots.plot_compute_quality(all_results, C.OUTPUT_DIR)


if __name__ == "__main__":
    main()

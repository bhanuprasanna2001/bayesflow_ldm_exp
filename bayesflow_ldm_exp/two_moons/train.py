import os
import gc
import numpy as np
import keras
import torch
import bayesflow as bf

from . import config as C
from . import plots
from . import metrics


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


def build_models(adapter):
    return {
        "DiffusionModel": bf.ContinuousApproximator(
            adapter=adapter,
            inference_network=bf.networks.DiffusionModel(
                subnet="time_mlp",
                subnet_kwargs=C.SUBNET_KWARGS,
                noise_schedule="cosine",
                integrate_kwargs=C.INTEGRATE_KWARGS,
            ),
        ),
        "Latent Diffusion": bf.ContinuousApproximator(
            adapter=adapter,
            inference_network=bf.networks.LatentInferenceNetwork(
                inference_network=bf.networks.DiffusionModel(
                    subnet="time_mlp",
                    subnet_kwargs=C.SUBNET_KWARGS,
                    noise_schedule="cosine",
                    integrate_kwargs=C.INTEGRATE_KWARGS,
                ),
                **C.LDM_KWARGS,
            ),
        ),
    }


def main():
    np.random.seed(C.SEED)
    keras.utils.set_random_seed(C.SEED)
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    simulator = bf.simulators.TwoMoons()
    train_data = simulator.sample(C.TRAIN_SAMPLES)
    val_data = simulator.sample(C.VAL_SAMPLES)

    adapter = bf.ContinuousApproximator.build_adapter(
        inference_variables=["parameters"],
        inference_conditions=["observables"],
    )
    train_ds = bf.datasets.OfflineDataset(data=train_data, adapter=adapter, batch_size=C.BATCH_SIZE)
    val_ds = bf.datasets.OfflineDataset(data=val_data, adapter=adapter, batch_size=C.BATCH_SIZE)

    models = build_models(adapter)
    histories = {}
    for name, approx in models.items():
        clear_memory()
        print(f"\n{'=' * 50}\nTraining {name}\n{'=' * 50}")
        approx.compile(optimizer=keras.optimizers.Adam(learning_rate=C.LEARNING_RATE))
        histories[name] = approx.fit(
            dataset=train_ds, validation_data=val_ds, epochs=C.EPOCHS, verbose=1
        )
        clear_memory()

    plots.plot_losses(histories)

    test_condition = {"observables": np.array(C.TEST_CONDITION, dtype=np.float32)}
    all_samples = {}
    for name, approx in models.items():
        clear_memory()
        s = approx.sample(
            num_samples=C.POSTERIOR_SAMPLES,
            conditions=test_condition,
            batch_size=C.SAMPLE_BATCH_SIZE,
            **C.INTEGRATE_KWARGS,
        )["parameters"]
        all_samples[name] = s.reshape(-1, s.shape[-1]) if s.ndim == 3 else s
        clear_memory()

    plots.plot_posterior_scatter(all_samples)
    plots.plot_posterior_kde(all_samples)

    sbc_data = simulator.sample(C.SBC_SIMULATIONS)
    sbc_conditions = {"observables": sbc_data["observables"].astype(np.float32)}
    calibration_per_model = {}
    for name, approx in models.items():
        clear_memory()
        print(f"\nSBC: {name}")
        sbc_samples = approx.sample(
            num_samples=C.SBC_POSTERIOR_SAMPLES,
            conditions=sbc_conditions,
            batch_size=C.SAMPLE_BATCH_SIZE,
            **C.INTEGRATE_KWARGS,
        )["parameters"]
        plots.plot_calibration_ecdf(name, sbc_samples, sbc_data["parameters"])
        calibration_per_model[name] = metrics.calibration(sbc_samples, sbc_data["parameters"])
        clear_memory()

    training = metrics.training_summary(histories)
    mode_cov = metrics.mode_coverage(all_samples)
    metrics.save_json(
        {"training": training, "mode_coverage": mode_cov, "calibration": calibration_per_model},
        "metrics.json",
    )
    metrics.print_summary(training, mode_cov, calibration_per_model)


if __name__ == "__main__":
    main()

from dataclasses import dataclass

import bayesflow as bf

from . import config as C
from ..spatial import ConditionResizeUNet, ConvDownsampler, ConvUpsampler


class GRFApproximator(bf.ContinuousApproximator):
    """ContinuousApproximator with a safe symbolic-build call for Keras torch."""

    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            return self.compute_metrics(
                inference_variables=inputs.get("inference_variables"),
                inference_conditions=inputs.get("inference_conditions"),
                summary_variables=inputs.get("summary_variables"),
                sample_weight=inputs.get("sample_weight"),
                stage="validation",
            )["loss"]
        return inputs


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    family: str
    target_key: str
    condition_key: str
    batch_size: int
    sample_batch_size: int
    sample_shape: tuple[int, ...] | str = "infer"


SPECS = {
    "spatial_direct": ModelSpec(
        key="spatial_direct",
        name=C.DISPLAY_NAMES["spatial_direct"],
        family="spatial",
        target_key="field",
        condition_key="params_expanded",
        batch_size=C.SPATIAL_BATCH_SIZE,
        sample_batch_size=C.SPATIAL_SAMPLE_BATCH_SIZE,
    ),
    "spatial_ldm_16x16x2": ModelSpec(
        key="spatial_ldm_16x16x2",
        name=C.DISPLAY_NAMES["spatial_ldm_16x16x2"],
        family="spatial",
        target_key="field",
        condition_key="params_expanded",
        batch_size=C.SPATIAL_BATCH_SIZE,
        sample_batch_size=C.SPATIAL_SAMPLE_BATCH_SIZE,
        sample_shape=(16, 16),
    ),
    "spatial_ldm_8x8x4": ModelSpec(
        key="spatial_ldm_8x8x4",
        name=C.DISPLAY_NAMES["spatial_ldm_8x8x4"],
        family="spatial",
        target_key="field",
        condition_key="params_expanded",
        batch_size=C.SPATIAL_BATCH_SIZE,
        sample_batch_size=C.SPATIAL_SAMPLE_BATCH_SIZE,
        sample_shape=(8, 8),
    ),
}


def selected_specs(keys=None):
    keys = tuple(keys or C.MODEL_ORDER)
    unknown = sorted(set(keys) - set(SPECS))
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Available: {list(SPECS)}")
    return [SPECS[key] for key in keys]


def build_adapter(spec):
    return bf.ContinuousApproximator.build_adapter(
        inference_variables=[spec.target_key],
        inference_conditions=[spec.condition_key],
    )


def build_approximator(spec):
    adapter = build_adapter(spec)
    return GRFApproximator(
        adapter=adapter,
        inference_network=build_inference_network(spec.key),
        standardize="inference_variables",
    )


def build_inference_network(key):
    if key == "spatial_direct":
        return _spatial_diffusion()
    if key == "spatial_ldm_16x16x2":
        return _spatial_latent_diffusion(latent_shape=(16, 16, 2), downsample_steps=1)
    if key == "spatial_ldm_8x8x4":
        return _spatial_latent_diffusion(latent_shape=(8, 8, 4), downsample_steps=2)
    raise ValueError(f"Unknown model key: {key}")


def _spatial_diffusion():
    return bf.networks.DiffusionModel(
        subnet=bf.networks.UNet,
        subnet_kwargs=C.UNET_KWARGS,
        noise_schedule="cosine",
        integrate_kwargs=C.INTEGRATE_KWARGS,
    )


def _spatial_latent_diffusion(latent_shape, downsample_steps):
    encoder_widths = (32, 64) if downsample_steps == 1 else (32, 64, 96)
    decoder_widths = tuple(reversed(encoder_widths))
    return bf.networks.LatentInferenceNetwork(
        inference_network=bf.networks.DiffusionModel(
            subnet=ConditionResizeUNet,
            subnet_kwargs={"unet_kwargs": C.UNET_KWARGS_LATENT},
            noise_schedule="cosine",
            integrate_kwargs=C.INTEGRATE_KWARGS,
        ),
        latent_shape=latent_shape,
        encoder=ConvDownsampler,
        decoder=ConvUpsampler,
        encoder_kwargs={"widths": encoder_widths, "downsample_steps": downsample_steps},
        decoder_kwargs={"widths": decoder_widths, "upsample_steps": downsample_steps},
        **C.LDM_KWARGS,
    )

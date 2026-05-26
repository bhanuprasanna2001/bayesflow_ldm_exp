import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"

SEED = 42

IMAGE_SHAPE = (28, 28, 1)

TRAIN_SAMPLES = 10_240
VAL_SAMPLES = 1_280
TEST_SAMPLES = 1_000
EPOCHS = 100

LEARNING_RATE = 5e-4
WARMUP_STEPS = 1_000

SPATIAL_BATCH_SIZE = 64
SPATIAL_SAMPLE_BATCH_SIZE = 8

POSTERIOR_SAMPLES = 100
PLOT_POSTERIOR_SAMPLES = 128
EVAL_CONDITIONS = 100
MMD_CONDITIONS = 1_000
MMD_SPLITS = 4

NOISE_MODE = "poisson"
PSF_WIDTH = 2.5
NOISE_SCALE = 1.0
NOISE_GAIN = 0.5

INTEGRATE_KWARGS = {"method": "euler", "steps": 30}

UNET_KWARGS = {
    "widths": (32, 64, 128),
    "res_blocks": 2,
    "attn_stage": (False, False, True),
    "time_emb_dim": 64,
    "activation": "mish",
    "dropout": 0.0,
    "groups": 8,
    "norm": "group",
}

UNET_KWARGS_LATENT = {
    "widths": (32, 64),
    "res_blocks": 2,
    "attn_stage": (False, True),
    "time_emb_dim": 64,
    "activation": "mish",
    "dropout": 0.0,
    "groups": 8,
    "norm": "group",
}

LDM_KWARGS = {
    "kl_weight": 1e-6,
    "reconstruction_weight": 1.0,
    "warmup_steps": WARMUP_STEPS,
}

MODEL_ORDER = (
    "spatial_direct",
    "spatial_ldm_14x14x2",
    "spatial_ldm_7x7x4",
)

DISPLAY_NAMES = {
    "spatial_direct": "U-Net Direct",
    "spatial_ldm_14x14x2": "Spatial LDM 14x14x2",
    "spatial_ldm_7x7x4": "Spatial LDM 7x7x4",
}

MAIN_GRID_MODELS = ("spatial_direct", "spatial_ldm_7x7x4")
CLASS_NAMES = (
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

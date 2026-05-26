import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"

SEED = 42

FIELD_SHAPE = (32, 32, 1)
PARAM_DIM = 2

TRAIN_SAMPLES = 5_000
VAL_SAMPLES = 500
TEST_SAMPLES = 1_000
EPOCHS = 100

LEARNING_RATE = 5e-4
WARMUP_STEPS = 1_000

SPATIAL_BATCH_SIZE = 32
SPATIAL_SAMPLE_BATCH_SIZE = 8

POSTERIOR_SAMPLES = 100
PLOT_POSTERIOR_SAMPLES = 64
EVAL_CONDITIONS = 100
MMD_CONDITIONS = 1_000
MMD_SPLITS = 4

C2ST_CONDITIONS = 1_000
C2ST_HIDDEN = (128, 64)
C2ST_MAX_ITER = 200
C2ST_TEST_FRAC = 0.2

# GRF prior.
LOG_STD_LOC = 0.0
LOG_STD_SCALE = 0.3
ALPHA_LOC = 3.0
ALPHA_SCALE = 0.5

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
    "spatial_ldm_16x16x2",
    "spatial_ldm_8x8x4",
)

DISPLAY_NAMES = {
    "spatial_direct": "U-Net Direct",
    "spatial_ldm_16x16x2": "Spatial LDM 16x16x2",
    "spatial_ldm_8x8x4": "Spatial LDM 8x8x4",
}

MAIN_GRID_MODELS = ("spatial_direct", "spatial_ldm_8x8x4")

import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

BASE_DIR = Path(__file__).resolve().parent

LATENT_DIM = 2
TRAIN_SAMPLES = 32768
VAL_SAMPLES = 2048
EPOCHS = 100
BATCH_SIZE = 128
SAMPLE_BATCH_SIZE = 64
LEARNING_RATE = 5e-4

SBC_SIMULATIONS = 300
POSTERIOR_SAMPLES = 2000
SBC_POSTERIOR_SAMPLES = 1000

SUBNET_KWARGS = {"widths": (128, 128, 128)}
ENCODER_DECODER_KWARGS = {"widths": (128, 128)}
LDM_KWARGS = dict(
    latent_dim=LATENT_DIM,
    encoder_kwargs=ENCODER_DECODER_KWARGS,
    decoder_kwargs=ENCODER_DECODER_KWARGS,
    kl_weight=1e-6,
    reconstruction_weight=1.0,
    warmup_steps=500,
)
INTEGRATE_KWARGS = {"method": "rk45", "steps": 50}

TEST_CONDITION = [[0.0, 0.0]]
OUTPUT_DIR = BASE_DIR / "results"
SEED = 42

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

DIM = 32
OBSERVED_INDICES = tuple(range(0, DIM, 2))
OBS_NOISE = 0.20
RHOS = (0.0, 0.6, 0.9)

LATENT_DIMS = (24, 16)

TRAIN_SAMPLES = 50_000
VAL_SAMPLES = 5_000
TEST_CONDITIONS = 100
EPOCHS = 100
BATCH_SIZE = 256
SAMPLE_BATCH_SIZE = 16
LEARNING_RATE = 5e-4

POSTERIOR_SAMPLES = 1024
MMD_CONDITIONS = 16
MMD_SAMPLES = 256
PAIR_DIMS = (14, 15)

SUBNET_KWARGS = {"widths": (128, 128, 128)}
ENCODER_DECODER_KWARGS = {"widths": (128, 128)}
LDM_KWARGS = dict(
    encoder_kwargs=ENCODER_DECODER_KWARGS,
    decoder_kwargs=ENCODER_DECODER_KWARGS,
    kl_weight=1e-6,
    reconstruction_weight=1.0,
    warmup_steps=500,
)
INTEGRATE_KWARGS = {"method": "rk45", "steps": 50}

SEED = 42

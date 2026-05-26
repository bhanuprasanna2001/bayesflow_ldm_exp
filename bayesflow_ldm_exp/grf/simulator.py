"""GRF simulator: Gaussian random fields with power-spectrum k^-alpha * exp(2 log_std)."""
import numpy as np
from FyeldGenerator import generate_field

from . import config as C


def _power_spectrum_fn(alpha, scale):
    def spectrum(k):
        return np.power(k, -alpha) * scale ** 2
    return spectrum


def _complex_distribution(rng):
    def distribution(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return distribution


def _sample_one(rng):
    log_std = rng.normal(loc=C.LOG_STD_LOC, scale=C.LOG_STD_SCALE)
    alpha = rng.normal(loc=C.ALPHA_LOC, scale=C.ALPHA_SCALE)
    field = generate_field(
        _complex_distribution(rng),
        _power_spectrum_fn(alpha, np.exp(log_std)),
        shape=C.FIELD_SHAPE[:2],
        unit_length=1.0 / (np.abs(alpha) + 1e-7),
    )
    return log_std, alpha, field.astype("float32")


def _make_split(n, rng):
    fields = np.empty((n,) + C.FIELD_SHAPE, dtype="float32")
    params = np.empty((n, C.PARAM_DIM), dtype="float32")
    for i in range(n):
        log_std, alpha, field = _sample_one(rng)
        fields[i, ..., 0] = field
        params[i, 0] = log_std
        params[i, 1] = alpha

    params_expanded = np.broadcast_to(
        params[:, None, None, :], (n,) + C.FIELD_SHAPE[:2] + (C.PARAM_DIM,)
    ).astype("float32").copy()

    return {
        "field": fields,
        "params": params,
        "params_expanded": params_expanded,
        "log_std": params[:, 0:1],
        "alpha": params[:, 1:2],
    }


def load_grf_splits(
    train_samples=C.TRAIN_SAMPLES,
    val_samples=C.VAL_SAMPLES,
    test_samples=C.TEST_SAMPLES,
    seed=C.SEED,
):
    rng_train = np.random.default_rng(seed + 11)
    rng_val = np.random.default_rng(seed + 23)
    rng_test = np.random.default_rng(seed + 37)
    return {
        "train": _make_split(train_samples, rng_train),
        "validation": _make_split(val_samples, rng_val),
        "test": _make_split(test_samples, rng_test),
    }


def select_diverse_indices(dataset, n=4, seed=C.SEED):
    """Pick n test indices spanning the alpha range — useful for figure grids."""
    alphas = dataset["alpha"].ravel()
    quantiles = np.linspace(0.1, 0.9, n)
    targets = np.quantile(alphas, quantiles)
    rng = np.random.default_rng(seed)
    indices = []
    for t in targets:
        order = np.argsort(np.abs(alphas - t))
        for idx in order:
            if int(idx) not in indices:
                indices.append(int(idx))
                break
    del rng
    return np.array(indices, dtype=int)

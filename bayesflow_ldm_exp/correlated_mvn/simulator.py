"""Linear-Gaussian SBI benchmark with an AR(1)-structured Gaussian prior.

    theta ~ N(0, Sigma_rho),   (Sigma_rho)_{ij} = rho^|i-j|
    y     = A theta + eps,     eps ~ N(0, obs_noise^2 I)

with A a selection of coordinates. The posterior is analytically Gaussian.
"""

import numpy as np


def ar1_covariance(dim, rho):
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must satisfy -1 < rho < 1.")
    idx = np.arange(dim)
    return (rho ** np.abs(idx[:, None] - idx[None, :])).astype(np.float64)


def posterior_covariance(prior_cov, observed_indices, obs_noise):
    if obs_noise <= 0:
        raise ValueError("obs_noise must be positive.")
    observed_indices = np.asarray(observed_indices, dtype=int)
    if observed_indices.ndim != 1:
        raise ValueError("observed_indices must be one-dimensional.")
    if np.unique(observed_indices).size != observed_indices.size:
        raise ValueError("observed_indices must be unique.")
    if observed_indices.min() < 0 or observed_indices.max() >= prior_cov.shape[0]:
        raise ValueError("observed_indices out of range.")

    precision = np.linalg.inv(prior_cov)
    precision[observed_indices, observed_indices] += 1.0 / obs_noise**2
    cov = np.linalg.inv(precision)
    return 0.5 * (cov + cov.T)


def posterior_mean(observables, posterior_cov, observed_indices, obs_noise):
    observables = np.atleast_2d(np.asarray(observables, dtype=np.float64))
    rhs = np.zeros((observables.shape[0], posterior_cov.shape[0]), dtype=np.float64)
    rhs[:, np.asarray(observed_indices, dtype=int)] = observables / obs_noise**2
    return rhs @ posterior_cov


def build_problem(dim, rho, observed_indices, obs_noise):
    prior_cov = ar1_covariance(dim, rho)
    post_cov = posterior_covariance(prior_cov, observed_indices, obs_noise)
    return {
        "dim": int(dim),
        "rho": float(rho),
        "obs_noise": float(obs_noise),
        "observed_indices": np.asarray(observed_indices, dtype=int),
        "prior_cov": prior_cov,
        "prior_chol": np.linalg.cholesky(prior_cov),
        "posterior_cov": post_cov,
        "posterior_chol": np.linalg.cholesky(post_cov),
    }


def simulate(problem, batch_size, rng):
    dim = problem["dim"]
    obs_idx = problem["observed_indices"]
    sigma = problem["obs_noise"]

    eps = rng.standard_normal((int(batch_size), dim))
    parameters = eps @ problem["prior_chol"].T
    observables = parameters[:, obs_idx] + sigma * rng.standard_normal((batch_size, obs_idx.size))

    return {
        "parameters": parameters.astype("float32"),
        "observables": observables.astype("float32"),
    }


def sample_posterior(problem, observables, num_samples, rng):
    mean = posterior_mean(observables, problem["posterior_cov"], problem["observed_indices"], problem["obs_noise"])
    eps = rng.standard_normal((mean.shape[0], int(num_samples), problem["dim"]))
    samples = mean[:, None, :] + eps @ problem["posterior_chol"].T
    return samples.astype("float32")


def exact_posterior(problem, observables):
    mean = posterior_mean(observables, problem["posterior_cov"], problem["observed_indices"], problem["obs_noise"])
    return {
        "mean": mean.astype("float32"),
        "covariance": problem["posterior_cov"].astype("float32"),
    }

# BayesFlow Latent Diffusion Experiments

This repository contains thesis experiments comparing direct diffusion models with latent diffusion models for simulation-based posterior inference in BayesFlow. Each experiment creates or downloads its data, trains the selected models, and writes figures plus `metrics.json` to its own `results/` folder.

## Structure

```text
.
|-- README.md
|-- requirements.txt
|-- .gitignore
`-- bayesflow_ldm_exp/
	|-- spatial.py
	|-- two_moons/
	|-- correlated_mvn/
	|-- grf/
	`-- bayesian_denoising/
```

## Experiments

- `bayesflow_ldm_exp/two_moons/`: toy posterior benchmark with direct diffusion and latent diffusion.
- `bayesflow_ldm_exp/correlated_mvn/`: 32-dimensional Gaussian posterior with exact ground truth for different correlations.
- `bayesflow_ldm_exp/grf/`: Gaussian random fields conditioned on spatial parameters, using direct U-Net diffusion and spatial LDMs.
- `bayesflow_ldm_exp/bayesian_denoising/`: Fashion-MNIST denoising after Poisson noise and Gaussian blur.
- `bayesflow_ldm_exp/spatial.py`: shared spatial U-Net, encoder, and decoder layers.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs BayesFlow from the `ldm-dev` branch used for these experiments.

## Run

From the repository root:

```bash
python -m bayesflow_ldm_exp.two_moons.train
python -m bayesflow_ldm_exp.correlated_mvn.train
python -m bayesflow_ldm_exp.grf.train
python -m bayesflow_ldm_exp.bayesian_denoising.train
```

For quick spatial smoke runs:

```bash
python -m bayesflow_ldm_exp.grf.train --epochs 2 --train-samples 128 --val-samples 32 --test-samples 32 --models spatial_direct
python -m bayesflow_ldm_exp.bayesian_denoising.train --epochs 2 --train-samples 128 --val-samples 32 --test-samples 32 --models spatial_direct
```

The default values in each `config.py` are the full experiment settings.
# Exoplanet Detection via PINNs + Bayesian Inference

This repository implements a robust exoplanet detection pipeline that combines:
- Physics-Informed Neural Networks (PINNs)
- Bayesian uncertainty estimation
- Dynamic instrument calibration

The system is designed to handle real astronomical data with noise, gaps,
and non-stationary systematics.

## Supported Data
- Kepler light curves
- TESS light curves
- Radial velocity datasets (optional)

## Key Features
- Keplerian orbital constraints via physics loss
- Time-varying calibration modeling
- Posterior distributions for planet parameters
- Injection–recovery validation
- Fully configurable via YAML

## Run Training
```bash
python scripts/run_training.py --config configs/experiment.yaml
```

## Run Evaluation

```bash
python scripts/run_evaluation.py --config configs/experiment.yaml
```

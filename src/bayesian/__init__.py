"""
Bayesian module for exoplanet detection pipeline.

This package provides Bayesian inference tools for uncertainty estimation in exoplanet detection models.
Includes variational inference, MCMC sampling, and uncertainty quantification utilities.

Astrophysical meaning:
- Enables robust posterior estimation of planet parameters
- Supports epistemic and aleatoric uncertainty decomposition

Assumptions:
- Models are compatible with torch/numpyro/pymc
- Data is preprocessed and structured

Limitations:
- Large models may require substantial compute for MCMC
- Assumes input data is cleaned and normalized
"""
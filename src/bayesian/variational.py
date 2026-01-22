"""
Variational inference for Bayesian uncertainty estimation.

This module implements variational inference to estimate both:
- Epistemic uncertainty (model uncertainty)
- Aleatoric uncertainty (data uncertainty)

Scientific Context:
    Bayesian methods provide principled uncertainty quantification, which is
    critical for exoplanet detection where false positives are costly.
    Variational inference approximates the true posterior distribution
    over model parameters, enabling uncertainty-aware predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np


class VariationalLayer(nn.Module):
    """
    Variational layer that learns posterior distributions over weights.
    
    Instead of point estimates, learns mean and log-variance of weight
    distributions. During inference, samples weights from these distributions.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0
    ):
        """
        Initialize variational layer.
        
        Parameters
        ----------
        in_features : int
            Input feature dimension.
        out_features : int
            Output feature dimension.
        prior_std : float
            Standard deviation of prior distribution.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Learnable parameters for weight distribution
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # Bias (point estimate for simplicity)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Forward pass with weight sampling.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample : bool
            Whether to sample weights (True) or use mean (False).
        
        Returns
        -------
        Tuple[torch.Tensor, float]
            (output, kl_divergence)
        """
        if sample:
            # Sample weights from learned distribution
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_std * weight_epsilon
        else:
            # Use mean weights
            weight = self.weight_mu
        
        # Forward pass
        output = F.linear(x, weight, self.bias)
        
        # Compute KL divergence (for loss)
        kl_div = self._kl_divergence(weight, self.weight_mu, self.weight_logvar)
        
        return output, kl_div
    
    def _kl_divergence(
        self,
        weight: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> float:
        """
        Compute KL divergence between posterior and prior.
        
        KL(q(w) || p(w)) where:
        - q(w) = N(mu, exp(logvar))
        - p(w) = N(0, prior_std^2)
        """
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(
            torch.log(self.prior_std ** 2) - logvar +
            (var + (mu - 0) ** 2) / (self.prior_std ** 2) - 1
        )
        return kl


class BayesianParameterHead(nn.Module):
    """
    Bayesian version of parameter head with uncertainty estimation.
    
    Predicts both mean and uncertainty for each transit parameter.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.1,
        use_variational: bool = True
    ):
        """
        Initialize Bayesian parameter head.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dims : list
            Hidden layer dimensions.
        dropout : float
            Dropout probability.
        use_variational : bool
            Whether to use variational layers.
        """
        super().__init__()
        self.use_variational = use_variational
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if use_variational:
                # Use standard layers for hidden (variational only in final layer)
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Output heads: mean and log-variance for each parameter
        if use_variational:
            # Use variational layers for final predictions
            self.period_head = VariationalLayer(in_dim, 1)
            self.t0_head = VariationalLayer(in_dim, 1)
            self.rp_rs_head = VariationalLayer(in_dim, 1)
            self.a_rs_head = VariationalLayer(in_dim, 1)
            self.b_head = VariationalLayer(in_dim, 1)
            self.u1_head = VariationalLayer(in_dim, 1)
            self.u2_head = VariationalLayer(in_dim, 1)
        else:
            # Standard layers with separate mean and logvar heads
            self.period_mu = nn.Linear(in_dim, 1)
            self.period_logvar = nn.Linear(in_dim, 1)
            self.t0_mu = nn.Linear(in_dim, 1)
            self.t0_logvar = nn.Linear(in_dim, 1)
            self.rp_rs_mu = nn.Linear(in_dim, 1)
            self.rp_rs_logvar = nn.Linear(in_dim, 1)
            self.a_rs_mu = nn.Linear(in_dim, 1)
            self.a_rs_logvar = nn.Linear(in_dim, 1)
            self.b_mu = nn.Linear(in_dim, 1)
            self.b_logvar = nn.Linear(in_dim, 1)
            self.u1_mu = nn.Linear(in_dim, 1)
            self.u1_logvar = nn.Linear(in_dim, 1)
            self.u2_mu = nn.Linear(in_dim, 1)
            self.u2_logvar = nn.Linear(in_dim, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features.
        sample : bool
            Whether to sample from distributions.
        
        Returns
        -------
        Tuple[Dict, Dict, float]
            (parameters_mean, parameters_std, total_kl_div)
        """
        features = self.feature_layers(x)
        
        total_kl = 0.0
        
        if self.use_variational:
            # Variational layers
            period_out, kl = self.period_head(features, sample)
            period_mu = F.softplus(period_out) + 0.5
            period_std = torch.ones_like(period_mu) * 0.1  # Simplified
            total_kl += kl
            
            t0_out, kl = self.t0_head(features, sample)
            t0_mu = t0_out
            t0_std = torch.ones_like(t0_mu) * 0.1
            total_kl += kl
            
            rp_rs_out, kl = self.rp_rs_head(features, sample)
            rp_rs_mu = F.sigmoid(rp_rs_out) * 0.2
            rp_rs_std = torch.ones_like(rp_rs_mu) * 0.01
            total_kl += kl
            
            a_rs_out, kl = self.a_rs_head(features, sample)
            a_rs_mu = F.softplus(a_rs_out) + 1.0
            a_rs_std = torch.ones_like(a_rs_mu) * 0.1
            total_kl += kl
            
            b_out, kl = self.b_head(features, sample)
            b_mu = F.sigmoid(b_out)
            b_std = torch.ones_like(b_mu) * 0.05
            total_kl += kl
            
            u1_out, kl = self.u1_head(features, sample)
            u1_mu = F.sigmoid(u1_out)
            u1_std = torch.ones_like(u1_mu) * 0.05
            total_kl += kl
            
            u2_out, kl = self.u2_head(features, sample)
            u2_mu = F.sigmoid(u2_out)
            u2_std = torch.ones_like(u2_mu) * 0.05
            total_kl += kl
        else:
            # Standard layers with explicit uncertainty
            period_mu = F.softplus(self.period_mu(features)) + 0.5
            period_logvar = self.period_logvar(features)
            period_std = torch.exp(0.5 * period_logvar)
            
            t0_mu = self.t0_mu(features)
            t0_logvar = self.t0_logvar(features)
            t0_std = torch.exp(0.5 * t0_logvar)
            
            rp_rs_mu = F.sigmoid(self.rp_rs_mu(features)) * 0.2
            rp_rs_logvar = self.rp_rs_logvar(features)
            rp_rs_std = torch.exp(0.5 * rp_rs_logvar)
            
            a_rs_mu = F.softplus(self.a_rs_mu(features)) + 1.0
            a_rs_logvar = self.a_rs_logvar(features)
            a_rs_std = torch.exp(0.5 * a_rs_logvar)
            
            b_mu = F.sigmoid(self.b_mu(features))
            b_logvar = self.b_logvar(features)
            b_std = torch.exp(0.5 * b_logvar)
            
            u1_mu = F.sigmoid(self.u1_mu(features))
            u1_logvar = self.u1_logvar(features)
            u1_std = torch.exp(0.5 * u1_logvar)
            
            u2_mu = F.sigmoid(self.u2_mu(features))
            u2_logvar = self.u2_logvar(features)
            u2_std = torch.exp(0.5 * u2_logvar)
        
        parameters_mean = {
            'period': period_mu.squeeze(-1),
            't0': t0_mu.squeeze(-1),
            'rp_rs': rp_rs_mu.squeeze(-1),
            'a_rs': a_rs_mu.squeeze(-1),
            'b': b_mu.squeeze(-1),
            'u1': u1_mu.squeeze(-1),
            'u2': u2_mu.squeeze(-1)
        }
        
        parameters_std = {
            'period': period_std.squeeze(-1),
            't0': t0_std.squeeze(-1),
            'rp_rs': rp_rs_std.squeeze(-1),
            'a_rs': a_rs_std.squeeze(-1),
            'b': b_std.squeeze(-1),
            'u1': u1_std.squeeze(-1),
            'u2': u2_std.squeeze(-1)
        }
        
        return parameters_mean, parameters_std, total_kl
    
    def sample_parameters(
        self,
        parameters_mean: Dict[str, torch.Tensor],
        parameters_std: Dict[str, torch.Tensor],
        n_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Sample parameters from predicted distributions.
        
        Parameters
        ----------
        parameters_mean : Dict[str, torch.Tensor]
            Mean predictions.
        parameters_std : Dict[str, torch.Tensor]
            Standard deviation predictions.
        n_samples : int
            Number of samples to draw.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Sampled parameters.
        """
        samples = {}
        for key in parameters_mean.keys():
            mu = parameters_mean[key]
            std = parameters_std[key]
            
            # Sample from normal distribution
            if n_samples > 1:
                epsilon = torch.randn(n_samples, *mu.shape, device=mu.device)
                samples[key] = mu.unsqueeze(0) + std.unsqueeze(0) * epsilon
            else:
                epsilon = torch.randn_like(mu)
                samples[key] = mu + std * epsilon
        
        return samples


def compute_epistemic_uncertainty(
    model: nn.Module,
    time: torch.Tensor,
    flux: torch.Tensor,
    flux_err: Optional[torch.Tensor],
    n_samples: int = 100
) -> Dict[str, torch.Tensor]:
    """
    Compute epistemic uncertainty via Monte Carlo sampling.
    
    Parameters
    ----------
    model : nn.Module
        Bayesian model.
    time : torch.Tensor
        Observation times.
    flux : torch.Tensor
        Observed flux.
    flux_err : torch.Tensor, optional
        Flux uncertainties.
    n_samples : int
        Number of Monte Carlo samples.
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with mean, std, and samples for each parameter.
    """
    model.eval()
    
    all_samples = {key: [] for key in ['period', 't0', 'rp_rs', 'a_rs', 'b', 'u1', 'u2']}
    
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(time, flux, flux_err)
            
            if 'parameters_std' in output:
                # Sample from predicted distributions
                samples = model.param_head.sample_parameters(
                    output['parameters'],
                    output['parameters_std'],
                    n_samples=1
                )
            else:
                samples = output['parameters']
            
            for key in all_samples.keys():
                all_samples[key].append(samples[key])
    
    # Compute statistics
    results = {}
    for key in all_samples.keys():
        stacked = torch.stack(all_samples[key], dim=0)
        results[f'{key}_mean'] = torch.mean(stacked, dim=0)
        results[f'{key}_std'] = torch.std(stacked, dim=0)
        results[f'{key}_samples'] = stacked
    
    return results

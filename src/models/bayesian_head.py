"""
Bayesian head for PINN with uncertainty estimation.

This module provides a Bayesian version of the PINN that can be used
as a drop-in replacement for uncertainty-aware predictions.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .pinn import TimeSeriesEncoder, AttentionPooling
from ..bayesian.variational import BayesianParameterHead


class BayesianPINN(nn.Module):
    """
    Bayesian version of PINN with uncertainty estimation.
    
    Uses variational inference to estimate both epistemic and aleatoric
    uncertainty in transit parameter predictions.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        encoder_dims: list = [64, 128, 256],
        encoder_kernels: list = [5, 5, 5],
        param_head_dims: list = [256, 128, 64],
        dropout: float = 0.1,
        use_variational: bool = True
    ):
        """
        Initialize Bayesian PINN.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        encoder_dims : list
            Encoder hidden dimensions.
        encoder_kernels : list
            Encoder kernel sizes.
        param_head_dims : list
            Parameter head hidden dimensions.
        dropout : float
            Dropout probability.
        use_variational : bool
            Whether to use variational layers.
        """
        super().__init__()
        
        self.encoder = TimeSeriesEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_dims,
            kernel_sizes=encoder_kernels,
            dropout=dropout
        )
        
        self.attention = AttentionPooling(self.encoder.output_dim)
        
        self.param_head = BayesianParameterHead(
            input_dim=self.encoder.output_dim,
            hidden_dims=param_head_dims,
            dropout=dropout,
            use_variational=use_variational
        )
    
    def forward(
        self,
        time: torch.Tensor,
        flux: torch.Tensor,
        flux_err: Optional[torch.Tensor] = None,
        sample: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass with uncertainty estimation.
        
        Parameters
        ----------
        time : torch.Tensor
            Time array.
        flux : torch.Tensor
            Flux array.
        flux_err : torch.Tensor, optional
            Flux errors.
        sample : bool
            Whether to sample from distributions.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with parameters, uncertainties, and KL divergence.
        """
        batch_size, time_steps = flux.shape
        
        # Prepare input
        if flux_err is not None:
            x = torch.stack([time, flux, flux_err], dim=1)
        else:
            x = torch.stack([time, flux, torch.zeros_like(flux)], dim=1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Pool with attention
        pooled, attn_weights = self.attention(encoded)
        
        # Predict parameters with uncertainty
        parameters_mean, parameters_std, kl_div = self.param_head(pooled, sample=sample)
        
        return {
            'parameters': parameters_mean,
            'parameters_std': parameters_std,
            'kl_divergence': kl_div,
            'features': pooled,
            'attention_weights': attn_weights
        }

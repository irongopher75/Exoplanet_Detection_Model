"""
Calibration network for time-varying instrument systematics.

This module implements a neural network that learns to correct for:
- Time-varying instrument calibration drifts
- Temperature-dependent effects
- Non-stationary systematics in light curves

Scientific Context:
    Real astronomical instruments have time-varying systematics:
    - Detector temperature changes
    - Pointing drifts
    - Flat-field variations
    - Background changes
    
    These must be corrected before transit detection to avoid false positives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CalibrationNetwork(nn.Module):
    """
    Neural network for learning time-varying calibration corrections.
    
    Takes time, flux, and flux_err as inputs and outputs a correction
    factor that can be applied to remove systematic effects.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: list = [32, 64, 32],
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize calibration network.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension (typically 3: time, flux, flux_err).
        hidden_dims : list
            Hidden layer dimensions.
        output_dim : int
            Output dimension (typically 1 for correction factor).
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Output layer (no activation - can be positive or negative correction)
        self.output_layer = nn.Linear(in_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).
            Typically: [time, flux, flux_err] or parameter vector.
        
        Returns
        -------
        torch.Tensor
            Calibration correction of shape (batch, output_dim).
        """
        features = self.feature_layers(x)
        correction = self.output_layer(features)
        
        return correction


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for uncertainty calibration.
    
    Learns a temperature parameter T such that calibrated uncertainty = uncertainty / T.
    Used to calibrate overconfident or underconfident uncertainty estimates.
    """
    
    def __init__(self):
        """Initialize temperature scaling."""
        super().__init__()
        # Temperature parameter (learned, initialized to 1.0)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Parameters
        ----------
        logits : torch.Tensor
            Model logits.
        
        Returns
        -------
        torch.Tensor
            Scaled logits.
        """
        return logits / self.temperature


class ParameterCalibrationNetwork(nn.Module):
    """
    Calibration network for transit parameters.
    
    Learns corrections to predicted transit parameters to account for
    systematic biases in the model predictions.
    """
    
    def __init__(
        self,
        param_dim: int = 7,
        hidden_dims: list = [64, 128, 64],
        dropout: float = 0.1
    ):
        """
        Initialize parameter calibration network.
        
        Parameters
        ----------
        param_dim : int
            Number of transit parameters (period, t0, rp_rs, a_rs, b, u1, u2).
        hidden_dims : list
            Hidden layer dimensions.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        layers = []
        in_dim = param_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Output: additive correction for each parameter
        self.correction_layer = nn.Linear(in_dim, param_dim)
        
        # Learnable scaling factors for each parameter
        self.scale_factors = nn.Parameter(torch.ones(param_dim))
    
    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        Apply calibration to parameters.
        
        Parameters
        ----------
        parameters : torch.Tensor
            Predicted parameters of shape (batch, param_dim).
        
        Returns
        -------
        torch.Tensor
            Calibrated parameters of shape (batch, param_dim).
        """
        features = self.feature_layers(parameters)
        correction = self.correction_layer(features)
        
        # Apply additive correction and scaling
        calibrated = parameters + correction * self.scale_factors.unsqueeze(0)
        
        return calibrated

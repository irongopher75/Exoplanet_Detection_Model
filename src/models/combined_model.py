"""
Combined model integrating PINN with calibration and Bayesian components.

This module provides a unified model that combines:
- PINN for transit detection
- Calibration network for systematics
- Bayesian head for uncertainty
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .pinn import PINN
from .calibration_net import CalibrationNetwork, ParameterCalibrationNetwork
from .bayesian_head import BayesianPINN


class CombinedExoplanetModel(nn.Module):
    """
    Combined model integrating all components.
    
    Architecture:
        1. PINN: Detects and predicts transit parameters
        2. Calibration: Corrects for systematics
        3. Bayesian: Provides uncertainty estimates
    """
    
    def __init__(
        self,
        use_bayesian: bool = True,
        use_calibration: bool = True,
        input_dim: int = 3,
        encoder_dims: list = [64, 128, 256],
        encoder_kernels: list = [5, 5, 5],
        param_head_dims: list = [256, 128, 64],
        dropout: float = 0.1
    ):
        """
        Initialize combined model.
        
        Parameters
        ----------
        use_bayesian : bool
            Whether to use Bayesian head for uncertainty.
        use_calibration : bool
            Whether to use calibration network.
        input_dim : int
            Input feature dimension.
        encoder_dims : list
            Encoder hidden dimensions.
        encoder_kernels : list
            Encoder kernel sizes.
        param_head_dims : list
            Parameter head dimensions.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        self.use_bayesian = use_bayesian
        self.use_calibration = use_calibration
        
        # Base PINN
        if use_bayesian:
            self.pinn = BayesianPINN(
                input_dim=input_dim,
                encoder_dims=encoder_dims,
                encoder_kernels=encoder_kernels,
                param_head_dims=param_head_dims,
                dropout=dropout
            )
        else:
            self.pinn = PINN(
                input_dim=input_dim,
                encoder_dims=encoder_dims,
                encoder_kernels=encoder_kernels,
                param_head_dims=param_head_dims,
                dropout=dropout
            )
        
        # Calibration network for parameters
        if use_calibration:
            self.calibration = ParameterCalibrationNetwork(
                param_dim=7,  # period, t0, rp_rs, a_rs, b, u1, u2
                hidden_dims=[64, 128, 64],
                dropout=dropout
            )
    
    def forward(
        self,
        time: torch.Tensor,
        flux: torch.Tensor,
        flux_err: Optional[torch.Tensor] = None,
        sample: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass through combined model.
        
        Parameters
        ----------
        time : torch.Tensor
            Time array.
        flux : torch.Tensor
            Flux array.
        flux_err : torch.Tensor, optional
            Flux uncertainties.
        sample : bool
            Whether to sample from distributions (Bayesian mode).
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with predictions, uncertainties, and metadata.
        """
        # Get PINN predictions
        if self.use_bayesian:
            pinn_output = self.pinn(time, flux, flux_err, sample=sample)
            parameters = pinn_output['parameters']
            parameters_std = pinn_output.get('parameters_std')
            kl_div = pinn_output.get('kl_divergence', torch.tensor(0.0))
        else:
            pinn_output = self.pinn(time, flux, flux_err)
            parameters = pinn_output['parameters']
            parameters_std = None
            kl_div = torch.tensor(0.0)
        
        # Apply calibration if enabled
        if self.use_calibration:
            # Stack parameters
            param_vector = torch.stack([
                parameters['period'],
                parameters['t0'],
                parameters['rp_rs'],
                parameters['a_rs'],
                parameters['b'],
                parameters['u1'],
                parameters['u2']
            ], dim=1)
            
            # Calibrate
            calibrated_vector = self.calibration(param_vector)
            
            # Unstack
            parameters = {
                'period': calibrated_vector[:, 0],
                't0': calibrated_vector[:, 1],
                'rp_rs': calibrated_vector[:, 2],
                'a_rs': calibrated_vector[:, 3],
                'b': calibrated_vector[:, 4],
                'u1': calibrated_vector[:, 5],
                'u2': calibrated_vector[:, 6]
            }
        
        return {
            'parameters': parameters,
            'parameters_std': parameters_std,
            'kl_divergence': kl_div,
            'features': pinn_output.get('features'),
            'attention_weights': pinn_output.get('attention_weights')
        }

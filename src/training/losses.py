"""
Training loss functions for exoplanet detection models.

This module provides loss functions for training PINN and other models.
It integrates with physics-informed losses and provides standard ML losses.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class CombinedLoss(nn.Module):
    """
    Combined loss function for PINN training.
    
    Combines physics-informed loss with standard ML losses.
    """
    
    def __init__(
        self,
        physics_loss_fn: nn.Module,
        weight_physics: float = 1.0,
        weight_reconstruction: float = 1.0
    ):
        """
        Initialize combined loss.
        
        Parameters
        ----------
        physics_loss_fn : nn.Module
            Physics-informed loss function.
        weight_physics : float
            Weight for physics loss.
        weight_reconstruction : float
            Weight for reconstruction loss.
        """
        super().__init__()
        self.physics_loss_fn = physics_loss_fn
        self.weight_physics = weight_physics
        self.weight_reconstruction = weight_reconstruction
    
    def forward(
        self,
        predicted_flux: torch.Tensor,
        target_flux: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        time: torch.Tensor,
        flux_err: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Parameters
        ----------
        predicted_flux : torch.Tensor
            Predicted flux.
        target_flux : torch.Tensor
            Target flux.
        parameters : Dict[str, torch.Tensor]
            Predicted parameters.
        time : torch.Tensor
            Observation times.
        flux_err : torch.Tensor, optional
            Flux uncertainties.
        **kwargs
            Additional arguments for physics loss.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Loss dictionary.
        """
        # Physics-informed loss
        physics_loss_dict = self.physics_loss_fn(
            predicted_flux=predicted_flux,
            target_flux=target_flux,
            parameters=parameters,
            time=time,
            flux_err=flux_err,
            **kwargs
        )
        
        # Total loss
        total_loss = (
            self.weight_physics * physics_loss_dict['total_loss'] +
            self.weight_reconstruction * physics_loss_dict['data_loss']
        )
        
        return {
            'total_loss': total_loss,
            **physics_loss_dict
        }

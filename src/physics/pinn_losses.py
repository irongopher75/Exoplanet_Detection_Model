"""
Physics-informed loss functions for PINN training.

This module implements loss functions that enforce physical constraints
on the predicted transit parameters. The key principle:

    Physics goes in the LOSS, not the architecture.

Loss Components:
    1. Data likelihood: Reconstruction error (flux prediction)
    2. Physics loss: Kepler's laws, transit duration constraints
    3. Regularization: Parameter bounds, smoothness

References:
    - Mandel & Agol (2002) for transit model
    - Kepler's laws for orbital mechanics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Callable

# Import transit model (will need PyTorch version or conversion)
try:
    from .transit_model import mandel_agol_transit, compute_transit_duration, keplers_third_law
except ImportError:
    pass


def mandel_agol_transit_torch(
    time: torch.Tensor,
    period: torch.Tensor,
    t0: torch.Tensor,
    rp_rs: torch.Tensor,
    a_rs: torch.Tensor,
    b: torch.Tensor,
    u1: torch.Tensor = torch.tensor(0.3),
    u2: torch.Tensor = torch.tensor(0.3),
    eccentricity: torch.Tensor = torch.tensor(0.0),
    omega: torch.Tensor = torch.tensor(90.0)
) -> torch.Tensor:
    """
    PyTorch implementation of Mandel & Agol transit model.
    
    This is a simplified version for gradient computation.
    For production, consider using a more complete implementation.
    
    Parameters
    ----------
    time : torch.Tensor
        Observation times (days).
    period : torch.Tensor
        Orbital period (days), shape (batch,).
    t0 : torch.Tensor
        Time of mid-transit (days), shape (batch,).
    rp_rs : torch.Tensor
        Planet-to-star radius ratio, shape (batch,).
    a_rs : torch.Tensor
        Semi-major axis in stellar radii, shape (batch,).
    b : torch.Tensor
        Impact parameter, shape (batch,).
    u1, u2 : torch.Tensor
        Limb darkening coefficients.
    eccentricity : torch.Tensor
        Orbital eccentricity.
    omega : torch.Tensor
        Argument of periastron (degrees).
    
    Returns
    -------
    torch.Tensor
        Relative flux (1.0 = no transit, <1.0 = transit).
    """
    batch_size = period.shape[0]
    
    # Handle time tensor shape: could be (time_steps,) or (batch, time_steps)
    if time.dim() == 1:
        # Single time array for all batches: expand to (1, time_steps)
        time = time.unsqueeze(0)  # (1, time_steps)
        n_times = time.shape[-1]
        # Expand to match batch size
        time = time.expand(batch_size, -1)  # (batch, time_steps)
    else:
        # Already batched: (batch, time_steps)
        n_times = time.shape[-1]
    
    # Expand parameters to match time dimension
    period = period.unsqueeze(-1)  # (batch, 1)
    t0 = t0.unsqueeze(-1)  # (batch, 1)
    rp_rs = rp_rs.unsqueeze(-1)  # (batch, 1)
    a_rs = a_rs.unsqueeze(-1)  # (batch, 1)
    b = b.unsqueeze(-1)  # (batch, 1)
    
    # Phase calculation: time is (batch, time_steps), t0 is (batch, 1)
    phase = ((time - t0) / period) % 1.0  # (batch, time_steps)
    phase = torch.where(phase > 0.5, phase - 1.0, phase)  # Center around 0
    
    # Orbital separation (simplified for circular orbits)
    if torch.any(eccentricity > 0):
        # Eccentric orbit (simplified)
        true_anomaly = 2 * torch.pi * phase
        r_sep = a_rs * (1.0 - eccentricity ** 2) / (1.0 + eccentricity * torch.cos(true_anomaly))
    else:
        r_sep = a_rs
    
    # Projected separation
    z = torch.sqrt((r_sep * torch.sin(2 * torch.pi * phase)) ** 2 + b ** 2)
    
    # Compute flux (simplified transit model)
    flux = torch.ones_like(z)  # (batch, time_steps)
    
    # Transit occurs when z < 1 + rp_rs
    in_transit = z < (1.0 + rp_rs)  # (batch, time_steps)
    
    # Simplified depth calculation with limb darkening
    # Ensure u1 and u2 are broadcastable
    if u1.dim() == 0:
        u1 = u1.unsqueeze(-1).unsqueeze(-1)  # (1, 1)
    elif u1.dim() == 1:
        u1 = u1.unsqueeze(-1)  # (batch, 1)
    if u2.dim() == 0:
        u2 = u2.unsqueeze(-1).unsqueeze(-1)  # (1, 1)
    elif u2.dim() == 1:
        u2 = u2.unsqueeze(-1)  # (batch, 1)
    
    depth = rp_rs ** 2 * (1.0 - u1 / 3.0 - u2 / 6.0)  # (batch, 1)
    
    # Apply depth where in transit
    flux = torch.where(in_transit, 1.0 - depth, flux)
    
    return flux  # (batch, time_steps)


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function for PINN training.
    
    Combines:
    1. Data reconstruction loss
    2. Physics constraint losses
    3. Regularization terms
    """
    
    def __init__(
        self,
        data_weight: float = 1.0,
        physics_weight: float = 0.1,
        kepler_weight: float = 0.1,
        duration_weight: float = 0.05,
        reg_weight: float = 0.01
    ):
        """
        Initialize loss function.
        
        Parameters
        ----------
        data_weight : float
            Weight for data reconstruction loss.
        physics_weight : float
            Weight for general physics constraints.
        kepler_weight : float
            Weight for Kepler's third law constraint.
        duration_weight : float
            Weight for transit duration constraint.
        reg_weight : float
            Weight for parameter regularization.
        """
        super().__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.kepler_weight = kepler_weight
        self.duration_weight = duration_weight
        self.reg_weight = reg_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        predicted_flux: torch.Tensor,
        target_flux: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        time: torch.Tensor,
        flux_err: Optional[torch.Tensor] = None,
        stellar_mass: Optional[torch.Tensor] = None,
        stellar_radius: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss.
        
        Parameters
        ----------
        predicted_flux : torch.Tensor
            Predicted flux from physics model.
        target_flux : torch.Tensor
            Observed flux.
        parameters : Dict[str, torch.Tensor]
            Predicted transit parameters.
        time : torch.Tensor
            Observation times.
        flux_err : torch.Tensor, optional
            Flux uncertainties (for weighted loss).
        stellar_mass : torch.Tensor, optional
            Stellar masses (for Kepler's law).
        stellar_radius : torch.Tensor, optional
            Stellar radii (for Kepler's law).
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss components and total loss.
        """
        # 1. Data reconstruction loss
        if flux_err is not None:
            # Weighted MSE
            weights = 1.0 / (flux_err ** 2 + 1e-8)
            data_loss = torch.mean(weights * (predicted_flux - target_flux) ** 2)
        else:
            data_loss = self.mse_loss(predicted_flux, target_flux)
        
        # 2. Physics constraint losses
        physics_losses = {}
        
        # 2a. Parameter bounds (soft constraints)
        param_loss = 0.0
        # period > 0.5: penalize if period <= 0.5
        param_loss += torch.mean(torch.clamp(0.5 - parameters['period'], min=0) ** 2)
        # rp_rs < 0.2: penalize if rp_rs >= 0.2
        param_loss += torch.mean(torch.clamp(parameters['rp_rs'] - 0.2, min=0) ** 2)
        # a_rs > 1.0: penalize if a_rs <= 1.0
        param_loss += torch.mean(torch.clamp(1.0 - parameters['a_rs'], min=0) ** 2)
        # b < 1.0: penalize if b >= 1.0
        param_loss += torch.mean(torch.clamp(parameters['b'] - 1.0, min=0) ** 2)
        physics_losses['parameter_bounds'] = param_loss
        
        # 2b. Kepler's third law (if stellar parameters available)
        kepler_loss = torch.tensor(0.0, device=predicted_flux.device)
        if stellar_mass is not None and stellar_radius is not None:
            # Compute expected a_rs from period using Kepler's law
            # a^3 / P^2 = G * M_star / (4 * π^2)
            # Convert to a_rs = a / R_star
            
            # Constants (in appropriate units)
            G = 6.67430e-11  # m^3 kg^-1 s^-2
            M_sun = 1.989e30  # kg
            R_sun = 6.957e8  # m
            day = 86400  # seconds
            AU = 1.496e11  # m
            
            # Convert to SI
            P_si = parameters['period'] * day  # seconds
            M_si = stellar_mass * M_sun  # kg
            R_si = stellar_radius * R_sun  # m
            
            # Compute expected a (in meters)
            # Note: Using torch.pi for consistency with PyTorch tensors
            pi_tensor = torch.tensor(np.pi, device=predicted_flux.device, dtype=predicted_flux.dtype)
            a_si = (G * M_si * P_si ** 2 / (4 * pi_tensor ** 2)) ** (1.0 / 3.0)
            
            # Convert to a_rs
            a_rs_expected = a_si / R_si
            
            # Loss: predicted a_rs should match expected
            kepler_loss = self.mse_loss(parameters['a_rs'], a_rs_expected)
        physics_losses['keplers_law'] = kepler_loss
        
        # 2c. Transit duration constraint
        # Duration should be consistent with period, a_rs, rp_rs, b
        # T14 ≈ (P / π) * arcsin(sqrt((1 + Rp/Rs)^2 - b^2) / (a/Rs))
        duration_loss = torch.tensor(0.0, device=predicted_flux.device)
        if len(time) > 1:
            # Simplified: check that transit duration is reasonable
            # This is a soft constraint
            time_span = time.max() - time.min()
            min_duration = time_span / 100.0  # At least 1% of observation span
            max_duration = time_span / 2.0  # At most 50% of observation span
            
            # Approximate duration from parameters
            arg = torch.sqrt((1.0 + parameters['rp_rs']) ** 2 - parameters['b'] ** 2) / parameters['a_rs']
            arg = torch.clamp(arg, max=0.99)  # Avoid arcsin(>1)
            duration_approx = (parameters['period'] / torch.pi) * torch.arcsin(arg)
            
            # Penalize durations outside reasonable range
            duration_loss = torch.mean(
                torch.clamp(min_duration - duration_approx, min=0) ** 2 +
                torch.clamp(duration_approx - max_duration, min=0) ** 2
            )
        physics_losses['duration'] = duration_loss
        
        # 3. Regularization
        reg_loss = 0.0
        for param_name, param_value in parameters.items():
            # L2 regularization on parameters
            reg_loss += torch.mean(param_value ** 2)
        physics_losses['regularization'] = reg_loss
        
        # Total loss
        total_loss = (
            self.data_weight * data_loss +
            self.physics_weight * physics_losses['parameter_bounds'] +
            self.kepler_weight * physics_losses['keplers_law'] +
            self.duration_weight * physics_losses['duration'] +
            self.reg_weight * physics_losses['regularization']
        )
        
        return {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'physics_losses': physics_losses
        }


def compute_pinn_loss(
    model: nn.Module,
    time: torch.Tensor,
    flux: torch.Tensor,
    flux_err: Optional[torch.Tensor],
    parameters: Dict[str, torch.Tensor],
    loss_fn: PhysicsInformedLoss,
    stellar_mass: Optional[torch.Tensor] = None,
    stellar_radius: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute complete PINN loss.
    
    This function:
    1. Uses predicted parameters to generate physics model
    2. Computes reconstruction loss
    3. Adds physics constraints
    
    Parameters
    ----------
    model : nn.Module
        PINN model (not used directly, but kept for interface).
    time : torch.Tensor
        Observation times.
    flux : torch.Tensor
        Observed flux.
    flux_err : torch.Tensor, optional
        Flux uncertainties.
    parameters : Dict[str, torch.Tensor]
        Predicted transit parameters.
    loss_fn : PhysicsInformedLoss
        Loss function.
    stellar_mass : torch.Tensor, optional
        Stellar masses.
    stellar_radius : torch.Tensor, optional
        Stellar radii.
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Loss dictionary.
    """
    # Generate physics model from parameters
    predicted_flux = mandel_agol_transit_torch(
        time=time,
        period=parameters['period'],
        t0=parameters['t0'],
        rp_rs=parameters['rp_rs'],
        a_rs=parameters['a_rs'],
        b=parameters['b'],
        u1=parameters.get('u1', torch.tensor(0.3)),
        u2=parameters.get('u2', torch.tensor(0.3))
    )
    
    # Compute loss
    loss_dict = loss_fn(
        predicted_flux=predicted_flux,
        target_flux=flux,
        parameters=parameters,
        time=time,
        flux_err=flux_err,
        stellar_mass=stellar_mass,
        stellar_radius=stellar_radius
    )
    
    return loss_dict

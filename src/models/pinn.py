"""
Physics-Informed Neural Network (PINN) for exoplanet transit detection.

This module implements a PINN that learns to detect and characterize exoplanet
transits while enforcing physical constraints through the loss function.

Architecture:
    - Input: Time series (time, flux, flux_err)
    - Encoder: Extracts features from light curve
    - Physics head: Predicts transit parameters (period, t0, rp_rs, etc.)
    - Reconstruction: Reconstructs light curve using physics model

Key Principle:
    Physics goes in the LOSS, not the architecture.
    The network learns parameters, physics enforces constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Tuple
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block for 1D convolutions."""
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Identity shortcut
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.conv1(x))) # SiLU is Swish
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.silu(out)
        return out


class TimeSeriesEncoder(nn.Module):
    """
    Encoder network for time series light curves.
    
    Uses 1D ResNet blocks to extract features from flux time series.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # time, flux, flux_err
        hidden_dims: list = [64, 128, 256],
        kernel_sizes: list = [5, 5, 5],
        dropout: float = 0.1
    ):
        """
        Initialize encoder.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dims : list
            Hidden layer dimensions.
        kernel_sizes : list
            Convolution kernel sizes.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            layers.append(
                ResidualBlock(in_dim, hidden_dim, kernel_size, dropout)
            )
            in_dim = hidden_dim
        
        self.resnet_layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, features, time_steps).
        
        Returns
        -------
        torch.Tensor
            Encoded features of shape (batch, hidden_dim, time_steps).
        """
        return self.resnet_layers(x)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate time series features.
    
    Learns which time steps are most important for transit detection.
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize attention pooling.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weights.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, features, time_steps).
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (pooled_features, attention_weights)
        """
        # x: (batch, features, time_steps)
        batch_size, features, time_steps = x.shape
        
        # Transpose for attention: (batch, time_steps, features)
        x_t = x.transpose(1, 2)
        
        # Compute attention weights: (batch, time_steps, 1)
        attn_weights = self.attention(x_t)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum: (batch, features)
        pooled = torch.sum(x_t * attn_weights, dim=1)
        
        return pooled, attn_weights.squeeze(-1)


class TransitParameterHead(nn.Module):
    """
    Head network that predicts transit parameters.
    
    Outputs: period, t0, rp_rs, a_rs, b, u1, u2
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.1
    ):
        """
        Initialize parameter head.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dims : list
            Hidden layer dimensions.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Output heads for each parameter
        self.period_head = nn.Linear(in_dim, 1)
        self.t0_head = nn.Linear(in_dim, 1)
        self.rp_rs_head = nn.Linear(in_dim, 1)
        self.a_rs_head = nn.Linear(in_dim, 1)
        self.b_head = nn.Linear(in_dim, 1)
        self.u1_head = nn.Linear(in_dim, 1)
        self.u2_head = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for parameter heads to sensible starting points."""
        # Initialize output weights to small values to start near biases
        for head in [self.period_head, self.t0_head, self.rp_rs_head, 
                    self.a_rs_head, self.b_head, self.u1_head, self.u2_head]:
            nn.init.xavier_uniform_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

        # Better initial biases for realistic starting points
        # Softplus inverse for a starting point: ln(exp(val) - 1)
        with torch.no_grad():
            self.period_head.bias.fill_(1.0) # start around 3 days (softplus(1)+0.5)
            self.rp_rs_head.bias.fill_(-3.0) # start small (~0.01)
            self.a_rs_head.bias.fill_(2.0)   # start around 8 stellar radii
            self.b_head.bias.fill_(0.0)      # center of star
            self.u1_head.bias.fill_(0.0)     # moderate limb darkening
            self.u2_head.bias.fill_(0.0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, features).
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of predicted parameters.
        """
        features = self.feature_layers(x)
        
        # Predict parameters with appropriate activations and safety clamps
        # Period: Softplus ensures > 0, +0.5 for minimum physical period
        period = F.softplus(torch.clamp(self.period_head(features), max=10.0)) + 0.5
        t0 = self.t0_head(features)  # Unconstrained
        
        # Radius ratio: usually < 0.2
        rp_rs = F.sigmoid(self.rp_rs_head(features)) * 0.25
        
        # Semi-major axis: usually > 1
        a_rs = F.softplus(torch.clamp(self.a_rs_head(features), max=20.0)) + 1.1
        
        # Impact parameter: 0 to 1
        b = F.sigmoid(self.b_head(features))
        
        # Limb darkening: 0 to 1
        u1 = F.sigmoid(self.u1_head(features))
        u2 = F.sigmoid(self.u2_head(features))
        
        return {
            'period': period.squeeze(-1),
            't0': t0.squeeze(-1),
            'rp_rs': rp_rs.squeeze(-1),
            'a_rs': a_rs.squeeze(-1),
            'b': b.squeeze(-1),
            'u1': u1.squeeze(-1),
            'u2': u2.squeeze(-1)
        }


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for exoplanet transit detection.
    
    Architecture:
        1. Encoder: Extracts features from light curve
        2. Attention pooling: Aggregates temporal features
        3. Parameter head: Predicts transit parameters
        4. Physics model: Reconstructs light curve (in loss, not forward)
    
    The physics model is applied in the loss function, not in the forward pass.
    This keeps the architecture clean while enforcing physics constraints.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        encoder_dims: list = [64, 128, 256],
        encoder_kernels: list = [5, 5, 5],
        param_head_dims: list = [256, 128, 64],
        dropout: float = 0.1
    ):
        """
        Initialize PINN.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension (time, flux, flux_err).
        encoder_dims : list
            Encoder hidden dimensions.
        encoder_kernels : list
            Encoder kernel sizes.
        param_head_dims : list
            Parameter head hidden dimensions.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        self.encoder = TimeSeriesEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_dims,
            kernel_sizes=encoder_kernels,
            dropout=dropout
        )
        
        self.attention = AttentionPooling(self.encoder.output_dim)
        
        self.param_head = TransitParameterHead(
            input_dim=self.encoder.output_dim,
            hidden_dims=param_head_dims,
            dropout=dropout
        )
    
    def forward(
        self,
        time: torch.Tensor,
        flux: torch.Tensor,
        flux_err: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        time : torch.Tensor
            Time array of shape (batch, time_steps).
        flux : torch.Tensor
            Flux array of shape (batch, time_steps).
        flux_err : torch.Tensor, optional
            Flux errors of shape (batch, time_steps).
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'parameters': Predicted transit parameters
            - 'features': Encoded features (for visualization)
            - 'attention_weights': Attention weights (for interpretability)
        """
        batch_size, time_steps = flux.shape
        
        # Prepare input: (batch, features, time_steps)
        if flux_err is not None:
            x = torch.stack([time, flux, flux_err], dim=1)
        else:
            x = torch.stack([time, flux, torch.zeros_like(flux)], dim=1)
        
        # Encode
        encoded = self.encoder(x)  # (batch, hidden_dim, time_steps)
        
        # Pool with attention
        pooled, attn_weights = self.attention(encoded)  # (batch, hidden_dim)
        
        # Predict parameters
        parameters = self.param_head(pooled)
        
        return {
            'parameters': parameters,
            'features': pooled,
            'attention_weights': attn_weights
        }
    
    def predict_transit_model(
        self,
        time: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        transit_model_fn: callable
    ) -> torch.Tensor:
        """
        Predict transit model using physics.
        
        This is called from the loss function, not the forward pass.
        
        Parameters
        ----------
        time : torch.Tensor
            Time array.
        parameters : Dict[str, torch.Tensor]
            Predicted parameters.
        transit_model_fn : callable
            Function to compute transit model (from physics module).
        
        Returns
        -------
        torch.Tensor
            Predicted flux model.
        """
        # Convert to numpy for physics model (or implement in PyTorch)
        # For now, this is a placeholder - actual implementation depends on
        # whether transit_model is in PyTorch or NumPy
        raise NotImplementedError(
            "This should be implemented in the loss function using "
            "the physics module's transit model."
        )


class LightCurveDataset(Dataset):
    """
    Dataset for light curve training.
    
    Loads standardized light curves and prepares them for training.
    """
    
    def __init__(
        self,
        light_curves: list,
        max_length: Optional[int] = None,
        normalize_time: bool = True
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        light_curves : list
            List of StandardizedLightCurve objects.
        max_length : int, optional
            Maximum sequence length (truncate or pad).
        normalize_time : bool
            Whether to normalize time to [0, 1].
            
        Warning
        -------
        If normalize_time=True, time is normalized to [0, 1] which loses
        absolute time scale. Period predictions must be interpreted in
        ORIGINAL time units using stored normalization metadata.
        """
        self.light_curves = light_curves
        self.max_length = max_length
        self.normalize_time = normalize_time
    
    def __len__(self) -> int:
        return len(self.light_curves)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single light curve.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with 'time', 'flux', 'flux_err', 'metadata'.
        """
        lc = self.light_curves[idx]
        
        time = torch.from_numpy(np.ascontiguousarray(lc.time.astype(np.float32)))
        flux = torch.from_numpy(np.ascontiguousarray(lc.flux.astype(np.float32)))
        
        if lc.flux_err is not None:
            flux_err_np = np.ascontiguousarray(lc.flux_err.astype(np.float32))
        else:
            flux_err_np = np.ascontiguousarray((np.ones_like(lc.flux) * np.nanstd(lc.flux)).astype(np.float32))
        flux_err = torch.from_numpy(flux_err_np)
        
        # IMPORTANT: Time normalization handling
        # If normalize_time=True, time is scaled to [0, 1] which loses absolute scale.
        # Any period predictions must be interpreted in ORIGINAL time units.
        # Downstream code must inverse-transform using stored metadata.
        time_normalization = None
        if self.normalize_time and len(time) > 0:
            time_min = time.min()
            time_max = time.max()
            time_range = time_max - time_min
            
            if time_range <= 0:
                raise ValueError(
                    f"Time array has zero range (min={time_min}, max={time_max}); "
                    "cannot normalize. Check for duplicate time values."
                )
            
            time = (time - time_min) / time_range
            
            # Store normalization parameters for inverse transform
            time_normalization = {
                "min": float(time_min.item()),
                "max": float(time_max.item()),
                "range": float(time_range.item())
            }
        
        # Pad or truncate to max_length
        if self.max_length is not None:
            if len(time) > self.max_length:
                # Truncate
                time = time[:self.max_length]
                flux = flux[:self.max_length]
                flux_err = flux_err[:self.max_length]
            elif len(time) < self.max_length:
                # Pad with last value
                pad_length = self.max_length - len(time)
                time = F.pad(time, (0, pad_length), mode='constant', value=time[-1])
                flux = F.pad(flux, (0, pad_length), mode='constant', value=flux[-1])
                flux_err = F.pad(flux_err, (0, pad_length), mode='constant', value=flux_err[-1])
        
        # Update metadata with normalization info
        metadata = lc.metadata.copy() if lc.metadata else {}
        if time_normalization is not None:
            metadata['time_normalization'] = time_normalization
        
        return {
            'time': time,
            'flux': flux,
            'flux_err': flux_err,
            'metadata': metadata
        }

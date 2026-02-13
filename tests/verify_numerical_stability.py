import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.pinn_losses import PhysicsInformedLoss, mandel_agol_transit_torch
import numpy as np

def test_stability():
    print("Testing numerical stability...")
    device = torch.device('cpu')
    loss_fn = PhysicsInformedLoss()
    
    # 1. Test with extreme parameters
    parameters = {
        'period': torch.tensor([1e6], device=device), # Huge period
        't0': torch.tensor([0.0], device=device),
        'rp_rs': torch.tensor([10.0], device=device),  # Huge radius ratio
        'a_rs': torch.tensor([1e6], device=device),   # Huge separation
        'b': torch.tensor([10.0], device=device),      # Huge impact parameter
        'u1': torch.tensor([0.5], device=device),
        'u2': torch.tensor([0.5], device=device)
    }
    
    time = torch.linspace(0, 30, 1000, device=device)
    target_flux = torch.ones_like(time)
    flux_err = torch.ones_like(time) * 1e-6 # Very small error
    
    # Step 1: Check transit model
    pred_flux = mandel_agol_transit_torch(
        time=time,
        period=parameters['period'],
        t0=parameters['t0'],
        rp_rs=parameters['rp_rs'],
        a_rs=parameters['a_rs'],
        b=parameters['b']
    )
    print(f"Transit model output range: [{pred_flux.min().item():.3f}, {pred_flux.max().item():.3f}]")
    assert torch.isfinite(pred_flux).all(), "Transit model produced non-finite values!"

    # Step 2: Check loss function
    loss_dict = loss_fn(
        predicted_flux=pred_flux,
        target_flux=target_flux,
        parameters=parameters,
        time=time,
        flux_err=flux_err
    )
    
    print(f"Total Loss: {loss_dict['total_loss'].item():.3f}")
    assert torch.isfinite(loss_dict['total_loss']), "Total loss is not finite!"
    
    # Check components
    for key, val in loss_dict['physics_losses'].items():
        print(f"Component {key}: {val.item():.3f}")
        assert torch.isfinite(val), f"Loss component {key} is not finite!"

    # 3. Test with zero parameters (division by zero test)
    parameters_zero = {k: torch.zeros_like(v) for k, v in parameters.items()}
    pred_flux_zero = mandel_agol_transit_torch(
        time=time,
        period=parameters_zero['period'],
        t0=parameters_zero['t0'],
        rp_rs=parameters_zero['rp_rs'],
        a_rs=parameters_zero['a_rs'],
        b=parameters_zero['b']
    )
    loss_dict_zero = loss_fn(
        predicted_flux=pred_flux_zero,
        target_flux=target_flux,
        parameters=parameters_zero,
        time=time,
        flux_err=flux_err
    )
    print(f"Zero parameter Loss: {loss_dict_zero['total_loss'].item():.3f}")
    assert torch.isfinite(loss_dict_zero['total_loss']), "Loss with zero parameters is not finite!"

    print("✅ All stability tests passed!")

if __name__ == "__main__":
    test_stability()

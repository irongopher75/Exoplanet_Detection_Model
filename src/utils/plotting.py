import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any

def plot_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    model_flux: Optional[np.ndarray] = None,
    title: str = "Light Curve",
    save_path: Optional[Path] = None
):
    """Plot light curve with optional model overlay."""
    plt.figure(figsize=(10, 6))
    plt.plot(time, flux, 'k.', markersize=2, alpha=0.5, label='Observed')
    
    if model_flux is not None:
        plt.plot(time, model_flux, 'r-', linewidth=1.5, alpha=0.8, label='Model')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_curves(
    losses: Dict[str, list],
    save_path: Optional[Path] = None
):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    
    for name, values in losses.items():
        plt.plot(values, label=name)
        
    plt.title("Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

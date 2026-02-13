import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

class AdaptiveLossWeighting:
    """
    Adaptive loss weighting for PINNs.
    
    Implements a gradient-balancing strategy where weights are adjusted 
    periodically based on the relative magnitudes of the gradients of 
    different loss terms.
    """
    
    def __init__(
        self,
        loss_keys: List[str],
        init_weights: Optional[Dict[str, float]] = None,
        alpha: float = 0.9,  # Momentum for weight updates
        update_every: int = 1
    ):
        self.loss_keys = loss_keys
        self.weights = {k: 1.0 for k in loss_keys}
        if init_weights:
            self.weights.update(init_weights)
            
        self.alpha = alpha
        self.update_every = update_every
        self.step_count = 0
        
        # Track gradient magnitudes
        self.grad_norms = {k: 0.0 for k in loss_keys}

    def update_weights(self, loss_dict: Dict[str, torch.Tensor], model: nn.Module):
        """
        Update weights based on gradient magnitudes.
        
        This should be called AFTER the backward passes for individual terms
        but BEFORE the optimizer step.
        """
        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return
            
        # 1. Compute gradients for each loss term separately
        current_norms = {}
        for key in self.loss_keys:
            if key not in loss_dict:
                continue
                
            loss = loss_dict[key]
            if not isinstance(loss, torch.Tensor) or loss.grad_fn is None:
                continue
                
            model.zero_grad()
            loss.backward(retain_graph=True)
            
            # Compute total gradient norm for this term
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            current_norms[key] = total_norm ** 0.5
            
        # 2. Update running average of norms and calculate new weights
        # Strategy: Inverse temperature / balance norms
        if not current_norms:
            return
            
        # Filter for finite values
        finite_norms = {k: v for k, v in current_norms.items() if np.isfinite(v) and v > 0}
        if not finite_norms:
            return

        target_norm = sum(finite_norms.values()) / len(finite_norms)
        if target_norm == 0 or not np.isfinite(target_norm):
            return

        for key in finite_norms:
            new_weight = target_norm / (finite_norms[key] + 1e-8)
            # Clip weight update to prevent extreme shifts (more conservative [0.1, 10])
            new_weight = min(max(new_weight, 0.1), 10.0)
            # Smooth update
            self.weights[key] = self.alpha * self.weights[key] + (1 - self.alpha) * new_weight

    def apply_weights(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply current weights to the loss terms and return total loss."""
        total_loss = 0.0
        for key, loss in loss_dict.items():
            if key in self.weights:
                total_loss += self.weights[key] * loss
            else:
                total_loss += loss # Default weight 1.0 for untracked terms
        return total_loss
